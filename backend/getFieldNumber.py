# 取得模型需要的欄位數
# usage: python getFieldNumber.py <model_path>
# ex:
#   python getFieldNumber.py upload/model.pkl

import sys
import os
import json
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import zipfile
import tempfile
import warnings

# 抑制 TabNet 的警告訊息，避免干擾 JSON 輸出
warnings.filterwarnings("ignore", category=UserWarning)


def get_field_number_from_training_result_zip(zip_path):
    """處理訓練結果 zip（包含 .json、.pkl 或內嵌 TabNet .zip）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # 排除的非模型 JSON 檔案
        excluded_jsons = {'metrics.json', 'status.json', 'model_params.json', 'stacking_config.json'}

        # 1. 先找 .json 檔案（XGBoost 模型）
        for filename in os.listdir(tmpdir):
            if filename.endswith('.json') and filename not in excluded_jsons:
                json_path = os.path.join(tmpdir, filename)
                try:
                    xgb_model = XGBClassifier()
                    xgb_model.load_model(json_path)
                    return xgb_model.n_features_in_
                except Exception:
                    continue  # 不是 XGBoost 模型，嘗試下一個

        # 2. 找 .pkl 檔案（sklearn, LightGBM, TabPFN 等）
        for filename in os.listdir(tmpdir):
            if filename.endswith('.pkl'):
                pkl_path = os.path.join(tmpdir, filename)
                try:
                    model = joblib.load(pkl_path)
                    if hasattr(model, "n_features_in_"):
                        return model.n_features_in_
                    if hasattr(model, "feature_names_in_"):
                        return len(model.feature_names_in_)
                except Exception:
                    continue
                # TabPFN 可能沒有這些屬性，繼續嘗試

        # 3. 找內嵌的 .zip（TabNet 模型）
        for filename in os.listdir(tmpdir):
            if filename.endswith('.zip'):
                inner_zip_path = os.path.join(tmpdir, filename)
                try:
                    tabnet_model = TabNetClassifier()
                    tabnet_model.load_model(inner_zip_path)
                    return tabnet_model.network.input_dim
                except Exception:
                    continue  # 不是 TabNet，嘗試下一個

        return None


def get_field_number_from_stacking_zip(zip_path):
    """處理 stacking 模型的 zip 檔案"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        files_in_zip = os.listdir(tmpdir)
        
        # 方法 1：有 stacking_config.json（標準格式）
        config_path = os.path.join(tmpdir, "stacking_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            # 優先從 config 讀取 n_features
            if "n_features" in config and config["n_features"] is not None:
                return config["n_features"]

            # 嘗試按 config 指定的 base model 順序
            base_model_paths = config.get("base_models", [])
            if base_model_paths:
                for raw_path in base_model_paths:
                    if raw_path == "xgb":
                        candidate = os.path.join(tmpdir, f"base_{raw_path}.json")
                    elif raw_path == "tabnet":
                        candidate = os.path.join(tmpdir, f"base_{raw_path}.zip")
                    else:
                        candidate = os.path.join(tmpdir, f"base_{raw_path}.pkl")
                    
                    if os.path.exists(candidate):
                        try:
                            return get_field_number(candidate)
                        except Exception:
                            continue
            # 如果 config 方式失敗，繼續嘗試方法 2

        # 方法 2：掃描 base_*.pkl / base_*.json / base_*.zip，按優先級排序
        # 注意：排除 base_results_* 這類圖片檔案
        model_extensions = ('.pkl', '.json', '.zip')
        base_files = [
            f for f in files_in_zip 
            if f.startswith('base_') 
            and not f.startswith('base_results_')  # 排除結果圖片
            and f.endswith(model_extensions)
        ]
        
        # 排序：優先嘗試最可能有 n_features_in_ 的模型，TabPFN 放最後
        def model_priority(filename):
            # 數字越小越優先
            priority_order = ['lgbm', 'lightgbm', 'xgb', 'rf', 'svm', 'logistic', 'tabnet', 'tabpfn']
            name_lower = filename.lower()
            for i, key in enumerate(priority_order):
                if key in name_lower:
                    return i
            return 100  # 未知模型放最後
        
        base_files.sort(key=model_priority)
        
        if base_files:
            errors = []
            for base_file in base_files:
                base_path = os.path.join(tmpdir, base_file)
                try:
                    return get_field_number(base_path)
                except Exception as e:
                    errors.append(f"{base_file}: {e}")
                    continue
            
            raise ValueError(f"Found base models but all failed: {errors}")

        # 不是 stacking ZIP
        raise ValueError("Not a stacking ZIP (no stacking_config.json or base_* model files)")


def get_field_number(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' doesn't exist!")

    file_extension = os.path.splitext(model_path)[-1].lower()

    # 讀取 XGBoost 模型（JSON 格式）
    if file_extension == ".json":
        xgb_model = XGBClassifier()
        xgb_model.load_model(model_path)
        return xgb_model.n_features_in_

    # 處理 .zip 檔案（TabNet、Stacking、或訓練結果）
    elif file_extension == ".zip":
        last_error = None
        
        # 先檢查 ZIP 內容來決定處理方式
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {model_path}")

        # 1. 檢查是否為 stacking ZIP
        # 條件：有 stacking_config.json，或同時有 base_* 和 meta_* 模型檔案
        has_stacking_config = "stacking_config.json" in file_list
        
        # 檢查是否有 base 和 meta 模型檔案（排除圖片等非模型檔案）
        model_extensions = ('.pkl', '.json', '.zip')
        has_base_model = any(
            f.startswith('base_') 
            and not f.startswith('base_results_')
            and f.endswith(model_extensions) 
            for f in file_list
        )
        has_meta_model = any(
            f.startswith('meta_') 
            and not f.startswith('meta_results_')
            and f.endswith(model_extensions) 
            for f in file_list
        )
        
        is_stacking = has_stacking_config or (has_base_model and has_meta_model)
        
        if is_stacking:
            try:
                return get_field_number_from_stacking_zip(model_path)
            except ValueError as e:
                last_error = f"Stacking: {e}"
            except Exception as e:
                last_error = f"Stacking: {e}"

        # 2. 檢查是否為純 TabNet ZIP（有 model_params.json）
        if "model_params.json" in file_list:
            try:
                tabnet_model = TabNetClassifier()
                tabnet_model.load_model(model_path)
                return tabnet_model.network.input_dim
            except Exception as e:
                last_error = f"TabNet: {e}"

        # 3. 檢查是否為訓練結果 ZIP（有 .json、.pkl 或內嵌 .zip）
        excluded_jsons = {'metrics.json', 'status.json', 'model_params.json', 'stacking_config.json'}
        has_model_json = any(f.endswith('.json') and f not in excluded_jsons for f in file_list)
        has_pkl = any(f.endswith('.pkl') for f in file_list)
        has_inner_zip = any(f.endswith('.zip') for f in file_list)
        
        if has_model_json or has_pkl or has_inner_zip:
            try:
                result = get_field_number_from_training_result_zip(model_path)
                if result is not None:
                    return result
                else:
                    last_error = "Training ZIP: model has no n_features_in_"
            except Exception as e:
                last_error = f"Training ZIP: {e}"

        # 4. 最後嘗試當作 TabNet 載入（fallback）
        try:
            tabnet_model = TabNetClassifier()
            tabnet_model.load_model(model_path)
            return tabnet_model.network.input_dim
        except Exception as e:
            if last_error is None:
                last_error = f"TabNet fallback: {e}"

        # 都失敗
        raise ValueError(f"Cannot determine field number from ZIP. Last error: {last_error}")

    # 讀取其他模型（PKL 格式）
    elif file_extension == ".pkl":
        model = joblib.load(model_path)

        # 標準 sklearn 屬性
        if hasattr(model, "n_features_in_"):
            return model.n_features_in_
        if hasattr(model, "feature_names_in_"):
            return len(model.feature_names_in_)
        
        # TabPFN 特殊處理
        model_type = type(model).__name__
        if "tabpfn" in model_type.lower():
            if hasattr(model, "n_features_"):
                return model.n_features_
            if hasattr(model, "_n_features"):
                return model._n_features
            if hasattr(model, "X_train_") and model.X_train_ is not None:
                return model.X_train_.shape[1]
            raise ValueError(f"TabPFN model does not have n_features attribute. "
                           f"Please use file mode with CSV for prediction.")
        
        raise ValueError(f"Model ({model_type}) does not have 'n_features_in_' attribute.")

    else:
        raise ValueError(f"Unsupported format: {file_extension}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python getFieldNumber.py <model_path>.",
        }))
        sys.exit(1)

    model_path = sys.argv[1]
    
    try:
        field_count = get_field_number(model_path)
        print(json.dumps({
            "status": "success",
            "field_count": field_count
        }))
    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown error occurred"
        print(json.dumps({
            "status": "error",
            "message": error_msg,
        }))
        sys.exit(1)