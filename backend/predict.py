# usage:
# python predict.py model.json file --data_path input.xlsx --output_name result
# python predict.py model.json input --input_values 1.0 TRUE 3.0 FALSE
# read_csv 和 read_excel 會自行轉換數值型 (int 或 float)和布林型 (bool)

import argparse
from tool_train import NumpyEncoder, extract_base64_images_and_clean_json
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os
import json
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import sys
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
try:
    # 嘗試使用 Docker 中的 NotoSansCJK 字型
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()

except Exception:
    # Fallback：改用本機字型清單
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']

# 顯示負號正常
matplotlib.rcParams['axes.unicode_minus'] = False
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import zipfile
import tempfile
import shutil
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC



def _fix_xgb_n_classes(model):
    """XGBoost 2.x workaround：load_model 之後有時不會還原 n_classes_"""
    if not hasattr(model, "n_classes_"):
        try:
            booster = model.get_booster()
            config = json.loads(booster.save_config())
            num_class = int(config.get("learner", {}).get("learner_model_param", {}).get("num_class", 2))
            # XGBoost binary 也可能回傳 num_class=0，代表 2 類
            model.n_classes_ = max(num_class, 2)
        except Exception:
            model.n_classes_ = 2
    return model

def load_model(model_path):
    if model_path.lower().endswith(".json"):  # .json 結尾
        model = XGBClassifier()
        model.load_model(model_path)
        model = _fix_xgb_n_classes(model)
    elif model_path.lower().endswith(".pkl"):
        model = joblib.load(model_path)
    elif model_path.lower().endswith(".zip"):
        # 修改：區分 TabNet zip 和訓練結果 zip
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # 檢查是否為 stacking zip
                if "stacking_config.json" in file_list:
                    # stacking 模型由 predict_stacking 處理，這裡不應該進來
                    raise ValueError("Stacking model should be handled by predict_stacking()")
                
                # ✅ 新增：檢查是否為訓練結果 zip（包含 .json 模型檔案）
                excluded_jsons = {'metrics.json', 'status.json', 'model_params.json', 'stacking_config.json'}
                json_model_files = [f for f in file_list if f.endswith('.json') and f not in excluded_jsons]
                if json_model_files:
                    # 解壓到暫存目錄並載入 XGBoost 模型
                    temp_dir = tempfile.mkdtemp()
                    try:
                        zip_ref.extractall(temp_dir)
                        json_path = os.path.join(temp_dir, json_model_files[0])
                        model = XGBClassifier()
                        model.load_model(json_path)
                        model = _fix_xgb_n_classes(model)
                        return model
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                
                # 檢查是否為訓練結果 zip（包含 .pkl 檔案）
                pkl_files = [f for f in file_list if f.endswith('.pkl')]
                if pkl_files:
                    # 解壓到暫存目錄並載入 pkl 模型
                    temp_dir = tempfile.mkdtemp()
                    try:
                        zip_ref.extractall(temp_dir)
                        pkl_path = os.path.join(temp_dir, pkl_files[0])
                        model = joblib.load(pkl_path)
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return model
                
                # 檢查是否為 TabNet 模型 zip（包含 model_params.json 等）
                if "model_params.json" in file_list or any('network' in f for f in file_list):
                    model = TabNetClassifier()
                    model.load_model(model_path)
                    return model
                
                # 檢查是否有內嵌的 .zip（可能是 TabNet）
                inner_zips = [f for f in file_list if f.endswith('.zip')]
                if inner_zips:
                    temp_dir = tempfile.mkdtemp()
                    try:
                        zip_ref.extractall(temp_dir)
                        inner_zip_path = os.path.join(temp_dir, inner_zips[0])
                        model = TabNetClassifier()
                        model.load_model(inner_zip_path)
                        return model
                    except Exception:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                
                # 都不是，嘗試當作 TabNet 載入
                model = TabNetClassifier()
                model.load_model(model_path)
                
        except zipfile.BadZipFile:
            print(json.dumps({
                "status": "error",
                "message": f"Invalid zip file: {model_path}",
            }))
            sys.exit(1)
    else:
        print(json.dumps({
            "status": "error",
            "message": f"Unsupported model format: {model_path}",
        }))
        sys.exit(1)
    return model


def load_data(data_path):
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path)
    # print(f"Data loaded from {data_path}")
    return data

def explain_with_shap(model, x_test, feature_names, focus_class=1):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        class_name = model.__class__.__name__.lower()
        is_tabnet_or_tabpfn = (
            hasattr(model, "predict_proba") and
            not hasattr(model, "predict")  # TabNet 沒有原生 predict，只有 predict_proba，把 TabPFN 也視為「用 KernelExplainer + predict_proba」的類型
        )\
        or class_name.startswith("tabnet")\
        or class_name.startswith("tabpfn")  # 新增：TabPFN

        if is_tabnet_or_tabpfn:
            # TabNet / TabPFN 共用：KernelExplainer + predict_proba
            background = x_test[:20]
            explain_data = x_test[:5]

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(explain_data)
            shap_data = explain_data

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values_for_plot = shap_values[:, :, int(focus_class)]
            elif isinstance(shap_values, list) and len(shap_values) >= 2:
                shap_values_for_plot = shap_values[int(focus_class)]
            else:
                shap_values_for_plot = shap_values

            if shap_values_for_plot.ndim != 2:
                raise ValueError(f"shap_values shape is invalid: {shap_values_for_plot.shape}")
        
        elif isinstance(model, Pipeline) and "sgdclassifier" in model.named_steps:
            estimator = model.named_steps["sgdclassifier"]
            scaler = model.named_steps["standardscaler"]
            x_scaled = scaler.transform(x_test)

            explainer = shap.LinearExplainer(estimator, x_scaled, feature_perturbation="interventional")
            shap_values = explainer.shap_values(x_scaled)
            shap_data = x_scaled

            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[int(focus_class)]
            else:
                shap_values_for_plot = shap_values

        elif isinstance(model, (MLPClassifier, AdaBoostClassifier)):
            # 使用 lambda 包裝 MLPClassifier，避免 KernelExplainer 報錯
            background = x_test[:20]
            explain_data = x_test[:5]

            predict_fn = lambda x: model.predict_proba(x)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(explain_data)
            shap_data = explain_data

            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[int(focus_class)]
            else:
                shap_values_for_plot = shap_values

        elif isinstance(model, SVC):
            background = x_test[:20]
            explain_data = x_test[:5]

            predict_fn = lambda x: model.predict_proba(x)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(explain_data)
            shap_data = explain_data

            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[int(focus_class)]
            else:
                shap_values_for_plot = shap_values

        else:
            # CatBoost 多類別模型帶背景資料時，shap 解析樹會發生記憶體損毀而使行程崩潰，
            # 因此與 train_catboost.py 相同，不帶背景資料（path-dependent TreeExplainer）
            if type(model).__name__.startswith("CatBoost"):
                explainer = shap.Explainer(model)
            else:
                explainer = shap.Explainer(model, x_test)
            shap_values = explainer(x_test)
            shap_data = x_test

            if hasattr(shap_values, "values"):
                raw_shap = shap_values.values
            else:
                raw_shap = shap_values

            if isinstance(raw_shap, list):
                shap_values_for_plot = raw_shap[int(focus_class)]
            elif raw_shap.ndim == 3:
                shap_values_for_plot = raw_shap[:, :, int(focus_class)]
            else:
                shap_values_for_plot = raw_shap

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, shap_data, feature_names=feature_names, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_abs_shap
        }).sort_values(by='Importance', ascending=False)

        result["shap_plot"] = shap_base64
        result["shap_importance"] = importance_df.to_dict(orient='records')

    except Exception as e:
        result["shap_error"] = str(e)

    return result


def explain_with_lime(model, x_test, y_pred, feature_names, focus_class=1):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        y_pred = np.array(y_pred)

        proba0 = model.predict_proba(x_test[:1])[0]
        num_class = int(len(proba0))
        class_names = [f"Class {i}" for i in range(num_class)]

        if num_class == 2:
            focus_class = 1
        else:
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= num_class:
                focus_class = int(np.argmax(proba0))

        explainer = LimeTabularExplainer(
            training_data=x_test,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )

        # 取第一筆作為範例說明
        explanation = explainer.explain_instance(
            x_test[0],
            model.predict_proba,
            num_features=len(feature_names),
            labels=[focus_class]
        )

        fig = explanation.as_pyplot_figure(label=focus_class)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        lime_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        result["lime_plot"] = lime_base64
        result["lime_example_0"] = explanation.as_list(label=focus_class)

    except Exception as e:
        result["lime_error"] = str(e)
    return result


def save_predictions(data, data_path, output_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)
    output_path = os.path.join(task_dir, output_name)

    if data_path.endswith(".csv"):
        data.to_csv(output_path + ".csv", index=False)
    elif data_path.endswith(".xlsx"):
        data.to_excel(output_path + ".xlsx", index=False)


def predict_input(model, model_path, input_values, task_dir):
    # 將輸入值解析為適當的型態
    parsed_values = []
    for value in input_values:
        if isinstance(value, str):
            if value.upper() == "TRUE":
                parsed_values.append(1)
            elif value.upper() == "FALSE":
                parsed_values.append(0)
            else:
                try:
                    parsed_values.append(float(value))
                except ValueError:
                    raise ValueError(f"無法解析輸入值：{value}，請確認格式正確！")
        else:
            parsed_values.append(float(value))
    input_df = pd.DataFrame([parsed_values])
    model_class_name = model.__class__.__name__.lower()
    if isinstance(model, TabNetClassifier) or model_class_name.startswith("tabpfn"):
        prediction = model.predict(input_df.to_numpy(dtype=np.float32))
    else:
        prediction = model.predict(input_df)
    
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    
    return {
        "status": "success",
        "prediction": prediction,
    }


def predict_single_model(model_path, mode, task_dir, data_path=None, label_column=None, output_name=None, input_values=None, focus_class=1):
    try:
        model = load_model(model_path)
        if mode == "file":
            result = {
                "status": "success",
            }
            data = load_data(data_path)
            feature_names = data.columns.tolist()
            x_test = data.copy()
            model_class_name = model.__class__.__name__.lower()
            if isinstance(model, TabNetClassifier) or model_class_name.startswith("tabpfn"):
                prediction = model.predict(x_test.to_numpy(dtype=np.float32))
            else:
                prediction = model.predict(x_test)
            
            data[label_column] = prediction
            # 儲存預測結果
            save_predictions(data, data_path, output_name, task_dir)

            explanations = {}
            try:
                shap_result = explain_with_shap(model, x_test.values, feature_names, focus_class=focus_class)
                explanations["shap"] = shap_result
            except Exception as e:
                explanations["shap"] = {"shap_error": str(e)}

            try:
                lime_result = explain_with_lime(model, x_test.values, prediction, feature_names, focus_class=focus_class)
                explanations["lime"] = lime_result
            except Exception as e:
                explanations["lime"] = {"lime_error": str(e)}

            result = {
                "status": "success",
            }
            shap_result = explanations.get("shap", {})
            if shap_result.get("shap_plot"):
                result["shap_plot"] = shap_result["shap_plot"]
            if shap_result.get("shap_importance"):
                result["shap_importance"] = shap_result["shap_importance"]
            lime_result = explanations.get("lime", {})
            if lime_result.get("lime_plot"):
                result["lime_plot"] = lime_result["lime_plot"]
            if lime_result.get("lime_example_0"):
                result["lime_example_0"] = lime_result["lime_example_0"]
        elif mode == "input":
            result = predict_input(model, model_path, input_values, task_dir)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"{e}"
        }


def infer_stacking_config(temp_dir):
    """從檔案名稱推斷 stacking 配置（當沒有 stacking_config.json 時）"""
    files = os.listdir(temp_dir)
    
    # 找出所有 base models
    base_models = []
    model_name_map = {
        'lgbm': 'lgbm',
        'xgb': 'xgb', 
        'rf': 'rf',
        'svm': 'svm',
        'tabnet': 'tabnet',
        'tabpfn': 'tabpfn',
        'logistic_regression': 'logistic_regression',
        'lr': 'logistic_regression',
    }
    
    # ✅ 修正：只處理模型檔案，排除圖片
    model_extensions = ('.pkl', '.json', '.zip')
    
    for f in files:
        # 排除 base_results_* 這類圖片檔案
        if f.startswith('base_') and not f.startswith('base_results_') and f.endswith(model_extensions):
            # 從檔名解析模型類型，例如 base_lgbm.pkl -> lgbm
            name_without_ext = os.path.splitext(f)[0]  # base_lgbm
            model_key = name_without_ext.replace('base_', '')  # lgbm
            
            # 標準化模型名稱
            if model_key in model_name_map:
                base_models.append(model_key)
            else:
                base_models.append(model_key)
    
    # 找出 meta model
    meta_model = None
    for f in files:
        # ✅ 修正：排除 meta_results_* 這類圖片檔案
        if f.startswith('meta_') and not f.startswith('meta_results_') and f.endswith(model_extensions):
            name_without_ext = os.path.splitext(f)[0]  # meta_xgb
            model_key = name_without_ext.replace('meta_', '')  # xgb
            meta_model = model_key
            break
    
    if not base_models:
        raise ValueError("No base models found in stacking ZIP")
    if not meta_model:
        raise ValueError("No meta model found in stacking ZIP")
    
    return {
        "base_models": base_models,
        "meta_model": meta_model
    }


def predict_stacking(model_path, mode, task_dir, data_path=None, label_column=None, output_name=None, input_values=None, focus_class=1):
    result = {
        "status": "success",
    }
    temp_dir = tempfile.mkdtemp()
    try:
        # 解壓 ZIP 到 temp 目錄
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 嘗試讀取 config，如果不存在則推斷
        config_path = os.path.join(temp_dir, "stacking_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            # 沒有 config，從檔案名稱推斷
            config = infer_stacking_config(temp_dir)
        
        # 載入 base models
        base_models = []
        for base_name in config.get("base_models", []):
            if base_name == "xgb":
                filename = f"base_{base_name}.json"
            elif base_name == "tabnet":
                filename = f"base_{base_name}.zip"
            else:
                filename = f"base_{base_name}.pkl"
            base_model_path = os.path.join(temp_dir, filename)
            model = load_model(base_model_path)
            base_models.append(model)
        
        # 載入 meta model
        meta_name = config.get("meta_model")
        if meta_name == "xgb":
            meta_file = f"meta_{meta_name}.json"
        elif meta_name == "tabnet":
            meta_file = f"meta_{meta_name}.zip"
        else:
            meta_file = f"meta_{meta_name}.pkl"
        meta_model_path = os.path.join(temp_dir, meta_file)
        meta_model = load_model(meta_model_path)
        
        # 檔案預測模式
        if mode == "file":
            data = load_data(data_path)
            feature_names = data.columns.tolist()
            x_test = data.copy()
            
            # 轉換成 meta features：每個 base model 的 predict_proba[:, focus_class]
            base_preds = []
            for model in base_models:
                model_class_name = model.__class__.__name__.lower()
                if isinstance(model, TabNetClassifier) or model_class_name.startswith("tabpfn"):
                    proba = model.predict_proba(x_test.to_numpy(dtype=np.float32))[:, int(focus_class)]
                else:
                    proba = model.predict_proba(x_test.values)[:, int(focus_class)]
                base_preds.append(proba)
            
            X_meta = np.column_stack(base_preds)
            meta_feature_df = pd.DataFrame(X_meta, columns=config.get("base_models", []))
            save_predictions(meta_feature_df, data_path, "meta_feature", task_dir)
            
            # 最終預測
            y_pred = meta_model.predict(X_meta)
            data[label_column] = y_pred
            
            # 儲存預測結果
            save_predictions(data, data_path, output_name, task_dir)

            # 評估（meta model 的 focus_class 固定 1，因為 meta feature 已經是 binary score）
            explanations = {}
            try:
                shap_result = explain_with_shap(meta_model, X_meta, config.get("base_models", []), focus_class=1)
                explanations["shap"] = shap_result
            except Exception as e:
                explanations["shap"] = {"shap_error": str(e)}

            try:
                lime_result = explain_with_lime(meta_model, X_meta, y_pred, config.get("base_models", []), focus_class=1)
                explanations["lime"] = lime_result
            except Exception as e:
                explanations["lime"] = {"lime_error": str(e)}

            shap_result = explanations.get("shap", {})
            if shap_result.get("shap_plot"):
                result["shap_plot"] = shap_result["shap_plot"]
            if shap_result.get("shap_importance"):
                result["shap_importance"] = shap_result["shap_importance"]
            lime_result = explanations.get("lime", {})
            if lime_result.get("lime_plot"):
                result["lime_plot"] = lime_result["lime_plot"]
            if lime_result.get("lime_example_0"):
                result["lime_example_0"] = lime_result["lime_example_0"]
        
        # 手動輸入預測模式
        else:
            # 將輸入值解析為適當的型態
            parsed_values = []
            for value in input_values:
                if isinstance(value, str):
                    if value.upper() == "TRUE":
                        parsed_values.append(1)
                    elif value.upper() == "FALSE":
                        parsed_values.append(0)
                    else:
                        try:
                            parsed_values.append(float(value))
                        except ValueError:
                            raise ValueError(f"無法解析輸入值：{value}，請確認格式正確！")
                else:
                    parsed_values.append(float(value))
            
            x_test = pd.DataFrame([parsed_values])
            
            # 轉換成 meta features
            base_preds = []
            for model in base_models:
                model_class_name = model.__class__.__name__.lower()
                if isinstance(model, TabNetClassifier) or model_class_name.startswith("tabpfn"):
                    proba = model.predict_proba(x_test.to_numpy(dtype=np.float32))[:, int(focus_class)]
                else:
                    proba = model.predict_proba(x_test.values)[:, int(focus_class)]
                base_preds.append(proba)

            X_meta = np.column_stack(base_preds)
            meta_feature_df = pd.DataFrame(X_meta, columns=config.get("base_models", []))
            save_predictions(meta_feature_df, ".csv", "meta_feature", task_dir)
            
            # 最終預測
            prediction = meta_model.predict(X_meta)
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            result = {
                "status": "success",
                "prediction": prediction,
            }
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"{e}"
        }
    finally:
        shutil.rmtree(temp_dir)  # 清除暫存資料夾


def is_stacking_zip(file_list):
    """判斷 ZIP 是否為 Stacking 模型"""
    # 方法 1：有 stacking_config.json
    if "stacking_config.json" in file_list:
        return True
    
    # 方法 2：有 base_* 和 meta_* 模型檔案（排除圖片）
    model_extensions = ('.pkl', '.json', '.zip')
    has_base = any(
        f.startswith('base_') 
        and not f.startswith('base_results_')
        and f.endswith(model_extensions) 
        for f in file_list
    )
    has_meta = any(
        f.startswith('meta_') 
        and not f.startswith('meta_results_')
        and f.endswith(model_extensions) 
        for f in file_list
    )
    
    return has_base and has_meta


def main(model_path, mode, task_dir, data_path=None, output_name=None, input_values=None, label_column="label"):
    try:
        focus_class = 1  # 預設

        if model_path.endswith("zip"):
            try:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    
                    # 改進的判斷邏輯：不只檢查 config，也檢查 base_/meta_ 檔案
                    if is_stacking_zip(file_list):
                        # 是 stacking zip
                        results = predict_stacking(model_path, mode, task_dir, data_path, label_column, output_name, input_values, focus_class=focus_class)
                    else:
                        # 是單一模型（TabNet 或訓練結果）
                        results = predict_single_model(model_path, mode, task_dir, data_path, label_column, output_name, input_values, focus_class=focus_class)
            except zipfile.BadZipFile:
                print(json.dumps({
                    "status": "error",
                    "message": f"Invalid zip file: {model_path}",
                }))
                return
        else:
            results = predict_single_model(model_path, mode, task_dir, data_path, label_column, output_name, input_values, focus_class=focus_class)
        
        results["task_dir"] = task_dir
        os.makedirs(task_dir, exist_ok=True)  
        result_json_path = os.path.join(task_dir, "metrics.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        extract_base64_images_and_clean_json(task_dir, "metrics.json")
        print(json.dumps(results))
            
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the trained model file (e.g., .json for XGBoost or .pkl for joblib models)")
    parser.add_argument('mode', type=str, choices=["file", "input"], help="模式選擇：file（檔案模式）或 input（手動輸入模式）")
    parser.add_argument("task_dir", type=str)
    parser.add_argument('--data_path', type=str, help="輸入檔案路徑（僅 file 模式需要）")
    parser.add_argument('--output_name', type=str, help="輸出檔案名稱（僅 file 模式需要）")
    parser.add_argument('--input_values', type=str, nargs='+', help="輸入特徵數值（僅 input 模式需要，數值間以空格分隔）")
    parser.add_argument('--label_column', type=str, help="自訂標籤欄位名稱")
    
    args = parser.parse_args()
    main(args.model_path, args.mode, args.task_dir, args.data_path, args.output_name, args.input_values, args.label_column)