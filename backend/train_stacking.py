# usage: python train_stacking.py upload/alice/長庚csv有答案.csv "[\"xgb\", \"lgbm\"]" label lgbm model/alice/20250721112233
# step 2: 每個 base model 在各 fold 上進行訓練與驗證
# step 3: 對驗證集產生預測，這些預測會組成 meta model 的輸入特徵（稱為 OOF 特徵）
# step 4: meta model 使用 OOF 特徵進行訓練，輸出結果（準確率等）記錄在 results["meta_results"] 中
# step 5: 每個 base model 重新使用全體資料進行訓練，並儲存模型，輸出結果（準確率等）記錄在 results["base_results"]["<model>"] 中
#           retrain 版本 base models 對全資料產生的 meta 特徵，儲存為 X_full_meta.csv
# step 6: 使用 X_full_meta 訓練 meta model，儲存為最終預測模型

import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import joblib
import os
import json
import sys
import os
import pandas as pd
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json
import argparse

def parse_base_models(value: str):
    """
    解析 base_models：
      - 先嘗試當成 JSON 解析，例如 '["xgb","lgbm"]'
      - 失敗時，再支援 '[xgb,lgbm]' 或 'xgb,lgbm' 這種格式
    """
    value = value.strip()
    # 先試標準 JSON
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # 去掉外層中括號（若有）
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    # 用逗號切開，清掉多餘空白與引號
    parts = [v.strip().strip('"').strip("'") for v in value.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"invalid base_models value: {value}")

    return parts


def save_meta_features_csv(meta_X, y, base_models, task_dir, filename):
    os.makedirs(task_dir, exist_ok=True)
    df = pd.DataFrame(meta_X, columns=base_models)
    df['y'] = y
    path = os.path.join(task_dir, filename)
    df.to_csv(path, index=False)
    return


def save_stacking_config(base_models, meta_model, n_features, task_dir):
    """儲存 stacking_config.json，供 getFieldNumber.py 和 predict.py 使用"""
    config = {
        "base_models": base_models,
        "meta_model": meta_model,
        "n_features": n_features
    }
    config_path = os.path.join(task_dir, "stacking_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return config_path


def main(file_path, base_models, label_column, meta_model, task_dir):
    results = {
        "status": "success",
        "base_results": {},
        "base_models": base_models,
        "meta_model": meta_model
    }
    # 1. 讀取資料
    try:
        X, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return
    os.makedirs(task_dir, exist_ok=True)

    # ✅ 儲存 stacking_config.json（包含 n_features）
    n_features = X.shape[1]
    save_stacking_config(base_models, meta_model, n_features, task_dir)

    # ✅ 自動偵測分類數，設定 focus_class
    num_class = int(len(np.unique(y)))
    focus_class = 1  # 預設用 class 1
    results["num_class"] = num_class
    results["focus_class"] = focus_class

    # 2. K-Fold 切分
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(X))  # 確保所有 model 用相同 folds

    # 3. 產生 OOF meta features
    meta_features = {name: np.zeros(X.shape[0]) for name in base_models}
    # base_models_per_fold = {name: [] for name in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        for name in base_models:
            module = import_module(f"stacking_{name}")
            preds = module.train_fold(
                X_train, y_train, X_val, focus_class=focus_class
            )
            meta_features[name][val_idx] = preds
    # 組成 meta_X，真實標籤用 y
    meta_X = np.column_stack([meta_features[name] for name in base_models])
    save_meta_features_csv(meta_X, y, base_models, task_dir, 'oof.csv')

    # 4. 訓練 meta model (OOF 階段)
    meta_module = import_module(f"stacking_{meta_model}")
    meta_results = meta_module.retrain(meta_X, y, base_models, task_dir, '0.8', False, 'meta', focus_class=focus_class)
    results["meta_results"] = meta_results

    # 5. Retrain base models on full data
    for name in base_models:
        module = import_module(f"stacking_{name}")
        base_result = module.retrain(X, y, feature_names, task_dir, '0.8', True, 'base', focus_class=focus_class)
        results["base_results"][f"{name}"] = base_result

    # 6. Retrain meta model on full data
    X_full_meta = []
    for name in base_models:
        module = import_module(f"stacking_{name}")
        preds = module.predict_full_meta(X, task_dir, focus_class=focus_class)
        X_full_meta.append(preds)
    X_full_meta = np.column_stack(X_full_meta)
    save_meta_features_csv(X_full_meta, y, base_models, task_dir, 'X_full_meta.csv')

    # 使用 retrain 函數訓練 meta model，儲存為 final 版本
    meta_module = import_module(f"stacking_{meta_model}")
    final_meta_result = meta_module.retrain(X_full_meta, y, base_models, task_dir, '1.0', True, 'meta', focus_class=focus_class)

    # ✅ 將 n_features 加入結果
    results["n_features"] = n_features
    results["task_dir"] = task_dir
    result_json_path = os.path.join(task_dir, "metrics.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
    extract_base64_images_and_clean_json(task_dir, "metrics.json")
    
    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument('base_models', type=parse_base_models) # JSON list of model names, e.g. \'["xgb","lgbm"]\'
    parser.add_argument("label_column", type=str)
    parser.add_argument("meta_model", type=str)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.base_models, args.label_column, args.meta_model, args.task_dir)