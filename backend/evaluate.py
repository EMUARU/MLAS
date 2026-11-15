# usage:
# python evaluate.py model.json data.csv eval_result label pred <task_dir>
# read_csv 和 read_excel 會自行轉換數值型 (int 或 float)和布林型 (bool)

# import subprocess
# import sys
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "scikit-learn==1.5.2"],
#     stdout=subprocess.DEVNULL
# )
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "--upgrade", "xgboost"],
#     stdout=subprocess.DEVNULL
# )

import argparse
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os
import json
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
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
import io
import base64
from tool_train import NumpyEncoder, extract_base64_images_and_clean_json
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
                
                # ✅ 檢查是否為訓練結果 zip（包含 .json 模型檔案）
                excluded_jsons = {'metrics.json', 'status.json', 'model_params.json', 'stacking_config.json'}
                json_model_files = [f for f in file_list if f.endswith('.json') and f not in excluded_jsons]
                if json_model_files:
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
                        # 注意：這裡不能刪除 temp_dir，因為某些模型可能需要它
                        # 實際上 joblib.load 後就不需要了，可以刪除
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return model
                
                # 檢查是否為 TabNet 模型 zip（包含 model_params.json 等）
                if "model_params.json" in file_list or any('network' in f for f in file_list):
                    model = TabNetClassifier()
                    model.load_model(model_path)
                    return model
                
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
    else:
        print(json.dumps({
            "status": "error",
            "message": f"Unsupported file format.",
        }))
        sys.exit(1)
    return data

def predict_labels(model, model_path, data, label_column, pred_column):
    # 先把 label 欄位拿掉，避免特徵數多一個
    x_test = data.drop(columns=[label_column], errors='ignore').copy()
    y_test = data[label_column].copy()

    class_name = model.__class__.__name__.lower()

    # TabNet (.zip) 或 TabPFN 都用 numpy.float32，其它照舊
    if model_path.lower().endswith(".zip") or class_name.startswith("tabpfn"):
        x_test = x_test.to_numpy(dtype=np.float32)
    else:
        x_test = x_test.values

    y_pred = model.predict(x_test)
    data[pred_column] = y_pred
    return x_test, y_test, y_pred, data


def save_predictions(data, data_path, output_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)

    if data_path.endswith(".csv"):
        output_path = os.path.join(task_dir, f"{output_name}.csv")
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif data_path.endswith(".xlsx"):
        output_path = os.path.join(task_dir, f"{output_name}.xlsx")
        data.to_excel(output_path, index=False)

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    # - 二元分類：照原本邏輯（class 1 為正類）
    # - 多元分類：轉成 focus_class vs rest 的二元視角
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== 1) 產生「二元視角」y_true_bin / y_score =====
    if num_class == 2:
        y_true_bin = y_test
        y_score = y_pred_proba[:, 1]
    else:
        focus_class = int(focus_class)
        y_true_bin = (y_test == focus_class).astype(int)
        y_score = y_pred_proba[:, focus_class]

    # ===== 2) 0.5 預設門檻的二元預測 =====
    y_pred_bin = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    result = {
        "status": "success",
        "num_class": int(num_class),
        "focus_class": int(focus_class) if num_class > 2 else 1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_true_bin, y_pred_bin) * 100),
            "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0) * 100),
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0) * 100),
            "f1_score": float(f1_score(y_true_bin, y_pred_bin, zero_division=0) * 100),
        }
    }

    # ===== 3) recall 門檻掃描 =====
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        # 二元：用 class 0 機率（與舊版 x[0] >= th 行為一致）
        score_for_threshold = y_pred_proba[:, 0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_threshold], dtype=int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            thresh_list.append(th)
            accuracy_score_list.append(accuracy_score(y_test, y_pred_th) * 100)
            precision_score_list.append(precision_score(y_test, y_pred_th, zero_division=0) * 100)
            recall_score_list.append(recall_score(y_test, y_pred_th, zero_division=0) * 100)
            f1_score_list.append(f1_score(y_test, y_pred_th, zero_division=0) * 100)
            specificity_score_list.append(specificity * 100)
            npv_score_list.append(npv * 100)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])
    else:
        # 多元：focus_class vs rest
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score >= th).astype(int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            thresh_list.append(th)
            accuracy_score_list.append(accuracy_score(y_true_bin, y_pred_th) * 100)
            precision_score_list.append(precision_score(y_true_bin, y_pred_th, zero_division=0) * 100)
            recall_score_list.append(recall_score(y_true_bin, y_pred_th, zero_division=0) * 100)
            f1_score_list.append(f1_score(y_true_bin, y_pred_th, zero_division=0) * 100)
            specificity_score_list.append(specificity * 100)
            npv_score_list.append(npv * 100)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])

    recall_standard_list = [80, 85, 90, 95]
    for recall_standard in recall_standard_list:
        high_recall_f1score = []
        high_recall_accuracy = []
        high_recall_recall = []
        high_recall_precision = []
        high_recall_specificity = []
        high_recall_npv = []
        high_recall_confusion_matrix = []
        high_recall_f2score = []
        for index, recall in enumerate(recall_score_list):
            if recall >= recall_standard:
                precision = precision_score_list[index]
                high_recall_f1score.append(f1_score_list[index])
                high_recall_accuracy.append(accuracy_score_list[index])
                high_recall_precision.append(precision_score_list[index])
                high_recall_recall.append(recall_score_list[index])
                high_recall_specificity.append(specificity_score_list[index])
                high_recall_npv.append(npv_score_list[index])
                high_recall_confusion_matrix.append(confusion_matrix_list[index])
                f2_score = (5 * precision * recall) / (4 * precision + recall)
                high_recall_f2score.append(f2_score)
        
        if high_recall_f1score:
            # 檢查是否有符合的 Recall 值
            high_recall_best_f1_score_index = np.argmax(high_recall_f1score)
            best_recall = high_recall_recall[high_recall_best_f1_score_index]
            best_specificity = high_recall_specificity[high_recall_best_f1_score_index]
            best_precision = high_recall_precision[high_recall_best_f1_score_index]
            best_npv = high_recall_npv[high_recall_best_f1_score_index]
            best_f1 = high_recall_f1score[high_recall_best_f1_score_index]
            best_f2 = high_recall_f2score[high_recall_best_f1_score_index]
            best_accuracy = high_recall_accuracy[high_recall_best_f1_score_index]
            best_confusion_matrix = high_recall_confusion_matrix[high_recall_best_f1_score_index]
        else:
            # 如果沒有符合 Recall ≥ 80, 85, 90, 95，填入 0
            best_recall = 0
            best_specificity = 0
            best_precision = 0
            best_npv = 0
            best_f1 = 0
            best_f2 = 0
            best_accuracy = 0
            best_confusion_matrix = [0, 0, 0, 0]

        key_name = f"recall_{recall_standard}"
        result[key_name] = {
            "recall": best_recall,
            "specificity": best_specificity,
            "precision": best_precision,
            "npv": best_npv,
            "f1_score": best_f1,
            "f2_score": best_f2,
            "accuracy": best_accuracy,
            "true_negative": best_confusion_matrix[0],
            "false_positive": best_confusion_matrix[1],
            "false_negative": best_confusion_matrix[2],
            "true_positive": best_confusion_matrix[3]
        }

    # plot
    fpr, tpr, thresholds = metrics.roc_curve(y_true_bin, y_score, pos_label=1)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    # plt.savefig('roc.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    result['roc'] = image_base64
    return result

def explain_with_shap(model, x_test, feature_names, focus_class=1):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        class_name = model.__class__.__name__.lower()
        is_tabnet_or_tabpfn = (
            hasattr(model, "predict_proba") and
            not hasattr(model, "predict")  # TabNet 沒有原生 predict，只有 predict_proba，TabNet / TabPFN 共用：用 KernelExplainer + predict_proba
        )\
        or class_name.startswith("tabnet")\
        or class_name.startswith("tabpfn")  # 新增：TabPFN

        if is_tabnet_or_tabpfn:
            # 專門為 TabNet / TabPFN 使用 KernelExplainer
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

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values_for_plot = shap_values[:, :, int(focus_class)]
            elif isinstance(shap_values, list) and len(shap_values) >= 2:
                shap_values_for_plot = shap_values[int(focus_class)]
            else:
                shap_values_for_plot = shap_values

            if shap_values_for_plot.ndim != 2:
                raise ValueError(f"shap_values shape is invalid: {shap_values_for_plot.shape}")
            
        # SVM
        elif isinstance(model, SVC):
            background = x_test[:20]
            x_explain = x_test[:5]

            if hasattr(model, "predict_proba"):
                f = lambda X: model.predict_proba(np.asarray(X, dtype=float))[:, int(focus_class)]
                explainer = shap.KernelExplainer(f, background, link="logit")
                shap_values = explainer.shap_values(x_explain, nsamples=100)
            else:
                f = lambda X: model.predict(np.asarray(X, dtype=float)).astype(float)
                explainer = shap.KernelExplainer(f, background, link="identity")
                shap_values = explainer.shap_values(x_explain, nsamples=100)

            shap_values_for_plot = shap_values
            shap_data = x_explain

        else:
            # 其他模型使用 shap.Explainer（TreeExplainer, etc.）
            # CatBoost 多類別模型帶背景資料時，shap 解析樹會發生記憶體損毀而使行程崩潰，
            # 因此與 train_catboost.py 相同，不帶背景資料（path-dependent TreeExplainer）
            if type(model).__name__.startswith("CatBoost"):
                explainer = shap.Explainer(model)
            else:
                explainer = shap.Explainer(model, x_test)
            shap_values = explainer(x_test, check_additivity=False)
            shap_data = shap_values.data

            if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                shap_values_for_plot = shap_values.values[:, :, int(focus_class)]
            else:
                shap_values_for_plot = shap_values.values

        # 平均重要度
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
        }

        # beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, shap_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

    except Exception as e:
        result["shap_error"] = str(e)

    return result

def explain_with_lime(model, x_test, y_test, feature_names, focus_class=1):
    result = {}
    try:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).astype(int)

        proba0 = model.predict_proba(x_test[:1])[0]
        num_class = int(len(proba0))
        class_names = [f"class_{i}" for i in range(num_class)]

        if num_class == 2:
            focus_class = 1
        else:
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= num_class:
                focus_class = int(np.argmax(proba0))

        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test[0],  # 第0筆樣本
            model.predict_proba,
            num_features=10,
            labels=[focus_class]
        )

        # 儲存圖片到記憶體 buffer
        fig = lime_result.as_pyplot_figure(label=focus_class)
        fig.set_size_inches(10, 6)  # 改變圖的尺寸
        fig.tight_layout()  # 自動調整 layout
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)  # 關閉圖避免重疊

        # 同時保留文字格式
        result["lime_example_0"] = lime_result.as_list(label=focus_class)

    except Exception as e:
        result["lime_error"] = str(e)
        
    return result

def predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names, focus_class=1):
    model = load_model(model_path)
    x_test, y_test, y_pred, data_with_preds = predict_labels(model, model_path, data, label_column, pred_column)
    save_predictions(data_with_preds, data_path, output_name, task_dir)
    results = evaluate_model(y_test, y_pred, model, x_test, focus_class=focus_class)
    shap_result = explain_with_shap(model, x_test, feature_names, focus_class=focus_class)
    results.update(shap_result)
    lime_result = explain_with_lime(model, x_test, y_test, feature_names, focus_class=focus_class)
    results.update(lime_result)
    return results

def predict_stacking(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names, focus_class=1):
    results = {
        "status": "success",
    }
    temp_dir = tempfile.mkdtemp()
    try:
        # 解壓 ZIP 到 temp 目錄
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        config_path = os.path.join(temp_dir, "stacking_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
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
        # 載入資料
        x_test_df = data.drop(columns=[label_column], errors='ignore').copy()
        y_test = data[label_column].copy()
        
        # 轉換成 meta features：每個 base model 的 predict_proba[:, focus_class]
        base_preds = []
        for model in base_models:
            model_class_name = model.__class__.__name__.lower()
            if isinstance(model, TabNetClassifier) or model_class_name.startswith("tabpfn"):
                proba = model.predict_proba(x_test_df.to_numpy(dtype=np.float32))[:, int(focus_class)]
            else:
                proba = model.predict_proba(x_test_df.values)[:, int(focus_class)]
            base_preds.append(proba)

        X_meta = np.column_stack(base_preds)
        meta_feature_df = pd.DataFrame(X_meta, columns=config.get("base_models", []))
        save_predictions(meta_feature_df, data_path, "meta_feature", task_dir)
        # 最終預測
        y_pred = meta_model.predict(X_meta)
        data[pred_column] = y_pred
        # 儲存預測結果
        save_predictions(data, data_path, output_name, task_dir)
        # 評估（meta model 的 focus_class 固定 1，因為 meta feature 已經是 binary score）
        results = evaluate_model(y_test, y_pred, meta_model, X_meta, focus_class=1)
        shap_result = explain_with_shap(meta_model, X_meta, config.get("base_models", []), focus_class=1)
        results.update(shap_result)
        lime_result = explain_with_lime(meta_model, X_meta, y_test, config.get("base_models", []), focus_class=1)
        results.update(lime_result)
        return results
    except Exception as e:
        return {
            "status": "error",
            "message": f"{e}"
        }
    finally:
        shutil.rmtree(temp_dir)  # 清除暫存資料夾

def main(model_path, data_path, output_name, label_column, pred_column, task_dir):
    try:
        data = load_data(data_path)
        feature_names = data.drop(columns=[label_column], errors='ignore').columns.tolist()

        focus_class = 1  # 預設

        if model_path.endswith("zip"):
            try:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    if "stacking_config.json" in zip_ref.namelist():
                        # 是 stacking zip
                        results = predict_stacking(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names, focus_class=focus_class)
                    else:
                        # 是 tabnet 模型
                        results = predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names, focus_class=focus_class)
            except zipfile.BadZipFile:
                print(json.dumps({
                    "status": "error",
                    "message": f"Invalid zip file: {model_path}",
                }))
                return
        else:
            results = predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names, focus_class=focus_class)
            
        results["task_dir"] = task_dir
        os.makedirs(task_dir, exist_ok=True)  
        result_json_path = os.path.join(task_dir, "metrics.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        extract_base64_images_and_clean_json(task_dir, "metrics.json")
        print(json.dumps(results, indent=4, cls=NumpyEncoder))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the trained model file (e.g., .json for XGBoost or .pkl for joblib models)")
    parser.add_argument('data_path', type=str, help="Path to input data file (.csv or .xlsx)")
    parser.add_argument('output_name', type=str, help="Output file name (without extension)")
    parser.add_argument('label_column', type=str, help="Name of the column that contains true labels")
    parser.add_argument("pred_column", type=str, help="Name of the column to store predictions")
    parser.add_argument("task_dir", type=str)
    
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_name, args.label_column, args.pred_column, args.task_dir)