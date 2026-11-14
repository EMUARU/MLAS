import json
import argparse
import os
from lightgbm import LGBMClassifier
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json

import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import numpy as np
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
from sklearn.model_selection import StratifiedKFold
import subprocess
import sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "shap==0.47.1"],
    stdout=subprocess.DEVNULL
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "lime==0.2.0.1"],
    stdout=subprocess.DEVNULL
)
import shap
from lime.lime_tabular import LimeTabularExplainer

def train_lgbm(x_train, y_train, x_val, y_val, model_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir):
    evals_result = {}

    # 自動偵測二元 / 多分類
    num_class = int(len(np.unique(y_train)))
    if num_class <= 2:
        objective = "binary"
        eval_metric_list = ["binary_logloss", "binary_error"]
        extra_kwargs = {}
    else:
        objective = "multiclass"
        eval_metric_list = ["multi_logloss", "multi_error"]
        extra_kwargs = {"num_class": num_class}

    lightgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        objective=objective,
        verbose=-1,
        **extra_kwargs
    )
    lightgbm.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_names=["training", "validation"],
        eval_metric=eval_metric_list,
        callbacks=[lgb.log_evaluation, lgb.record_evaluation(evals_result)]
    )
    
    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(lightgbm, f"{task_dir}/{model_name}.pkl")

    return lightgbm, evals_result

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

    # 多元：focus_class vs rest
    else:
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

    # ===== 挑 best threshold =====
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

    # ===== 4) ROC =====
    fpr, tpr, thresholds = metrics.roc_curve(y_true_bin, y_score, pos_label=1)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result['roc'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return result

def explain_with_shap(model, x_test, feature_names, focus_class=1):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        
        # 使用 SHAP.Explainer，禁用加總檢查
        explainer = shap.Explainer(model, x_test)
        shap_values = explainer(x_test, check_additivity=False)

        # 處理 SHAP 輸出：二元 vs 多元
        values = shap_values.values
        shap_data = shap_values.data

        if values.ndim == 3:
            # (samples, features, classes) → 取 focus_class
            shap_values_for_plot = values[:, :, int(focus_class)]
        else:
            shap_values_for_plot = values

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

def plot_loss(evals_result):
    # 自動判斷 binary_logloss 或 multi_logloss
    if 'binary_logloss' in evals_result['training']:
        loss_key = 'binary_logloss'
        ylabel = 'Log Loss'
        title = 'Training vs Validation Log Loss'
    elif 'multi_logloss' in evals_result['training']:
        loss_key = 'multi_logloss'
        ylabel = 'Multi-class Log Loss'
        title = 'Training vs Validation Multi-class Log Loss'
    else:
        loss_key = list(evals_result['training'].keys())[0]
        ylabel = loss_key
        title = f'Training vs Validation {loss_key}'

    logloss_train = evals_result['training'][loss_key]
    logloss_val = evals_result['validation'][loss_key]
    epochs = range(1, len(logloss_train) + 1)

    plt.figure()
    plt.plot(epochs, logloss_train, label='Train LogLoss')
    plt.plot(epochs, logloss_val, label='Validation LogLoss')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def plot_accuracy(evals_result):
    # 自動判斷 binary_error 或 multi_error
    if 'binary_error' in evals_result['training']:
        err_key = 'binary_error'
        title = 'Training vs Validation Accuracy'
    elif 'multi_error' in evals_result['training']:
        err_key = 'multi_error'
        title = 'Training vs Validation Accuracy (Multi-class)'
    else:
        keys0 = list(evals_result['training'].keys())
        candidates = [k for k in keys0 if 'error' in k.lower()]
        err_key = candidates[0] if candidates else keys0[0]
        title = f'Training vs Validation Accuracy ({err_key})'

    error_train = evals_result['training'][err_key]
    error_val = evals_result['validation'][err_key]
    acc_train = [1 - e for e in error_train]
    acc_val = [1 - e for e in error_val]
    epochs = range(1, len(acc_train) + 1)

    plt.figure()
    plt.plot(epochs, acc_train, label='Train Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def kfold_evaluation(x, y, cv_folds, model_name, n_estimators, learning_rate, max_depth, num_leaves, feature_names, task_dir, focus_class=1):
    x = np.asarray(x)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    folds_result = []
    all_metrics = {
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1_score": [],
    }
    total_tn = total_fp = total_fn = total_tp = 0

    for fold_index, (train_index, val_index) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model_fold_name = f"{model_name}_fold_{fold_index}"
        model, evals_result = train_lgbm(x_train, y_train, x_val, y_val, model_fold_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir)

        y_pred = model.predict(x_val)
        fold_eval = evaluate_model(y_val, y_pred, model, x_val, focus_class=focus_class)

        tn = fold_eval["confusion_matrix"]["true_negative"]
        fp = fold_eval["confusion_matrix"]["false_positive"]
        fn = fold_eval["confusion_matrix"]["false_negative"]
        tp = fold_eval["confusion_matrix"]["true_positive"]

        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        all_metrics["accuracy"].append(fold_eval["metrics"]["accuracy"])
        all_metrics["recall"].append(fold_eval["metrics"]["recall"])
        all_metrics["precision"].append(fold_eval["metrics"]["precision"])
        all_metrics["f1_score"].append(fold_eval["metrics"]["f1_score"])

        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)
        shap_result = explain_with_shap(model, x_val, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, x_val, y_val, feature_names, focus_class=focus_class)

        folds_result.append({
            "fold": fold_index,
            "metrics": fold_eval.get("metrics"),
            "confusion_matrix": fold_eval.get("confusion_matrix"),
            "roc": fold_eval.get("roc"),
            "loss_plot": loss_base64,
            "accuracy_plot": acc_base64,
            "shap_plot": shap_result.get("shap_plot"),
            "shap_importance": shap_result.get("shap_importance"),
            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0"),
            "recall_80": fold_eval.get("recall_80"),
            "recall_85": fold_eval.get("recall_85"),
            "recall_90": fold_eval.get("recall_90"),
            "recall_95": fold_eval.get("recall_95"),
        })

    avg_result = {
        "accuracy": float(np.mean(all_metrics["accuracy"])) if all_metrics["accuracy"] else 0.0,
        "recall": float(np.mean(all_metrics["recall"])) if all_metrics["recall"] else 0.0,
        "precision": float(np.mean(all_metrics["precision"])) if all_metrics["precision"] else 0.0,
        "f1_score": float(np.mean(all_metrics["f1_score"])) if all_metrics["f1_score"] else 0.0,
        "confusion_matrix": {
            "true_negative": int(total_tn),
            "false_positive": int(total_fp),
            "false_negative": int(total_fn),
            "true_positive": int(total_tp),
        }
    }

    result = {
        "status": "success",
        "folds": folds_result,
        "average": avg_result
    }
    return result

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir):
    try:
        x, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

    if split_strategy == "train_test_split":
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=float(split_value), stratify=y, random_state=30
            )
            model, evals_result = train_lgbm(x_train, y_train, x_test, y_test, model_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir)
            y_pred = model.predict(x_test)
            results = evaluate_model(y_test, y_pred, model, x_test, focus_class=1)
            results["loss_plot"] = plot_loss(evals_result)
            results["accuracy_plot"] = plot_accuracy(evals_result)
            shap_result = explain_with_shap(model, x_test, feature_names, focus_class=1)
            results.update(shap_result)
            lime_result = explain_with_lime(model, x_test, y_test, feature_names, focus_class=1)
            results.update(lime_result)
        except ValueError as e:
            print(json.dumps({
                "status": "error",
                "message": f"{e}",
            }))
            return

    elif split_strategy == "k_fold":
        try:
            results = kfold_evaluation(
                x, y,
                int(split_value),  # split_value 為 fold 數
                model_name,
                n_estimators, learning_rate, max_depth, num_leaves,
                feature_names,
                task_dir,
                focus_class=1
            )
        except ValueError as e:
            print(json.dumps({
                "status": "error",
                "message": f"{e}",
            }))
            return

    else:
        print(json.dumps({
            "status": "error",
            "message": "Unsupported split strategy"
        }))
        return

    results["task_dir"] = task_dir
    result_json_path = os.path.join(task_dir, "metrics.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
    extract_base64_images_and_clean_json(task_dir, "metrics.json")
    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("label_column", type=str)
    parser.add_argument("split_strategy", type=str)
    parser.add_argument("split_value", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("num_leaves", type=int)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.n_estimators, args.learning_rate, args.max_depth, args.num_leaves, args.task_dir)