# train_catboost.py

import json
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "catboost==1.2.8"],
    stdout=subprocess.DEVNULL
)
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
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
import shap
from lime.lime_tabular import LimeTabularExplainer
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json

# 字型設置
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def train_catboost(x_train, y_train, x_val, y_val, model_name, iterations, learning_rate, depth, task_dir):
    # 自動偵測二元 / 多分類
    num_class = int(len(np.unique(y_train)))
    if num_class <= 2:
        loss_function = "Logloss"
        eval_metric = "Logloss"
        custom_metric = ["Accuracy"]
    else:
        loss_function = "MultiClass"
        eval_metric = "MultiClass"
        custom_metric = ["Accuracy"]

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        eval_metric=eval_metric,
        verbose=0,
        use_best_model=True,
        custom_metric=custom_metric,
    )
    model.fit(
        x_train, y_train,
        eval_set=(x_val, y_val),
        plot=False
    )
    evals_result = model.get_evals_result()
    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(model, f"{task_dir}/{model_name}.pkl")
    return model, evals_result

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    if num_class == 2:
        y_true_bin = y_test
        y_score = y_pred_proba[:, 1]
    else:
        focus_class = int(focus_class)
        y_true_bin = (y_test == focus_class).astype(int)
        y_score = y_pred_proba[:, focus_class]

    y_pred_bin = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
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
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        score_for_threshold = y_pred_proba[:, 0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_threshold], dtype=int)
            tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th, labels=[0, 1]).ravel()
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
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score >= th).astype(int)
            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_th, labels=[0, 1]).ravel()
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
        high_recall_precision = []
        high_recall_specificity = []
        high_recall_npv = []
        high_recall_confusion_matrix = []
        high_recall_f2score = []
        high_recall_recall = []

        for i, recall in enumerate(recall_score_list):
            if recall >= recall_standard:
                precision = precision_score_list[i]
                high_recall_f1score.append(f1_score_list[i])
                high_recall_accuracy.append(accuracy_score_list[i])
                high_recall_precision.append(precision)
                high_recall_recall.append(recall)
                high_recall_specificity.append(specificity_score_list[i])
                high_recall_npv.append(npv_score_list[i])
                high_recall_confusion_matrix.append(confusion_matrix_list[i])
                f2_score = (5 * precision * recall) / (4 * precision + recall)
                high_recall_f2score.append(f2_score)

        if high_recall_f1score:
            idx = np.argmax(high_recall_f1score)
            best_conf = high_recall_confusion_matrix[idx]
            result[f"recall_{recall_standard}"] = {
                "recall": high_recall_recall[idx],
                "specificity": high_recall_specificity[idx],
                "precision": high_recall_precision[idx],
                "npv": high_recall_npv[idx],
                "f1_score": high_recall_f1score[idx],
                "f2_score": high_recall_f2score[idx],
                "accuracy": high_recall_accuracy[idx],
                "true_negative": best_conf[0],
                "false_positive": best_conf[1],
                "false_negative": best_conf[2],
                "true_positive": best_conf[3]
            }

    # ROC plot
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot([0, 1], [0, 1], color='0', linestyle="-.")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
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
        explainer = shap.Explainer(model)
        shap_values = explainer(x_test)

        shap_importance = np.abs(shap_values.values).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
        }

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, x_test, feature_names=feature_names, show=False)
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
        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=[f"class_{i}" for i in range(len(model.predict_proba(x_test[:1])[0]))],
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test[0], model.predict_proba, num_features=10,
            labels=[int(focus_class)]
        )

        fig = lime_result.as_pyplot_figure(label=int(focus_class))
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        result["lime_example_0"] = lime_result.as_list(label=int(focus_class))
    except Exception as e:
        result["lime_error"] = str(e)
    return result

def plot_loss(evals_result):
    # 自動判斷 Logloss (binary) 或 MultiClass (multi)
    if 'Logloss' in evals_result['learn']:
        loss_key = 'Logloss'
    elif 'MultiClass' in evals_result['learn']:
        loss_key = 'MultiClass'
    else:
        loss_key = list(evals_result['learn'].keys())[0]

    logloss_train = evals_result['learn'][loss_key]
    logloss_val = evals_result['validation'][loss_key]
    epochs = range(1, len(logloss_train) + 1)

    plt.figure()
    plt.plot(epochs, logloss_train, label='Train LogLoss')
    plt.plot(epochs, logloss_val, label='Validation LogLoss')
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training vs Validation Log Loss")
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
    acc_train = evals_result['learn']['Accuracy']
    acc_val = evals_result['validation']['Accuracy']
    epochs = range(1, len(acc_train) + 1)

    plt.figure()
    plt.plot(epochs, acc_train, label='Train Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def kfold_evaluation(x, y, cv_folds, model_name, iterations, learning_rate, depth, feature_names, task_dir, focus_class=1):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    folds_result = []
    total_tn = total_fp = total_fn = total_tp = 0
    all_metrics = {"accuracy": [], "recall": [], "precision": [], "f1_score": [], "auc": []}

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model, evals_result = train_catboost(x_train, y_train, x_val, y_val, f"{model_name}_fold_{fold_index}", iterations, learning_rate, depth, task_dir)

        # Multi-class: convert to binary view
        _num_class = len(model.classes_) if hasattr(model, "classes_") else int(model.predict_proba(x_val[:1]).shape[1])
        if _num_class > 2:
            y_val_bin = (y_val == int(focus_class)).astype(int)
        else:
            y_val_bin = y_val

        y_pred_proba = model.predict_proba(x_val)[:, int(focus_class)]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_bin, y_pred, labels=[0, 1]).ravel()

        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        accuracy = accuracy_score(y_val_bin, y_pred)
        recall = recall_score(y_val_bin, y_pred, zero_division=0)
        precision = precision_score(y_val_bin, y_pred, zero_division=0)
        f1 = f1_score(y_val_bin, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_val_bin, y_pred_proba)
        auc_score = auc(fpr, tpr)

        all_metrics["accuracy"].append(accuracy)
        all_metrics["recall"].append(recall)
        all_metrics["precision"].append(precision)
        all_metrics["f1_score"].append(f1)
        all_metrics["auc"].append(auc_score)

        plt.figure()
        plt.plot(fpr, tpr, color='m', label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], color='0', linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        roc_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()

        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)
        shap_result = explain_with_shap(model, x_val, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, x_val, y_val, feature_names, focus_class=focus_class)

        folds_result.append({
            "fold": fold_index,
            "metrics": {
                "accuracy": accuracy * 100,
                "recall": recall * 100,
                "precision": precision * 100,
                "f1_score": f1 * 100,
                "auc": auc_score * 100,
            },
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "roc": roc_base64,
            "loss_plot": loss_base64,
            "accuracy_plot": acc_base64,
            "shap_plot": shap_result.get("shap_plot"),
            "shap_importance": shap_result.get("shap_importance"),
            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0")
        })

    avg_result = {
        "accuracy": float(np.mean(all_metrics["accuracy"])) * 100,
        "recall": float(np.mean(all_metrics["recall"])) * 100,
        "precision": float(np.mean(all_metrics["precision"])) * 100,
        "f1_score": float(np.mean(all_metrics["f1_score"])) * 100,
        "auc": float(np.mean(all_metrics["auc"])) * 100,
        "confusion_matrix": {
            "true_negative": total_tn,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "true_positive": total_tp,
        }
    }
    return {"status": "success", "folds": folds_result, "average": avg_result}

def main(file_path, label_column, split_strategy, split_value, model_name, iterations, learning_rate, depth, task_dir):
    try:
        x, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return

    if split_strategy == "train_test_split":
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=float(split_value), stratify=y, random_state=30)
        model, evals_result = train_catboost(x_train, y_train, x_test, y_test, model_name, iterations, learning_rate, depth, task_dir)
        y_pred = model.predict(x_test)
        result = evaluate_model(y_test, y_pred, model, x_test, focus_class=1)
        result["loss_plot"] = plot_loss(evals_result)
        result["accuracy_plot"] = plot_accuracy(evals_result)
        result.update(explain_with_shap(model, x_test, feature_names, focus_class=1))
        result.update(explain_with_lime(model, x_test, y_test, feature_names, focus_class=1))
    elif split_strategy == "k_fold":
        result = kfold_evaluation(x, y, int(split_value), model_name, iterations, learning_rate, depth, feature_names, task_dir, focus_class=1)
    else:
        print(json.dumps({"status": "error", "message": "Unsupported split strategy"}))
        return

    result["task_dir"] = task_dir
    with open(os.path.join(task_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
    extract_base64_images_and_clean_json(task_dir, "metrics.json")
    print(json.dumps(result, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("label_column", type=str)
    parser.add_argument("split_strategy", type=str)
    parser.add_argument("split_value", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("iterations", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("depth", type=int)
    parser.add_argument("task_dir", type=str)
    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.iterations, args.learning_rate, args.depth, args.task_dir)