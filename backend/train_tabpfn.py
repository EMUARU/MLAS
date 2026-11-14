# usage:
#   python train_tabpfn.py upload/xxx.csv label train_test_split 0.8 tabpfn_model <task_dir>
#   python train_tabpfn.py upload/xxx.csv label k_fold 5 "" tabpfn_model <task_dir>
#
#   進階可選參數（前端目前還沒用到）：
#     --model_version {default,v2}
#     --device {auto,cuda,cpu}
#     --n_ensembles N
#     --allow_cpu_large_dataset

import json
import argparse
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 字型設定（避免中文亂碼）
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = font_prop.get_name()
except Exception:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

import io
import base64
import torch

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 安裝 shap / lime（跟原始專案一致）
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

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json


# ============================================================
# 共用繪圖 / 評估工具
# ============================================================

def create_line_plot(x, y, xlabel, ylabel, title):
    """
    建立一般折線圖並回傳 base64 字串
    """
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return image_base64


def generate_threshold_curves(y_true_bin, y_pred_proba, num_class=2, y_score=None):
    """
    在不同 threshold (0.01~1.00) 底下計算多種指標。
    y_pred_proba: (n_samples, 2) 或 (n_samples,)
                  表示 positive 類別的機率。
    回傳：
      thresh_list, accuracy_list, precision_list, recall_list,
      f1_list, specificity_list, npv_list, confusion_matrix_list
    所有分數皆為百分比 (0~100)。
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)

    # 若有提供 y_score（多分類時已是 focus_class 的機率），直接用
    if y_score is not None:
        pos_proba = y_score
    elif y_pred_proba.ndim == 2:
        pos_proba = y_pred_proba[:, 1]
    else:
        pos_proba = y_pred_proba

    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    for th_int in range(1, 101):
        th = th_int * 0.01
        y_pred_th = (pos_proba >= th).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_th, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (fn + tn) if (fn + tn) > 0 else 0.0

        thresh_list.append(th)
        accuracy_score_list.append(accuracy_score(y_true_bin, y_pred_th) * 100)
        precision_score_list.append(precision_score(y_true_bin, y_pred_th, zero_division=0) * 100)
        recall_score_list.append(recall_score(y_true_bin, y_pred_th, zero_division=0) * 100)
        f1_score_list.append(f1_score(y_true_bin, y_pred_th, zero_division=0) * 100)
        specificity_score_list.append(specificity * 100)
        npv_score_list.append(npv * 100)
        confusion_matrix_list.append([tn, fp, fn, tp])

    return (
        thresh_list,
        accuracy_score_list,
        precision_score_list,
        recall_score_list,
        f1_score_list,
        specificity_score_list,
        npv_score_list,
        confusion_matrix_list,
    )


def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    """
    單一 train/test split 評估：
      - confusion matrix + 基本 metrics
      - 針對 recall=80/85/90/95 的最佳點
      - ROC 圖
      - 「類 epoch」的 threshold 曲線 (accuracy / pseudo-loss)
    """
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    x_test_np = np.array(x_test, dtype=np.float32)
    y_pred_proba = model.predict_proba(x_test_np)
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
        },
    }

    # 門檻掃描（recall 分析 + pseudo-epoch 曲線）

    (
        thresh_list,
        accuracy_score_list,
        precision_score_list,
        recall_score_list,
        f1_score_list,
        specificity_score_list,
        npv_score_list,
        confusion_matrix_list,
    ) = generate_threshold_curves(y_true_bin, y_pred_proba, num_class, y_score)

    # 針對 recall=80/85/90/95 找到 F1 最佳點
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

        for idx, recall_v in enumerate(recall_score_list):
            if recall_v >= recall_standard:
                precision_v = precision_score_list[idx]

                high_recall_f1score.append(f1_score_list[idx])
                high_recall_accuracy.append(accuracy_score_list[idx])
                high_recall_precision.append(precision_score_list[idx])
                high_recall_recall.append(recall_score_list[idx])
                high_recall_specificity.append(specificity_score_list[idx])
                high_recall_npv.append(npv_score_list[idx])
                high_recall_confusion_matrix.append(confusion_matrix_list[idx])

                # F2: recall 權重更大
                f2_num = 5 * precision_v * recall_v
                f2_den = 4 * precision_v + recall_v
                f2_score = f2_num / f2_den if f2_den > 0 else 0.0
                high_recall_f2score.append(f2_score)

        if high_recall_f1score:
            best_idx = int(np.argmax(high_recall_f1score))
            best_recall = high_recall_recall[best_idx]
            best_specificity = high_recall_specificity[best_idx]
            best_precision = high_recall_precision[best_idx]
            best_npv = high_recall_npv[best_idx]
            best_f1 = high_recall_f1score[best_idx]
            best_f2 = high_recall_f2score[best_idx]
            best_accuracy = high_recall_accuracy[best_idx]
            best_confusion_matrix = high_recall_confusion_matrix[best_idx]
        else:
            best_recall = best_specificity = best_precision = best_npv = 0.0
            best_f1 = best_f2 = best_accuracy = 0.0
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
            "true_positive": best_confusion_matrix[3],
        }

    # 「Pseudo-epoch」圖：x 軸 threshold，y 軸 accuracy / loss
    accuracy_plot = create_line_plot(
        thresh_list,
        accuracy_score_list,
        xlabel="Decision Threshold",
        ylabel="Accuracy (%)",
        title="TabPFN Accuracy vs Decision Threshold",
    )
    pseudo_loss = [100.0 - f1 for f1 in f1_score_list]
    loss_plot = create_line_plot(
        thresh_list,
        pseudo_loss,
        xlabel="Decision Threshold",
        ylabel="Pseudo Loss (100 - F1)",
        title="TabPFN Error Rate vs Decision Threshold",
    )
    result["accuracy_plot"] = accuracy_plot
    result["loss_plot"] = loss_plot

    # ROC 曲線
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score, pos_label=1)

    plt.figure()
    plt.plot(fpr, tpr, color="m", label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color="0", linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()

    result["roc"] = image_base64
    return result


def explain_with_shap(model, x_train, x_test, feature_names, focus_class=1):
    """
    參考 TabNet 的做法：用 KernelExplainer + predict_proba 做 SHAP

    為避免使用測試資料作為 background 造成資訊洩漏，本函式之 background
    自訓練資料中抽樣，並對測試資料中之少量代表樣本進行解釋。
    """
    result = {}
    try:
        x_train = np.array(x_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)

        # background 自訓練資料抽樣（隨機抽 20 筆，若不足則取全部）
        n_bg = min(20, len(x_train))
        if n_bg < len(x_train):
            rng = np.random.default_rng(seed=42)
            bg_idx = rng.choice(len(x_train), size=n_bg, replace=False)
            background = x_train[bg_idx]
        else:
            background = x_train

        # 待解釋樣本仍由測試資料取得（使用者實際關心之預測對象）
        explain_data = x_test[:15]

        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(explain_data)

        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) -> 取正類
            shap_values_for_plot = shap_values[:, :, int(focus_class)]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_for_plot = shap_values[int(focus_class)]
        else:
            shap_values_for_plot = shap_values

        if shap_values_for_plot.ndim != 2:
            raise ValueError(f"shap_values shape invalid: {shap_values_for_plot.shape}")

        shap_data = explain_data

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
        plt.savefig(buf, format="png")
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

    except Exception as e:
        result["shap_error"] = str(e)

    return result


def explain_with_lime(model, x_train, y_train, x_test, y_test, feature_names, focus_class=1):
    """
    用 LimeTabularExplainer 做局部解釋

    為避免以測試資料作為 LIME 局部擾動之參考分布而造成資訊洩漏，
    本函式之 training_data 改自訓練資料抽取，並對測試資料中之第一筆樣本進行解釋。
    """
    result = {}
    try:
        x_train_np = np.array(x_train, dtype=np.float32)
        y_train_np = np.array(y_train)
        x_test_np = np.array(x_test, dtype=np.float32)

        lime_explainer = LimeTabularExplainer(
            training_data=x_train_np,
            mode="classification",
            training_labels=y_train_np,
            feature_names=feature_names,
            class_names=[f"class_{i}" for i in range(len(model.predict_proba(x_train_np[:1])[0]))],
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test_np[0],
            model.predict_proba,
            num_features=10,
            labels=[int(focus_class)]
        )

        fig = lime_result.as_pyplot_figure(label=int(focus_class))
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

        result["lime_example_0"] = lime_result.as_list(label=int(focus_class))

    except Exception as e:
        result["lime_error"] = str(e)

    return result


def plot_placeholder(message):
    """
    如果之後還有地方要顯示「沒有資料」的空白圖，可以用這個。
    目前主流程已不使用，只保留相容性。
    """
    plt.figure()
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return image_base64


# ============================================================
# TabPFN 訓練 / K-fold
# ============================================================

def train_tabpfn(
    x_train,
    y_train,
    model_name,
    task_dir,
    model_version="default",
    device_mode="auto",
    n_ensembles=None,
    allow_cpu_large_dataset=True,
):
    """
    實際訓練 TabPFN。
    model_version : "default" 使用 TabPFN-2.5；"v2" 使用 TabPFNv2 權重
    device_mode   : "auto" / "cuda" / "cpu"
    n_ensembles   : N_ensemble_configurations，None 則使用官方預設
    """
    os.makedirs(task_dir, exist_ok=True)

    x_train_np = np.array(x_train, dtype=np.float32)
    y_train_np = np.array(y_train)

    # 選擇運算裝置
    if device_mode == "cuda":
        device = "cuda"
    elif device_mode == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 是否允許 CPU 上的大資料集（會很慢，所以預設關掉）
    if True or (allow_cpu_large_dataset and device == "cpu"):
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    # ---- n_ensembles 防呆處理：允許 0，整數，上限 32 ----
    if n_ensembles is not None:
        try:
            n_ensembles_int = int(n_ensembles)
        except (TypeError, ValueError):
            # 轉型失敗就視為未指定，讓 TabPFN 用預設
            n_ensembles_int = None
        else:
            if n_ensembles_int < 0:
                # 負數視為無效，改用預設
                n_ensembles_int = None
            elif n_ensembles_int == 0:
                # 0 代表「不要覆寫，用 TabPFN 預設」
                n_ensembles_int = None
            elif n_ensembles_int > 32:
                # 太大就鎖在上限 32
                n_ensembles_int = 32
        n_ensembles = n_ensembles_int

    # TabPFN 新版 API：用 n_estimators 控制 ensemble 數量
    clf_kwargs = {
        "device": device,
        "ignore_pretraining_limits": True, # 忽略預訓練模型的資料集大小限制
    }
    if n_ensembles is not None:
        clf_kwargs["n_estimators"] = n_ensembles

    # 選擇模型版本
    if model_version == "v2":
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **clf_kwargs)
    else:
        # default = 2.5 權重
        clf = TabPFNClassifier(**clf_kwargs)

    clf.fit(x_train_np, y_train_np)

    # 儲存模型
    model_path = os.path.join(task_dir, f"{model_name}_tabpfn.pkl")
    import joblib
    joblib.dump(clf, model_path)

    return clf, model_path


def kfold_evaluation(
    X,
    y,
    cv_folds,
    model_name,
    feature_names,
    task_dir,
    model_version="default",
    device_mode="auto",
    n_ensembles=None,
    allow_cpu_large_dataset=False,
    focus_class=1,
):
    """
    K-fold 評估
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    folds_result = []

    all_metrics = {
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1_score": [],
        "auc": [],
    }

    total_tn = total_fp = total_fn = total_tp = 0

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        model_fold_name = f"{model_name}_fold_{fold}"
        model, model_path = train_tabpfn(
            X_train,
            Y_train,
            model_fold_name,
            task_dir,
            model_version=model_version,
            device_mode=device_mode,
            n_ensembles=n_ensembles,
            allow_cpu_large_dataset=allow_cpu_large_dataset,
        )

        X_test_np = np.array(X_test, dtype=np.float32)
        y_pred_proba = model.predict_proba(X_test_np)
        y_pred = (y_pred_proba[:, int(focus_class)] >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        accuracy = accuracy_score(Y_test, y_pred)
        recall_v = recall_score(Y_test, y_pred, zero_division=0)
        precision_v = precision_score(Y_test, y_pred, zero_division=0)
        f1_v = f1_score(Y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(Y_test, y_pred_proba[:, int(focus_class)])
        auc_score = auc(fpr, tpr)

        all_metrics["accuracy"].append(accuracy)
        all_metrics["recall"].append(recall_v)
        all_metrics["precision"].append(precision_v)
        all_metrics["f1_score"].append(f1_v)
        all_metrics["auc"].append(auc_score)

        # ROC 圖
        plt.figure()
        plt.plot(fpr, tpr, color="m", label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], color="0", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        roc_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

        # 由 threshold 曲線產生 pseudo-epoch 圖
        (
            thresh_list,
            accuracy_score_list,
            precision_score_list,
            recall_score_list,
            f1_score_list,
            specificity_score_list,
            npv_score_list,
            confusion_matrix_list,
        ) = generate_threshold_curves(
            (Y_test == int(focus_class)).astype(int) if y_pred_proba.shape[1] > 2 else Y_test,
            y_pred_proba,
            y_pred_proba.shape[1],
            y_pred_proba[:, int(focus_class)]
        )

        accuracy_plot = create_line_plot(
            thresh_list,
            accuracy_score_list,
            xlabel="Decision Threshold",
            ylabel="Accuracy (%)",
            title=f"Fold {fold} Accuracy vs Threshold",
        )
        pseudo_loss = [100.0 - f1 for f1 in f1_score_list]
        loss_plot = create_line_plot(
            thresh_list,
            pseudo_loss,
            xlabel="Decision Threshold",
            ylabel="Pseudo Loss (100 - F1)",
            title=f"Fold {fold} Pseudo-loss vs Threshold",
        )

        shap_result = explain_with_shap(model, X_train, X_test, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, X_train, Y_train, X_test, Y_test, feature_names, focus_class=focus_class)

        folds_result.append({
            "fold": fold,
            "metrics": {
                "accuracy": accuracy * 100,
                "recall": recall_v * 100,
                "precision": precision_v * 100,
                "f1_score": f1_v * 100,
                "auc": auc_score * 100,
            },
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "roc": roc_base64,
            "loss_plot": loss_plot,
            "accuracy_plot": accuracy_plot,
            "shap_plot": shap_result.get("shap_plot"),
            "shap_importance": shap_result.get("shap_importance"),
            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0"),
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
        },
    }

    return {
        "status": "success",
        "folds": folds_result,
        "average": avg_result,
    }


# ============================================================
# main
# ============================================================

def main(
    file_path,
    label_column,
    split_strategy,
    split_value,
    model_name,
    task_dir,
    model_version="default",
    device_mode="auto",
    n_ensembles=None,
    allow_cpu_large_dataset=False,
    focus_class=1,
):
    try:
        X, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

    if split_strategy == "train_test_split":
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, train_size=float(split_value), stratify=y, random_state=30
            )

            model, model_path = train_tabpfn(
                x_train,
                y_train,
                model_name,
                task_dir,
                model_version=model_version,
                device_mode=device_mode,
                n_ensembles=n_ensembles,
                allow_cpu_large_dataset=allow_cpu_large_dataset,
            )

            x_test_np = np.array(x_test, dtype=np.float32)
            y_pred = model.predict(x_test_np)

            results = evaluate_model(y_test, y_pred, model, x_test_np, focus_class=1)

            shap_result = explain_with_shap(model, x_train, x_test, feature_names, focus_class=1)
            results.update(shap_result)
            lime_result = explain_with_lime(model, x_train, y_train, x_test, y_test, feature_names, focus_class=1)
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
                np.array(X),
                np.array(y),
                int(split_value),
                model_name,
                feature_names,
                task_dir,
                model_version=model_version,
                device_mode=device_mode,
                n_ensembles=n_ensembles,
                allow_cpu_large_dataset=allow_cpu_large_dataset,
                focus_class=1,
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
            "message": "Unsupported split strategy",
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
    parser.add_argument("task_dir", type=str)

    # TabPFN 特有參數（前端目前還沒用到，先保留 CLI 用）
    parser.add_argument(
        "--model_version",
        type=str,
        default="default",
        choices=["default", "v2"],
        help="TabPFN model version: 'default' (2.5) or 'v2'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run TabPFN on.",
    )
    parser.add_argument(
        "--n_ensembles",
        type=int,
        default=None,
        help="N_ensemble_configurations for TabPFN (None = library default).",
    )
    parser.add_argument(
        "--allow_cpu_large_dataset",
        action="store_true",
        help="Allow running large datasets on CPU (may be very slow).",
    )

    args = parser.parse_args()
    main(
        args.file_path,
        args.label_column,
        args.split_strategy,
        args.split_value,
        args.model_name,
        args.task_dir,
        model_version=args.model_version,
        device_mode=args.device,
        n_ensembles=args.n_ensembles,
        allow_cpu_large_dataset=args.allow_cpu_large_dataset,
    )