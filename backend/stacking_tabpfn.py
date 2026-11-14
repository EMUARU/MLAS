import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn import metrics
import matplotlib.pyplot as plt
import base64
import io
import matplotlib
from matplotlib import font_manager
from contextlib import contextmanager
import sys

from tabpfn import TabPFNClassifier

# ========= 字型設定（跟其他 stacking_* 檔案一致） =========
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

# 關掉 HuggingFace 進度條，避免太多輸出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


@contextmanager
def suppress_stdout():
    """暫時關閉標準輸出，避免 TabPFN 在訓練時印出太多資訊"""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ========= 共用繪圖 / 門檻掃描工具 =========

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
    y_pred_proba: (n_samples, 2)，TabPFNClassifier.predict_proba() 的輸出。
    回傳：
      thresh_list, accuracy_list, precision_list, recall_list,
      f1_list, specificity_list, npv_list, confusion_matrix_list
    所有分數皆為百分比 (0~100)。
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)

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
        if num_class == 2:
            y_pred_th = [0 if (x[0] >= th) else 1 for x in y_pred_proba]
        else:
            y_pred_th = (y_score >= th).astype(int)

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


# ========= Stacking 需要的三個主要介面 =========

def train_fold(X_train, y_train, X_val, n_estimators=32, device="cuda", focus_class=1):
    """
    給 stacking 的 K-fold 用：
      在 (X_train, y_train) 上訓練 TabPFN，然後回傳 X_val 的預測機率 (class 1)。
    """
    X_train_np = np.array(X_train, dtype=np.float32)
    X_val_np = np.array(X_val, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.int64)

    clf = TabPFNClassifier(
        device=device,
        n_estimators=n_estimators
    )

    with suppress_stdout():
        clf.fit(X_train_np, y_train_np)

    val_preds = clf.predict_proba(X_val_np)[:, int(focus_class)]
    return val_preds


def retrain(
    X,
    y,
    feature_names,
    task_dir,
    split_value=1.0,
    save_model=True,
    model_role='base',
    n_estimators=32,
    device="cuda",
    focus_class=1,
):
    """
    給 stacking 用的「重新訓練」介面：
      - split_value != '1.0'：會先切 train/val，做一次完整評估 + XAI
      - split_value == '1.0'：用全部資料訓練，結果不做評估（results = None）
    """
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int64)

    clf = TabPFNClassifier(
        device=device,
        n_estimators=n_estimators
    )

    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, train_size=float(split_value), stratify=y_np, random_state=30
        )
        with suppress_stdout():
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # 基本評估 + ROC + threshold 曲線 (pseudo-epoch)
        results = evaluate_model(y_test, y_pred, clf, X_test, focus_class=focus_class)

        # SHAP / LIME
        shap_result = explain_with_shap(clf, X_test, feature_names, focus_class=focus_class)
        results.update(shap_result)
        lime_result = explain_with_lime(clf, X_test, y_test, feature_names, focus_class=focus_class)
        results.update(lime_result)
    else:
        with suppress_stdout():
            clf.fit(X_np, y_np)
        results = None

    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        model_path = f"{task_dir}/{model_role}_tabpfn.pkl"
        with suppress_stdout():
            joblib.dump(clf, model_path)

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    """
    給 stacking 第二層 meta model 用：
      讀取 base_tabpfn.pkl，對整個 X 做 predict_proba()，回傳 class 1 的機率。
    """
    model_path = f"{task_dir}/base_tabpfn.pkl"
    clf = joblib.load(model_path)
    X_np = np.array(X, dtype=np.float32)
    preds = clf.predict_proba(X_np)[:, int(focus_class)]
    return preds


# ========= 評估 + XAI =========

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    """
    評估：
      - confusion matrix
      - accuracy / precision / recall / f1
      - recall=80/85/90/95 下的最佳點 (和其他 stacking_* 一樣)
      - ROC 圖
      - 「類 epoch」的 threshold 曲線：accuracy_plot / loss_plot
    """
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

    # ===== 門檻掃描（recall 分析 + pseudo-epoch 曲線）=====

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

        for index, recall_v in enumerate(recall_score_list):
            if recall_v >= recall_standard:
                precision_v = precision_score_list[index]
                high_recall_f1score.append(f1_score_list[index])
                high_recall_accuracy.append(accuracy_score_list[index])
                high_recall_precision.append(precision_score_list[index])
                high_recall_recall.append(recall_score_list[index])
                high_recall_specificity.append(specificity_score_list[index])
                high_recall_npv.append(npv_score_list[index])
                high_recall_confusion_matrix.append(confusion_matrix_list[index])

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

    # ===== Pseudo-epoch 圖：x 軸 threshold，y 軸 accuracy / pseudo-loss =====
    accuracy_plot = create_line_plot(
        thresh_list,
        accuracy_score_list,
        xlabel="Decision Threshold",
        ylabel="Accuracy (%)",
        title="TabPFN (stacking) Accuracy vs Threshold",
    )
    pseudo_loss = [100.0 - f1 for f1 in f1_score_list]
    loss_plot = create_line_plot(
        thresh_list,
        pseudo_loss,
        xlabel="Decision Threshold",
        ylabel="Pseudo Loss (100 - F1)",
        title="TabPFN (stacking) Pseudo-loss vs Threshold",
    )
    result["accuracy_plot"] = accuracy_plot
    result["loss_plot"] = loss_plot

    # ===== ROC 曲線 =====
    fpr, tpr, thresholds = metrics.roc_curve(y_true_bin, y_score, pos_label=1)

    plt.figure()
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    result['roc'] = image_base64
    return result


def explain_with_shap(model, x_test, feature_names, focus_class=1):
    """
    使用 KernelExplainer 產生 SHAP 解釋（跟 train_tabpfn.py 一致）
    """
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        background = x_test[:20]
        explain_data = x_test[:5]

        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(explain_data)

        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # shape = (n_samples, n_features, n_classes)，選正類 (class 1)
            shap_values_for_plot = shap_values[:, :, int(focus_class)]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_for_plot = shap_values[int(focus_class)]
        else:
            shap_values_for_plot = shap_values

        if shap_values_for_plot.ndim != 2:
            raise ValueError(f"shap_values shape is invalid: {shap_values_for_plot.shape}")

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
        plt.savefig(buf, format='png')
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

    except Exception as e:
        result["shap_error"] = str(e)

    return result


def explain_with_lime(model, x_test, y_test, feature_names, focus_class=1):
    """
    使用 LimeTabularExplainer 做局部解釋（跟其他 stacking_* 檔案一致）
    """
    result = {}
    try:
        x_np = np.array(x_test, dtype=np.float32)
        y_np = np.array(y_test)

        lime_explainer = LimeTabularExplainer(
            training_data=x_np,
            mode="classification",
            training_labels=y_np,
            feature_names=feature_names,
            class_names=[f"class_{i}" for i in range(len(model.predict_proba(x_np[:1])[0]))],
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_np[0],  # 第 0 筆樣本
            model.predict_proba,
            num_features=10,
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