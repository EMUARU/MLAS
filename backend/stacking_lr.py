from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, log_loss
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
import shap
from lime.lime_tabular import LimeTabularExplainer
import os
import joblib

def train_fold(X_train, y_train, X_val, penalty="l2", alpha=1.0, max_iter=500, focus_class=1):
    logistic_reg = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss='log_loss',
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            warm_start=True,
        )
    )
    logistic_reg.fit(X_train, y_train)
    val_preds = logistic_reg.predict_proba(X_val)[:, int(focus_class)]
    return val_preds

def retrain(X, y, feature_names, task_dir, split_value=1.0, save_model=True, model_role='base', penalty="l2", alpha=1.0, max_iter=500, focus_class=1):
    class_weight = "balanced" if should_use_class_weight(y) else None
    logistic_reg = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss='log_loss',
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            warm_start=True,
            class_weight=class_weight,
        )
    )
    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(split_value), stratify=y, random_state=30)
        loss_list_train = []
        acc_list_train = []
        loss_list_val = []
        acc_list_val = []
        for epoch in range(max_iter):
            logistic_reg.fit(X_train, y_train)
            y_train_proba = logistic_reg.predict_proba(X_train)
            y_val_proba = logistic_reg.predict_proba(X_test)
            loss_train = log_loss(y_train, y_train_proba)
            loss_val = log_loss(y_test, y_val_proba)
            acc_train = accuracy_score(y_train, logistic_reg.predict(X_train))
            acc_val = accuracy_score(y_test, logistic_reg.predict(X_test))
            loss_list_train.append(loss_train)
            acc_list_train.append(acc_train)
            loss_list_val.append(loss_val)
            acc_list_val.append(acc_val)
        evals_result = {
            "training": {"logloss": loss_list_train, "accuracy": acc_list_train},
            "validation": {"logloss": loss_list_val, "accuracy": acc_list_val},
        }
        y_pred = logistic_reg.predict(X_test)
        results = evaluate_model(y_test, y_pred, logistic_reg, X_test, focus_class=focus_class)
        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)
        shap_result = explain_with_shap(logistic_reg, X_test, feature_names, focus_class=focus_class)
        results.update(shap_result)
        lime_result = explain_with_lime(logistic_reg, X_test, y_test, feature_names, focus_class=focus_class)
        results.update(lime_result)
    else:
        logistic_reg.fit(X, y)
        results = None
    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(logistic_reg, f"{task_dir}/{model_role}_lr.pkl")
    return results

def predict_full_meta(X, task_dir, focus_class=1):
    model_path = f"{task_dir}/base_lr.pkl"
    model = joblib.load(model_path)
    preds = model.predict_proba(X)[:, int(focus_class)]
    return preds

def should_use_class_weight(y):
    counts = Counter(y)
    total = sum(counts.values())
    ratios = [v / total for v in counts.values()]
    return max(ratios) - min(ratios) > 0.25  # 自動啟用條件

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

    # recall 門檻掃描
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

    # plot ROC
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
        estimator = model.named_steps["sgdclassifier"]
        scaler = model.named_steps["standardscaler"]
        x_scaled = scaler.transform(x_test)

        explainer = shap.LinearExplainer(estimator, x_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(x_scaled)

        if isinstance(shap_values, list):
            shap_values_for_plot = shap_values[int(focus_class)]
        else:
            shap_values_for_plot = shap_values

        shap_data = x_scaled

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
        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=[f"class_{i}" for i in range(len(model.predict_proba(x_test[:1])[0]))],
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test[0],  # 第0筆樣本
            model.predict_proba,
            num_features=10,
            labels=[int(focus_class)]
        )

        # 儲存圖片到記憶體 buffer
        fig = lime_result.as_pyplot_figure(label=int(focus_class))
        fig.set_size_inches(10, 6)  # 改變圖的尺寸
        fig.tight_layout()  # 自動調整 layout
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)  # 關閉圖避免重疊

        # 同時保留文字格式
        result["lime_example_0"] = lime_result.as_list(label=int(focus_class))

    except Exception as e:
        result["lime_error"] = str(e)
        
    return result

def plot_loss(evals_result):
    logloss_train = evals_result['training']['logloss']
    logloss_val = evals_result['validation']['logloss']
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
    error_train = evals_result['training']['accuracy']
    error_val = evals_result['validation']['accuracy']
    acc_train = [1 - e for e in error_train]
    acc_val = [1 - e for e in error_val]
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