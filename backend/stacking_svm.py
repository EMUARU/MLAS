import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn import metrics
import matplotlib.pyplot as plt
import io
import base64
import os
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
from sklearn.svm import SVC

def train_fold(X_train, y_train, X_val, C=1.0, kernel='rbf', focus_class=1):
    # Train SVC on a given fold and return validation predictions and the model.
    svm = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    val_preds = svm.predict_proba(X_val)[:, int(focus_class)]
    return val_preds


def retrain(X, y, feature_names, task_dir, split_value=1.0, save_model=True, model_role='base', C=1.0, kernel='rbf', focus_class=1):
    # Retrain SVC on full data, save model, and generate XAI (SHAP & LIME).
    svm = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(split_value), stratify=y, random_state=30)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        results = evaluate_model(y_test, y_pred, svm, X_test, focus_class=focus_class)
        shap_result = explain_with_shap(svm, X_test, feature_names, focus_class=focus_class)
        results.update(shap_result)
        lime_result = explain_with_lime(svm, X_test, y_test, feature_names, focus_class=focus_class)
        results.update(lime_result)
    else:
        svm.fit(X, y)
        results = None
    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(svm, f"{task_dir}/{model_role}_svm.pkl")
    return results

def predict_full_meta(X, task_dir, focus_class=1):
    model_path = f"{task_dir}/base_svm.pkl"
    model = joblib.load(model_path)
    preds = model.predict_proba(X)[:, int(focus_class)]
    return preds

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
        # 轉成 numpy，避免物件型造成 SHAP/LIME 出錯
        x_test = np.asarray(x_test, dtype=float)

        # 僅取前 20 筆當背景
        background = x_test[:20]

        # 僅解釋前 5 筆
        x_explain = x_test[:5]

        # 若模型有 predict_proba 且為二分類，僅回傳正類機率 (scalar output)
        if hasattr(model, "predict_proba"):
            # 以正類機率作為 f(x)；這樣 link='logit' 就能作用在標量上
            f = lambda X: model.predict_proba(np.asarray(X, dtype=float))[:, int(focus_class)]
            explainer = shap.KernelExplainer(f, background, link="logit")
            shap_values = explainer.shap_values(x_explain, nsamples=100)  # shape: (n_samples, n_features)

        else:
            # 沒有 predict_proba 時，退而求其次用 predict + identity 連結
            f = lambda X: model.predict(np.asarray(X, dtype=float)).astype(float)
            explainer = shap.KernelExplainer(f, background, link="identity")
            shap_values = explainer.shap_values(x_explain, nsamples=100)

        # 平均重要度
        shap_importance = np.abs(shap_values).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
        }

        # beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_explain, feature_names=feature_names, show=False)
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