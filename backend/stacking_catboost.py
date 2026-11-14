from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import base64
import io
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
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
from sklearn.model_selection import train_test_split
import os
import joblib

def _detect_catboost_params(y):
    """自動偵測 binary / multi-class，回傳 (loss_function, eval_metric, custom_metric)"""
    num_class = int(len(np.unique(y)))
    if num_class <= 2:
        return "Logloss", "Logloss", ["Accuracy"]
    else:
        return "MultiClass", "MultiClass", ["Accuracy"]

def train_fold(X_train, y_train, X_val, iterations=500, learning_rate=0.009, depth=6, focus_class=1):
    loss_function, eval_metric, custom_metric = _detect_catboost_params(y_train)
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        eval_metric=eval_metric,
        verbose=0,
    )
    model.fit(
        X_train, y_train,
        plot=False
    )
    val_preds = model.predict_proba(X_val)[:, int(focus_class)]
    return val_preds

def retrain(X, y, feature_names, task_dir, split_value=1.0, save_model=True, model_role='base', iterations=500, learning_rate=0.009, depth=6, focus_class=1):
    loss_function, eval_metric, custom_metric = _detect_catboost_params(y)
    if split_value != '1.0':
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(split_value), stratify=y, random_state=30)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )
        evals_result = model.get_evals_result()
        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred, model, X_test, focus_class=focus_class)
        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)
        shap_result = explain_with_shap(model, X_test, feature_names, focus_class=focus_class)
        results.update(shap_result)
        lime_result = explain_with_lime(model, X_test, y_test, feature_names, focus_class=focus_class)
        results.update(lime_result)
    else:
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
            eval_metric=eval_metric,
            verbose=0,
        )
        model.fit(X, y)
        results = None
    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(model, f"{task_dir}/{model_role}_catboost.pkl")
    return results

def predict_full_meta(X, task_dir, focus_class=1):
    model_path = f"{task_dir}/base_catboost.pkl"
    model = CatBoostClassifier()
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