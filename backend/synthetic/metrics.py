#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.metrics

合成資料品質評估：精確性（KL/JS 散度）、隱私性（DCR）、實用性（TSTR/TRTR/TRTS）與 Discriminator Measure。

"""

import numpy as np
import pandas as pd

from .utils import prepare_ml_features, convert_datetime_columns


def compute_js_divergence(p, q, base=np.e):
    """計算 Jensen-Shannon Divergence"""
    from scipy.spatial.distance import jensenshannon
    p = np.asarray(p, dtype=float) + 1e-8
    q = np.asarray(q, dtype=float) + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    return jensenshannon(p, q, base=base) ** 2


def compute_kl_divergence(p, q, base=np.e):
    """計算 Kullback-Leibler Divergence"""
    from scipy.special import rel_entr
    p = np.asarray(p, dtype=float) + 1e-8
    q = np.asarray(q, dtype=float) + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    kl = np.sum(rel_entr(p, q))
    if base != np.e:
        kl /= np.log(base)
    return kl


def divergence_for_column(df1, df2, column, bins=10, method='js'):
    """Calculate JS or KL divergence for a single column."""
    compute_fn = compute_js_divergence if method == 'js' else compute_kl_divergence
    
    if pd.api.types.is_numeric_dtype(df1[column]):
        combined = pd.concat([df1[column], df2[column]])
        bin_edges = np.histogram_bin_edges(combined.dropna(), bins=bins)
        p_hist, _ = np.histogram(df1[column].dropna(), bins=bin_edges)
        q_hist, _ = np.histogram(df2[column].dropna(), bins=bin_edges)
    else:
        p_hist = df1[column].value_counts().sort_index()
        q_hist = df2[column].value_counts().sort_index()
        all_keys = sorted(set(p_hist.index).union(set(q_hist.index)))
        p_hist = p_hist.reindex(all_keys, fill_value=0)
        q_hist = q_hist.reindex(all_keys, fill_value=0)
    
    return compute_fn(p_hist, q_hist)


def evaluate_fidelity(real_df, synthetic_df, metadata):
    """評估合成資料的 Fidelity（對照論文 Fidelity.ipynb）"""
    from sdv.evaluation.single_table import evaluate_quality
    
    result = {}
    
    try:
        quality_report = evaluate_quality(real_df, synthetic_df, metadata, verbose=False)
        props = quality_report.get_properties()
        result['single_column_similarity'] = float(props['Score'][0]) if len(props) > 0 else None
        result['multi_column_similarity'] = float(props['Score'][1]) if len(props) > 1 else None
    except Exception as e:
        result['single_column_similarity'] = None
        result['multi_column_similarity'] = None
        result['sdv_error'] = str(e)
    
    # KL / JS Divergence
    common_cols = sorted(set(real_df.columns).intersection(synthetic_df.columns))
    kl_scores = {}
    js_scores = {}
    
    for col in common_cols:
        try:
            kl_scores[col] = divergence_for_column(real_df, synthetic_df, col, method='kl')
            js_scores[col] = divergence_for_column(real_df, synthetic_df, col, method='js')
        except Exception:
            continue
    
    if kl_scores:
        result['kl_divergence'] = float(np.mean(list(kl_scores.values())))
    if js_scores:
        result['js_divergence'] = float(np.mean(list(js_scores.values())))
    
    result['details'] = {
        'kl_by_column': {k: float(v) for k, v in kl_scores.items()},
        'js_by_column': {k: float(v) for k, v in js_scores.items()}
    }
    
    return result


# ============================================================
# Privacy 評估函數（對照論文 Privacy.ipynb）
# ============================================================


def evaluate_privacy(real_df, synthetic_df, metadata_dict, label_column=None, grouped=False):
    """評估合成資料的 Privacy（對照論文 Privacy.ipynb）"""
    from sdmetrics.single_table import DCRBaselineProtection
    
    result = {}
    
    try:
        # 先計算整體的 Median DCR（無論是否分群）
        try:
            # 移除標籤欄位進行整體評估
            real_for_dcr = real_df.drop(columns=[label_column], errors='ignore') if label_column else real_df
            synth_for_dcr = synthetic_df.drop(columns=[label_column], errors='ignore') if label_column else synthetic_df
            
            # 確保欄位一致
            common_cols = list(set(real_for_dcr.columns) & set(synth_for_dcr.columns))
            real_for_dcr = real_for_dcr[common_cols]
            synth_for_dcr = synth_for_dcr[common_cols]
            
            meta_for_dcr = {k: v for k, v in metadata_dict.items() if k != label_column}
            if 'columns' in meta_for_dcr:
                meta_for_dcr['columns'] = {k: v for k, v in meta_for_dcr['columns'].items() if k != label_column}
            
            subsample_overall = min(500, len(synth_for_dcr), len(real_for_dcr))
            breakdown_overall = DCRBaselineProtection.compute_breakdown(
                real_data=real_for_dcr,
                synthetic_data=synth_for_dcr,
                metadata=meta_for_dcr,
                num_rows_subsample=subsample_overall,
                num_iterations=5
            )
            result['median_dcr_synthetic'] = float(breakdown_overall['median_DCR_to_real_data']['synthetic_data'])
            result['median_dcr_baseline'] = float(breakdown_overall['median_DCR_to_real_data']['random_data_baseline'])
        except Exception as e:
            result['median_dcr_synthetic'] = None
            result['median_dcr_baseline'] = None
        
        if grouped and label_column and label_column in real_df.columns:
            labels = list(real_df[label_column].dropna().unique())
            per_class_scores = {}
            
            meta_no_label = {k: v for k, v in metadata_dict.items() if k != label_column}
            if 'columns' in meta_no_label:
                meta_no_label['columns'] = {k: v for k, v in meta_no_label['columns'].items() if k != label_column}
            
            for lab in labels:
                real_class = real_df[real_df[label_column] == lab].drop(columns=[label_column], errors='ignore')
                synth_class = synthetic_df[synthetic_df[label_column] == lab].drop(columns=[label_column], errors='ignore')
                
                if len(synth_class) == 0 or len(real_class) == 0:
                    continue
                
                common_cols = list(set(real_class.columns) & set(synth_class.columns))
                real_class = real_class[common_cols]
                synth_class = synth_class[common_cols]
                
                subsample = min(500, len(synth_class), len(real_class))
                
                try:
                    breakdown = DCRBaselineProtection.compute_breakdown(
                        real_data=real_class,
                        synthetic_data=synth_class,
                        metadata=meta_no_label,
                        num_rows_subsample=subsample,
                        num_iterations=5
                    )
                    per_class_scores[str(lab)] = float(breakdown['score'])
                except Exception as e:
                    per_class_scores[str(lab)] = None
            
            if per_class_scores:
                valid = [v for v in per_class_scores.values() if v is not None]
                result['score'] = float(np.mean(valid)) if valid else None
                result['per_class_scores'] = per_class_scores
                
                # 處理二元分類的正負類標籤（支援多種格式）
                label_set = set(str(l) for l in labels)
                if len(labels) == 2:
                    if {'0', '1'} == label_set or {0, 1} == set(labels):
                        pos_key = '1' if '1' in per_class_scores else 1
                        neg_key = '0' if '0' in per_class_scores else 0
                        result['score_positive'] = per_class_scores.get(str(pos_key))
                        result['score_negative'] = per_class_scores.get(str(neg_key))
        else:
            # 整體評估
            subsample = min(500, len(synthetic_df), len(real_df))
            breakdown = DCRBaselineProtection.compute_breakdown(
                real_data=real_df,
                synthetic_data=synthetic_df,
                metadata=metadata_dict,
                num_rows_subsample=subsample,
                num_iterations=5
            )
            result['score'] = float(breakdown['score'])
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ============================================================
# ML Efficacy 評估（對照論文 Utility--ML_Efficacy.ipynb）
# ============================================================


def evaluate_ml_efficacy(real_df, synthetic_df, label_column):
    """評估 ML Efficacy（TRTR vs TSTR）"""
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                                 f1_score, confusion_matrix, roc_auc_score)
    
    result = {}
    
    try:
        if label_column not in real_df.columns or label_column not in synthetic_df.columns:
            return {'error': f'Label column {label_column} not found'}
        
        feature_cols = [c for c in real_df.columns if c != label_column]
        common_cols = [c for c in feature_cols if c in synthetic_df.columns]
        
        X_real = real_df[common_cols].copy()
        y_real = real_df[label_column].values
        X_synth = synthetic_df[common_cols].copy()
        y_synth = synthetic_df[label_column].values
        
        # ============================================================
        # 1. 先轉換 datetime 類型（修復 XGBoost datetime64 錯誤）
        # ============================================================
        X_real = convert_datetime_columns(X_real)
        X_synth = convert_datetime_columns(X_synth)
        
        # 確保兩者有相同的欄位（日期轉換可能會刪除某些欄位）
        common_cols_after = [c for c in X_real.columns if c in X_synth.columns]
        X_real = X_real[common_cols_after]
        X_synth = X_synth[common_cols_after]
        
        # ============================================================
        # 2. 處理數值欄位的 NaN 和無限值
        # ============================================================
        for col in X_real.select_dtypes(include=[np.number]).columns:
            X_real[col] = X_real[col].replace([np.inf, -np.inf], np.nan)
            med = X_real[col].median()
            fill_val = med if pd.notna(med) else 0
            X_real[col] = X_real[col].fillna(fill_val)
            X_synth[col] = X_synth[col].replace([np.inf, -np.inf], np.nan).fillna(fill_val)
        
        # ============================================================
        # 3. 處理類別欄位
        # ============================================================
        for col in X_real.select_dtypes(include=['object', 'category']).columns:
            X_real[col] = pd.Categorical(X_real[col]).codes
            X_synth[col] = pd.Categorical(X_synth[col]).codes
        
        # ============================================================
        # 為 TRTR 準備：只用 real 的類別編碼
        # ============================================================
        unique_real = sorted(set(y_real))
        real_label_to_idx = {label: idx for idx, label in enumerate(unique_real)}
        y_real_encoded = np.array([real_label_to_idx[v] for v in y_real])
        
        # Train/Test split（用於 TRTR 和 TSTR 的測試集）
        X_real_train, X_real_test, y_real_train, y_real_test, idx_train, idx_test = train_test_split(
            X_real, y_real_encoded, np.arange(len(y_real)), 
            test_size=0.2, random_state=42, stratify=y_real_encoded
        )
        
        # 原始標籤（用於 TSTR 重新編碼）
        y_real_test_original = y_real[idx_test]
        
        classes = unique_real
        is_binary = len(classes) == 2
        
        def build_xgb():
            return XGBClassifier(
                n_estimators=100, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
        
        def compute_metrics_binary(y_true, y_pred, y_prob):
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # 處理只有一個類別的情況
                return {
                    'accuracy': round(accuracy_score(y_true, y_pred), 4),
                    'recall': 0, 'specificity': 0, 'precision': 0,
                    'npv': 0, 'f1': 0, 'f2': 0, 'gmean': 0, 'auc': None,
                    'confusion_matrix': cm.tolist()
                }
            
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = precision_score(y_true, y_pred, zero_division=0)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f2 = (5 * precision * recall) / ((4 * precision) + recall) if (4 * precision + recall) > 0 else 0
            gmean = (recall * specificity) ** 0.5
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = None
            
            return {
                'accuracy': round(accuracy, 4),
                'recall': round(recall, 4),
                'specificity': round(specificity, 4),
                'precision': round(precision, 4),
                'npv': round(npv, 4),
                'f1': round(f1, 4),
                'f2': round(f2, 4),
                'gmean': round(gmean, 4),
                'auc': round(auc, 4) if auc else None,
                'confusion_matrix': cm.tolist()
            }
        
        def compute_metrics_multiclass(y_true, y_pred, y_prob, class_list):
            accuracy = accuracy_score(y_true, y_pred)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            per_class_recall = {}
            recall_values = []  # 收集各類別 recall 用於計算 G-mean
            for idx, c in enumerate(class_list):
                mask = (y_true == idx)
                if mask.sum() > 0:
                    rec = recall_score(mask, y_pred == idx, zero_division=0)
                    per_class_recall[str(c)] = round(rec, 4)
                    recall_values.append(rec)
            
            # 計算多分類 G-mean（各類別 Recall 的幾何平均）
            if recall_values and all(r > 0 for r in recall_values):
                gmean = np.prod(recall_values) ** (1 / len(recall_values))
            else:
                gmean = 0.0
            
            try:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except:
                auc = None
            
            return {
                'accuracy': round(accuracy, 4),
                'recall_macro': round(recall_macro, 4),
                'precision_macro': round(precision_macro, 4),
                'f1_macro': round(f1_macro, 4),
                'gmean': round(gmean, 4),
                'auc': round(auc, 4) if auc else None,
                'per_class_recall': per_class_recall
            }
        
        # ============================================================
        # TRTR (Baseline): Train Real, Test Real
        # ============================================================
        xgb_trtr = build_xgb()
        xgb_trtr.fit(X_real_train, y_real_train)
        y_pred_trtr = xgb_trtr.predict(X_real_test)
        y_prob_trtr = xgb_trtr.predict_proba(X_real_test)[:, 1] if is_binary else xgb_trtr.predict_proba(X_real_test)
        
        # ============================================================
        # TSTR: Train Synthetic, Test Real
        # 用 synth 自己的類別編碼（確保從 0 開始）
        # ============================================================
        unique_synth = sorted(set(y_synth))
        synth_label_to_idx = {label: idx for idx, label in enumerate(unique_synth)}
        y_synth_encoded = np.array([synth_label_to_idx[v] for v in y_synth])
        
        # 對 real_test 用 synth 的映射編碼（不存在的類別標記為 -1）
        y_real_test_for_tstr = np.array([synth_label_to_idx.get(v, -1) for v in y_real_test_original])
        
        # 只保留有效的測試樣本（類別在 synth 中存在）
        valid_mask = (y_real_test_for_tstr >= 0)
        
        if valid_mask.sum() > 0:
            xgb_tstr = build_xgb()
            xgb_tstr.fit(X_synth, y_synth_encoded)
            
            X_real_test_valid = X_real_test.iloc[valid_mask]
            y_real_test_valid = y_real_test_for_tstr[valid_mask]
            
            y_pred_tstr = xgb_tstr.predict(X_real_test_valid)
            y_prob_tstr = xgb_tstr.predict_proba(X_real_test_valid)[:, 1] if len(unique_synth) == 2 else xgb_tstr.predict_proba(X_real_test_valid)
            
            is_binary_tstr = len(unique_synth) == 2
            
            if is_binary:
                result['baseline'] = compute_metrics_binary(y_real_test, y_pred_trtr, y_prob_trtr)
            else:
                result['baseline'] = compute_metrics_multiclass(y_real_test, y_pred_trtr, y_prob_trtr, classes)
            
            if is_binary_tstr:
                result['tstr'] = compute_metrics_binary(y_real_test_valid, y_pred_tstr, y_prob_tstr)
            else:
                result['tstr'] = compute_metrics_multiclass(y_real_test_valid, y_pred_tstr, y_prob_tstr, unique_synth)
        else:
            # 沒有有效的測試樣本
            if is_binary:
                result['baseline'] = compute_metrics_binary(y_real_test, y_pred_trtr, y_prob_trtr)
            else:
                result['baseline'] = compute_metrics_multiclass(y_real_test, y_pred_trtr, y_prob_trtr, classes)
            result['tstr'] = {'error': 'No valid test samples (synth classes do not overlap with real)'}
        
        result['classes'] = [str(c) for c in classes]
        result['is_binary'] = is_binary
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ============================================================
# Discriminator Measure（對照論文 Utility--Discriminator_Measure.ipynb）
# ============================================================


def evaluate_discriminator_measure(real_df, synthetic_df, label_column=None, grouped=False, n_iterations=5):
    """
    評估 Discriminator Measure（對照論文 Utility--Discriminator_Measure.ipynb）
    
    論文方法：
    1. 將合成資料標記為 0（陰性），真實資料標記為 1（陽性）
    2. 依照 1:1 比例建立混合資料集（從合成資料隨機抽取與真實資料相同數量）
    3. 分別使用 Random Forest 與 XGBoost 訓練分類器
    4. 將混合資料集依照 8:2 比例劃分為訓練集與測試集
    5. 重複隨機抽取 5 次，進行 5 次實驗
    6. 將所有結果加總後，計算整體的 Accuracy、Recall
    
    判讀：Accuracy 越接近 50% 代表越難區分真實與合成資料
    """
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
    
    result = {}
    
    try:
        if label_column and label_column in real_df.columns:
            feature_cols = [c for c in real_df.columns if c != label_column]
        else:
            feature_cols = list(real_df.columns)
        
        common_cols = [c for c in feature_cols if c in synthetic_df.columns]
        
        def run_discriminator(real_data, synth_data, prefix=''):
            """論文方法：每次迭代重新從合成資料抽樣"""
            X_real_all = real_data[common_cols].copy()
            X_synth_all = synth_data[common_cols].copy()
            
            # ============================================================
            # 1. 先轉換 datetime 類型（修復 XGBoost/RF datetime64 錯誤）
            # ============================================================
            X_real_all = convert_datetime_columns(X_real_all)
            X_synth_all = convert_datetime_columns(X_synth_all)
            
            # 確保兩者有相同的欄位
            common_cols_after = [c for c in X_real_all.columns if c in X_synth_all.columns]
            X_real_all = X_real_all[common_cols_after]
            X_synth_all = X_synth_all[common_cols_after]
            
            # ============================================================
            # 2. 處理數值欄位的 NaN 和無限值
            # ============================================================
            for col in X_real_all.select_dtypes(include=[np.number]).columns:
                X_real_all[col] = X_real_all[col].replace([np.inf, -np.inf], np.nan)
                med = X_real_all[col].median()
                fill_val = med if pd.notna(med) else 0
                X_real_all[col] = X_real_all[col].fillna(fill_val)
                X_synth_all[col] = X_synth_all[col].replace([np.inf, -np.inf], np.nan).fillna(fill_val)
            
            # ============================================================
            # 3. 處理類別欄位
            # ============================================================
            for col in X_real_all.select_dtypes(include=['object', 'category']).columns:
                X_real_all[col] = pd.Categorical(X_real_all[col]).codes
                X_synth_all[col] = pd.Categorical(X_synth_all[col]).codes
            
            n_real = len(X_real_all)
            n_synth = len(X_synth_all)
            
            # 累計混淆矩陣（論文：將所有結果加總）
            rf_cm_total = np.zeros((2, 2))
            xgb_cm_total = np.zeros((2, 2))
            
            # 重複 5 次實驗，每次重新抽樣
            for i in range(n_iterations):
                # 【論文關鍵】每次從合成資料重新隨機抽取
                if n_synth >= n_real:
                    X_synth = X_synth_all.sample(n=n_real, random_state=42+i)
                    X_real = X_real_all.copy()
                else:
                    X_synth = X_synth_all.copy()
                    X_real = X_real_all.sample(n=n_synth, random_state=42+i)
                
                # 標記：合成=0（陰性），真實=1（陽性）
                X_real['_is_real'] = 1
                X_synth['_is_real'] = 0
                
                # 合併建立混合資料集
                combined = pd.concat([X_real, X_synth], ignore_index=True)
                y = combined['_is_real']
                X = combined.drop(columns=['_is_real'])
                
                # 8:2 劃分訓練集與測試集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42+i, stratify=y
                )
                
                # Random Forest
                rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42+i)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                rf_cm_total += confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
                
                # XGBoost
                xgb = XGBClassifier(
                    n_estimators=100, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                    random_state=42+i, use_label_encoder=False, eval_metric='logloss'
                )
                xgb.fit(X_train, y_train)
                y_pred_xgb = xgb.predict(X_test)
                xgb_cm_total += confusion_matrix(y_test, y_pred_xgb, labels=[0, 1])
            
            # 從累計混淆矩陣計算整體指標
            def calc_metrics_from_cm(cm):
                tn, fp, fn, tp = cm.ravel()
                total = tn + fp + fn + tp
                accuracy = (tp + tn) / total if total > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                return {
                    'accuracy': round(accuracy, 4),
                    'recall': round(recall, 4)
                }
            
            return {
                f'{prefix}random_forest': calc_metrics_from_cm(rf_cm_total),
                f'{prefix}xgboost': calc_metrics_from_cm(xgb_cm_total)
            }
        
        if grouped and label_column and label_column in real_df.columns:
            labels = list(real_df[label_column].dropna().unique())
            per_class = {}
            
            for lab in labels:
                real_class = real_df[real_df[label_column] == lab]
                synth_class = synthetic_df[synthetic_df[label_column] == lab]
                
                if len(synth_class) > 0 and len(real_class) > 0:
                    per_class[str(lab)] = run_discriminator(real_class, synth_class)
            
            result['per_class'] = per_class
            
            # 計算整體平均
            rf_accs = [v['random_forest']['accuracy'] for v in per_class.values() if 'random_forest' in v]
            xgb_accs = [v['xgboost']['accuracy'] for v in per_class.values() if 'xgboost' in v]
            
            result['random_forest'] = {'accuracy': round(np.mean(rf_accs), 4) if rf_accs else None}
            result['xgboost'] = {'accuracy': round(np.mean(xgb_accs), 4) if xgb_accs else None}
        else:
            overall = run_discriminator(real_df, synthetic_df)
            result['random_forest'] = overall['random_forest']
            result['xgboost'] = overall['xgboost']
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ============================================================
# TGANT 蒸餾篩選（對照論文 TGANT_CTGAN.ipynb / TGANT_TVAE.ipynb）
# ============================================================
