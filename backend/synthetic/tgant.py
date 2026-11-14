#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.tgant

TGANT 數據蒸餾：以雙向 Recall 收斂條件篩選合成樣本（沿用張正昕 (2025) 之演算法）。

"""

import numpy as np
import pandas as pd

from .utils import prepare_ml_features, convert_datetime_columns


def tgant_distill_filter(
    real_df,
    synthetic_df,
    label_column,
    recall_threshold=0.95,
    test_size=0.2,
    max_generations=10,
    random_state=42,
    metric='pos',
    enable_stage2=True,
    stage2_threshold=0.025,
):
    """
    TGANT 蒸餾篩選（兩階段，泛化版，支援 2~N 類別）。

    TGANT 演算法設計：
      第一階段：迭代執行 TRTS / TSTR，直到雙向 Recall 同時達到 recall_threshold
                （嚴格版收斂條件）或達到最大迭代次數。
      第二階段：自最後一代往前回推，分別以前代之合成資料訓練 XGBoost、
                以全部真實資料測試，比對「最後一代」與「當前代」於少數類別
                樣本數量上之相對差異。若差異 < stage2_threshold（預設 2.5%），
                則採用更前一代以保留更多樣本；若差異 ≥ 閾值則停止回退。

    metric: 'pos' = 二元分類時使用 pos_label=1 的 recall（完全對照論文）
            'macro' = 多類別使用 macro recall
            'min' = 使用所有類別中最低的 recall
    enable_stage2: True 時執行第二階段跨代回退（對齊論文）；
                   False 時跳過第二階段。
    stage2_threshold: 第二階段比對閾值（預設 0.025，即 2.5%，對齊論文）。
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import recall_score
    
    if label_column not in real_df.columns or label_column not in synthetic_df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    
    feature_cols = [c for c in real_df.columns if c != label_column]
    common_cols = [c for c in feature_cols if c in synthetic_df.columns]
    
    metric_l = metric.lower()
    
    def calc_recall(y_true, y_pred):
        """計算 recall，根據 metric 設定選擇計算方式"""
        current_labels = np.unique(y_true)
        n_classes = len(current_labels)
        
        if metric_l == 'pos' and n_classes <= 2:
            if 1 in current_labels:
                return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            else:
                return float(recall_score(y_true, y_pred, average='binary', zero_division=0))
        elif metric_l == 'macro':
            return float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        else:  # min
            per_class = []
            for lab in current_labels:
                mask_true = (y_true == lab)
                mask_pred = (y_pred == lab)
                if mask_true.sum() > 0:
                    rec = recall_score(mask_true, mask_pred, zero_division=0)
                    per_class.append(rec)
            return float(min(per_class)) if per_class else 0.0
    
    def prepare_features(df):
        """準備特徵資料（處理 datetime、NaN 和類別編碼）"""
        X = df[common_cols].copy()
        
        # 1. 先轉換 datetime 類型
        X = convert_datetime_columns(X)
        
        # 2. 處理數值欄位的 NaN 和無限值
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            med = X[col].median()
            X[col] = X[col].fillna(med if pd.notna(med) else 0)
        
        # 3. 處理類別欄位
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        return X

    def _minority_count(df):
        """估算 df 中少數類別之樣本數，用於第二階段比對。

        定義：以原始 real_df 之類別分布判定少數類別，
              再於傳入之 df 中計算該類別之樣本數。
        """
        if label_column not in df.columns:
            return 0
        original_counts = real_df[label_column].value_counts()
        if len(original_counts) == 0:
            return 0
        minority_label = original_counts.idxmin()
        return int((df[label_column] == minority_label).sum())

    history = []
    synthetic_dfs = [synthetic_df.copy().reset_index(drop=True)]
    real_dfs = [real_df.copy().reset_index(drop=True)]
    
    gen = 0
    recall_trts, recall_tstr = 0.0, 0.0
    best_gen = 0
    best_min_recall = 0.0
    
    # ============================================================
    # 第一階段：迭代 TRTS / TSTR 至雙向 Recall 收斂
    # ============================================================
    while (recall_trts < recall_threshold or recall_tstr < recall_threshold) and gen < max_generations:
        gen += 1
        current_real = real_dfs[-1]
        current_synth = synthetic_dfs[-1]
        
        # 準備特徵
        X_real = prepare_features(current_real)
        X_synth = prepare_features(current_synth)
        
        # ============================================================
        # TRTS: Train Real, Test Synthetic
        # 使用 real 資料的類別建立映射（確保從 0 開始）
        # ============================================================
        real_labels = current_real[label_column].values
        synth_labels = current_synth[label_column].values
        
        # 用 real 的類別建立映射
        unique_real = sorted(set(real_labels))
        real_label_to_idx = {label: idx for idx, label in enumerate(unique_real)}
        
        # 編碼 real（保證從 0 開始連續）
        y_real = np.array([real_label_to_idx[v] for v in real_labels])
        
        # 編碼 synth（只編碼 real 中存在的類別，不存在的標記為 -1）
        y_synth = np.array([real_label_to_idx.get(v, -1) for v in synth_labels])
        
        # 訓練 TRTS 分類器
        clf_trts = XGBClassifier(
            n_estimators=200, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, use_label_encoder=False, eval_metric='logloss'
        )
        clf_trts.fit(X_real, y_real)
        y_pred_trts = clf_trts.predict(X_synth)
        
        # 計算 recall（只考慮有效的樣本）
        valid_mask = (y_synth >= 0)
        if valid_mask.sum() > 0:
            recall_trts = calc_recall(y_synth[valid_mask], y_pred_trts[valid_mask])
        else:
            recall_trts = 0.0
        
        # 篩選 synthetic: 保留預測正確的樣本
        mask_synth = (y_synth == y_pred_trts) & valid_mask
        filtered_synth = current_synth.loc[mask_synth].reset_index(drop=True)
        
        # 防止過度篩選
        min_rows = max(50, int(len(synthetic_df) * 0.01))
        if len(filtered_synth) < min_rows:
            filtered_synth = current_synth.copy()
        
        # ============================================================
        # TSTR: Train Synthetic, Test Real
        # 使用 synth 資料的類別建立映射
        # ============================================================
        X_synth_filtered = prepare_features(filtered_synth)
        
        synth_filtered_labels = filtered_synth[label_column].values
        real_labels_tstr = current_real[label_column].values
        
        # 用篩選後的 synth 類別建立映射
        unique_synth = sorted(set(synth_filtered_labels))
        synth_label_to_idx = {label: idx for idx, label in enumerate(unique_synth)}
        
        # 編碼
        y_synth_filtered = np.array([synth_label_to_idx[v] for v in synth_filtered_labels])
        y_real_tstr = np.array([synth_label_to_idx.get(v, -1) for v in real_labels_tstr])
        
        # 訓練 TSTR 分類器
        clf_tstr = XGBClassifier(
            n_estimators=200, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, use_label_encoder=False, eval_metric='logloss'
        )
        clf_tstr.fit(X_synth_filtered, y_synth_filtered)
        y_pred_tstr = clf_tstr.predict(X_real)
        
        # 計算 recall
        valid_mask_tstr = (y_real_tstr >= 0)
        if valid_mask_tstr.sum() > 0:
            recall_tstr = calc_recall(y_real_tstr[valid_mask_tstr], y_pred_tstr[valid_mask_tstr])
        else:
            recall_tstr = 0.0
        
        # 篩選 real: 保留預測正確的樣本
        mask_real = (y_real_tstr == y_pred_tstr) & valid_mask_tstr
        filtered_real = current_real.loc[mask_real].reset_index(drop=True)
        
        min_rows_real = max(50, int(len(real_df) * 0.01))
        if len(filtered_real) < min_rows_real:
            filtered_real = current_real.copy()
        
        synthetic_dfs.append(filtered_synth)
        real_dfs.append(filtered_real)
        
        min_recall = min(recall_trts, recall_tstr)
        if min_recall > best_min_recall:
            best_min_recall = min_recall
            best_gen = gen
        
        history.append({
            'generation': gen,
            'recall_trts': round(recall_trts, 4),
            'recall_tstr': round(recall_tstr, 4),
            'synthetic_rows': len(filtered_synth),
            'real_rows': len(filtered_real),
            'converged': recall_trts >= recall_threshold and recall_tstr >= recall_threshold
        })
    
    converged = recall_trts >= recall_threshold and recall_tstr >= recall_threshold
    
    # ============================================================
    # 確定第一階段選用之代
    #   - 若收斂：採用最後一代
    #   - 若未收斂：採用 best_gen（迭代過程中雙向 Recall 最佳之代）
    # ============================================================
    if converged:
        stage1_used_gen = gen
    else:
        stage1_used_gen = best_gen if best_gen > 0 else len(synthetic_dfs) - 1

    # ============================================================
    # 第二階段：跨代回退（TGANT_sec2）
    #
    # 自第一階段選用之代往前回推，逐代以前代合成資料訓練 XGBoost、以全部
    # 真實資料測試，比對「最後一代」與「當前代」之少數類別樣本數量相對差異。
    # 差異 < stage2_threshold 則採用更前一代以保留更多樣本；
    # 差異 >= stage2_threshold 則停止回退。
    # ============================================================
    stage2_history = []
    stage2_real_residuals = []  # list of {'gen': int, 'residuals': df}
    stage2_used_gen = stage1_used_gen

    if enable_stage2 and stage1_used_gen >= 2:
        full_real = real_dfs[0].copy()
        X_full_real = prepare_features(full_real)
        full_real_labels = full_real[label_column].values

        # 自 stage1_used_gen 代往前回推到第 2 代（不含第 1 代）
        for i in range(stage1_used_gen, 1, -1):
            prev_synth = synthetic_dfs[i].copy()
            X_prev_synth = prepare_features(prev_synth)
            prev_synth_labels = prev_synth[label_column].values

            unique_prev_synth = sorted(set(prev_synth_labels))
            prev_synth_label_to_idx = {label: idx for idx, label in enumerate(unique_prev_synth)}
            y_prev_synth = np.array([prev_synth_label_to_idx[v] for v in prev_synth_labels])

            y_full_real_for_pred = np.array(
                [prev_synth_label_to_idx.get(v, -1) for v in full_real_labels]
            )

            clf_stage2 = XGBClassifier(
                n_estimators=200, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, use_label_encoder=False, eval_metric='logloss'
            )
            clf_stage2.fit(X_prev_synth, y_prev_synth)
            y_pred_stage2 = clf_stage2.predict(X_full_real)

            valid_s2 = (y_full_real_for_pred >= 0)
            mask_correct = (y_full_real_for_pred == y_pred_stage2) & valid_s2
            real_residuals_i = full_real.loc[mask_correct].reset_index(drop=True)
            stage2_real_residuals.append({'gen': i, 'residuals': real_residuals_i})

            stage2_history.append({
                'generation': i,
                'minority_count_in_residuals': _minority_count(real_residuals_i),
                'residual_rows': len(real_residuals_i),
            })

            if len(stage2_real_residuals) >= 2:
                last_minority = _minority_count(stage2_real_residuals[0]['residuals'])
                curr_minority = _minority_count(stage2_real_residuals[-1]['residuals'])
                if curr_minority > 0:
                    rel_diff = abs(last_minority - curr_minority) / curr_minority
                else:
                    rel_diff = float('inf')

                stage2_history[-1]['relative_diff_vs_last'] = round(rel_diff, 4)

                if rel_diff >= stage2_threshold:
                    break
                else:
                    stage2_used_gen = i

    # ============================================================
    # 最終輸出
    # ============================================================
    used_gen = stage2_used_gen
    if used_gen < len(synthetic_dfs):
        final_synth = synthetic_dfs[used_gen]
    else:
        final_synth = synthetic_dfs[-1]

    final_real = None
    if enable_stage2 and stage2_real_residuals:
        for entry in stage2_real_residuals:
            if entry['gen'] == used_gen:
                final_real = entry['residuals']
                break
    if final_real is None:
        final_real = real_dfs[used_gen] if used_gen < len(real_dfs) else real_dfs[-1]

    summary = {
        'converged': converged,
        'total_generations': gen,
        'stage1_used_generation': stage1_used_gen,
        'stage2_used_generation': stage2_used_gen,
        'used_generation': used_gen,
        'final_recall_trts': round(recall_trts, 4),
        'final_recall_tstr': round(recall_tstr, 4),
        'original_synthetic_rows': len(synthetic_df),
        'final_synthetic_rows': len(final_synth),
        'original_real_rows': len(real_df),
        'final_real_rows': len(final_real),
        'threshold': recall_threshold,
        'stage2_threshold': stage2_threshold,
        'stage2_enabled': bool(enable_stage2),
        'metric': metric,
    }

    full_history = {
        'stage1': history,
        'stage2': stage2_history,
    }

    return final_synth, final_real, full_history, summary


# ============================================================
# Metadata 偵測
# ============================================================
