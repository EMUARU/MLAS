#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.utils

工具函式：GPU 偵測、資料型別轉換、ML 特徵準備、輸出檔名推導、metadata 偵測。

"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def detect_gpu():
    """偵測 GPU 並返回是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"{gpu_name} ({gpu_mem:.1f} GB)"
        return False, "No CUDA GPU"
    except Exception as e:
        return False, str(e)


def log_gpu_status(model_name, use_gpu, device_info=""):
    """輸出 GPU 使用狀態到 stderr（不影響 JSON 輸出）"""
    status = "GPU" if use_gpu else "CPU"
    msg = f"[{model_name}] 使用 {status}"
    if device_info:
        msg += f" - {device_info}"
    print(msg, file=sys.stderr)


def get_gpu_memory_usage():
    """獲取當前 GPU 記憶體使用量"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return f"GPU Memory: {allocated:.1f} MB (peak: {peak:.1f} MB)"
    except:
        pass
    return ""


# ============================================================
# 日期欄位轉換輔助函數
# ============================================================


def convert_datetime_columns(df):
    """
    將 DataFrame 中的 datetime 類型欄位轉換為數值類型
    
    這是為了解決 XGBoost 無法處理 datetime64 類型的問題：
    "DataFrame.dtypes for data must be int, float, bool or category"
    
    轉換方式：將日期轉換為時間戳（從 1970-01-01 起的秒數）
    """
    df_converted = df.copy()
    
    # 找出所有 datetime 類型的欄位
    datetime_cols = df_converted.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
    
    for col in datetime_cols:
        try:
            # 轉換為時間戳（秒）
            df_converted[col] = pd.to_datetime(df_converted[col]).astype('int64') // 10**9
        except Exception:
            # 如果轉換失敗，嘗試轉換為字串再編碼
            try:
                df_converted[col] = pd.Categorical(df_converted[col].astype(str)).codes
            except Exception:
                # 最後手段：刪除該欄位
                df_converted = df_converted.drop(columns=[col])
    
    return df_converted


def prepare_ml_features(df, label_column=None):
    """
    準備機器學習特徵（統一處理 datetime、NaN、類別編碼）
    
    用於 evaluate_ml_efficacy 和 evaluate_discriminator_measure
    """
    feature_cols = [c for c in df.columns if c != label_column] if label_column else list(df.columns)
    X = df[feature_cols].copy()
    
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


# ============================================================
# Output naming helpers
# ============================================================


def _infer_output_prefix(file_path: str) -> str:
    """Infer a reasonable output prefix based on the input filename."""
    stem = Path(file_path).stem.lower()
    if "asthma" in stem:
        return "asthma_synthetic_"
    return "synthetic_"


def _make_synthetic_filename(file_path: str, generator_type: str, num_samples: str, num_to_generate: int) -> str:
    prefix = _infer_output_prefix(file_path)
    g = generator_type.lower()

    m = re.match(r"^\s*(\d+)x\s*$", str(num_samples), flags=re.IGNORECASE)
    if m:
        mult = m.group(1)
        return f"{prefix}{g}_{mult}x.csv"

    return f"{prefix}{g}_{num_to_generate}.csv"


def _infer_dataset_suffix(file_path: str) -> str:
    """Infer dataset suffix such as _A40 from the input filename/path."""
    p = str(file_path)
    if "A40" in p or "a40" in p:
        return "_A40"
    return ""


def _make_core_filename(file_path: str, generator_type: str) -> str:
    """Filename for TGANT distilled core synthetic dataset."""
    prefix = _infer_output_prefix(file_path)
    suffix = _infer_dataset_suffix(file_path)
    g = generator_type.lower()
    return f"{prefix}{g}_core1{suffix}.csv"



# ============================================================
# Fidelity 評估函數（對照論文 Fidelity.ipynb）
# ============================================================


def detect_metadata(df):
    """偵測 DataFrame 的 metadata"""
    from sdv.metadata import Metadata
    
    metadata = Metadata.detect_from_dataframe(data=df)
    
    metadata_dict = {"columns": {}}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10:
                metadata_dict["columns"][col] = {"sdtype": "categorical"}
            else:
                metadata_dict["columns"][col] = {"sdtype": "numerical"}
        else:
            metadata_dict["columns"][col] = {"sdtype": "categorical"}
    
    return metadata, metadata_dict


# ============================================================
# 模組層級之自動參數調整 helper 函數
# ------------------------------------------------------------
# 設計變更說明：
# 原本這些 helper 為 nested function（定義於各 train_grouped_* 內部），
# 導致 train_mixed_* 無法呼叫，使得 mixed 模式無自動參數調整。
# 現提升為 module-level，並把原本依賴的外部變數改為函數參數，
# 使 grouped 與 mixed 模式皆能呼叫，保證行為一致。
# ============================================================
