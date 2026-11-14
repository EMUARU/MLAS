#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.generators

合成資料生成器訓練：CTGAN / TVAE / TabDDPM 於 grouped 與 mixed 兩種模式之訓練流程。

"""

import os
import sys
import logging
import warnings
import contextlib

import numpy as np
import pandas as pd

from .utils import (
    detect_gpu, log_gpu_status, get_gpu_memory_usage,
    convert_datetime_columns, prepare_ml_features, detect_metadata,
    _infer_output_prefix, _make_synthetic_filename,
    _infer_dataset_suffix, _make_core_filename,
)
from .auto_params import (
    _auto_epochs_ctgan, _auto_dims_ctgan, _auto_discriminator_steps_ctgan,
    _auto_batch_ctgan, _get_batch_sizes_to_try,
    _auto_epochs_tvae, _auto_dims_tvae, _auto_batch_tvae,
    _get_batch_sizes_to_try_tvae,
    _auto_n_iter_tabddpm, _auto_batch_tabddpm, _auto_timesteps_tabddpm,
)


def train_grouped_ctgan(
    df,
    label_column,
    metadata_dict,
    epochs,
    batch_size,
    discriminator_steps,
    generator_dim,
    discriminator_dim,
    embedding_dim,
    generator_lr,
    discriminator_lr,
    num_to_generate,
    *,
    use_auto_params: bool = True,
    epochs_by_label: dict = None,
):
    """
    分群訓練 CTGAN（泛化版：支援 2~N 類別）。
    """
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    df = df.copy()
    label_dtype = df[label_column].dtype
    original_labels_map = {}
    
    if pd.api.types.is_float_dtype(label_dtype):
        df[label_column] = df[label_column].astype(str)
        for orig_val in df[label_column].unique():
            try:
                original_labels_map[orig_val] = float(orig_val)
            except:
                original_labels_map[orig_val] = orig_val

    labels = list(pd.Series(df[label_column]).dropna().unique())
    if len(labels) < 2:
        raise ValueError(f"grouped 模式至少需要 2 個類別，目前僅 {len(labels)} 個")

    vc = df[label_column].value_counts(dropna=False).to_dict()
    total = int(sum(vc.values()))
    allocations = {}
    remain = int(num_to_generate)
    for i, lab in enumerate(labels):
        n_lab = int(vc.get(lab, 0))
        if i < len(labels) - 1:
            k = int(round(num_to_generate * (n_lab / max(total, 1))))
            k = max(0, k)
            allocations[lab] = k
            remain -= k
        else:
            allocations[lab] = max(0, remain)

    synth_models = {}
    loss_history = {}

    # ============================================================
    # 自動參數調整 (基於 NeurIPS 2019 論文和 SDV 官方最佳實踐)
    # ============================================================
    # 論文: "Modeling Tabular data using Conditional GAN"
    # - Xu et al., NeurIPS 2019
    # 
    # SDV 官方預設: epochs=300, batch_size=500, dims=(256,256)
    # 論文驗證: 小數據集需要更多 epochs 和較小網路
    # 
    # 【關鍵發現】CTGAN 對以下參數敏感:
    # - discriminator_decay: 1e-6 (SDV 官方修正)
    # - discriminator_steps: 1-5 (根據數據集大小調整)
    # - batch_size: 必須是 10 的倍數，且 < 樣本數
    # ============================================================
    
    # ------------------------------------------------------------
    # 自動參數調整：本函數內部呼叫處已使用 _auto_epochs_ctgan(n) 等 nested name，
    # 故此處保留同名「薄包裝」函數，內部轉發至 module-level 實作
    # （見檔案頂部 _auto_*_ctgan / _get_batch_sizes_to_try），
    # 確保 grouped 與 mixed 模式採用相同邏輯，避免行為分歧。
    # ------------------------------------------------------------
    def _auto_epochs_ctgan_local(n: int) -> int:
        return _auto_epochs_ctgan(n, epochs, use_auto_params)

    def _auto_dims_ctgan_local(n: int) -> tuple:
        return _auto_dims_ctgan(n, generator_dim, use_auto_params)

    def _auto_discriminator_steps_ctgan_local(n: int) -> int:
        return _auto_discriminator_steps_ctgan(n, discriminator_steps, use_auto_params)

    def _auto_batch_ctgan_local(n: int) -> int:
        return _auto_batch_ctgan(n, batch_size)

    def _get_batch_sizes_to_try_local(n: int, preferred_bs: int) -> list:
        return _get_batch_sizes_to_try(n, preferred_bs)

    skipped_classes = {}
    MIN_SAMPLES_FOR_CTGAN = 10
    
    for lab in labels:
        df_lab = df[df[label_column] == lab].drop(columns=[label_column]).reset_index(drop=True)
        n = int(len(df_lab))
        if n <= 0:
            continue
        
        if n < MIN_SAMPLES_FOR_CTGAN:
            skipped_classes[str(lab)] = {
                "reason": f"樣本數不足 (n={n}, 最少需要 {MIN_SAMPLES_FOR_CTGAN})",
                "n_rows": n
            }
            continue
        
        df_lab = df_lab.dropna(axis=1, how='all')
        
        for col in df_lab.select_dtypes(include=[np.number]).columns:
            df_lab[col] = df_lab[col].replace([np.inf, -np.inf], np.nan)
            if df_lab[col].isna().any():
                df_lab[col] = df_lab[col].fillna(df_lab[col].median())
        
        for col in df_lab.select_dtypes(include=['object', 'category']).columns:
            if df_lab[col].isna().any():
                df_lab[col] = df_lab[col].fillna(df_lab[col].mode().iloc[0] if len(df_lab[col].mode()) > 0 else 'Unknown')
        
        if df_lab.empty or len(df_lab.columns) == 0:
            skipped_classes[str(lab)] = {"reason": "資料清理後為空", "n_rows": 0}
            continue

        # 自動參數調整
        e = int(epochs_by_label.get(lab, epochs)) if isinstance(epochs_by_label, dict) else int(epochs)
        e = _auto_epochs_ctgan_local(n) if (use_auto_params and not (isinstance(epochs_by_label, dict) and lab in epochs_by_label)) else e
        bs = _auto_batch_ctgan_local(n)
        auto_dims = _auto_dims_ctgan_local(n)
        auto_disc_steps = _auto_discriminator_steps_ctgan_local(n)

        metadata_lab = Metadata.detect_from_dataframe(data=df_lab)

        # GPU 偵測
        use_gpu, gpu_info = detect_gpu()
        if lab == labels[0]:  # 只在第一個類別輸出
            log_gpu_status("CTGAN-grouped", use_gpu, gpu_info)

        synthesizer = None
        last_error = None
        actual_bs = bs
        
        # 獲取安全的 batch_size 嘗試列表
        batch_sizes_to_try = _get_batch_sizes_to_try_local(n, bs)
        
        for try_bs in batch_sizes_to_try:
            try:
                # 使用自動調整的維度 (如果啟用自動參數)
                gen_dims = auto_dims if use_auto_params else auto_dims  # auto_dims 已經正確解析
                disc_dims = auto_dims if use_auto_params else auto_dims
                emb_dim = auto_dims[0] if use_auto_params else auto_dims[0]
                disc_steps = auto_disc_steps if use_auto_params else int(discriminator_steps)
                
                # 如果未啟用自動參數，使用用戶指定的維度
                if not use_auto_params:
                    if isinstance(generator_dim, (list, tuple)):
                        gen_dims = tuple(generator_dim)
                    else:
                        gen_dims = (int(generator_dim), int(generator_dim))
                    if isinstance(discriminator_dim, (list, tuple)):
                        disc_dims = tuple(discriminator_dim)
                    else:
                        disc_dims = (int(discriminator_dim), int(discriminator_dim))
                    if isinstance(embedding_dim, (list, tuple)):
                        emb_dim = int(embedding_dim[0])
                    else:
                        emb_dim = int(embedding_dim)
                
                synthesizer = CTGANSynthesizer(
                    metadata=metadata_lab,
                    epochs=e,
                    batch_size=try_bs,
                    discriminator_steps=disc_steps,
                    generator_dim=gen_dims,
                    discriminator_dim=disc_dims,
                    embedding_dim=emb_dim,
                    generator_lr=float(generator_lr),
                    discriminator_lr=float(discriminator_lr),
                    enforce_min_max_values=True,
                    enforce_rounding=True,
                    verbose=False,
                    cuda=use_gpu  # 啟用 GPU
                )
                synthesizer.fit(df_lab)
                actual_bs = try_bs
                actual_disc_steps = disc_steps
                break
            except (AssertionError, ValueError) as err:
                last_error = err
                synthesizer = None
                continue
            except TypeError:
                try:
                    synthesizer = CTGANSynthesizer(
                        metadata=metadata_lab,
                        epochs=e,
                        batch_size=try_bs,
                        verbose=False,
                        cuda=use_gpu  # 啟用 GPU
                    )
                    synthesizer.fit(df_lab)
                    actual_bs = try_bs
                    actual_disc_steps = 1
                    break
                except Exception as err:
                    last_error = err
                    synthesizer = None
                    continue
        
        if synthesizer is None:
            error_details = f"""
CTGAN fit 失敗 (所有 batch_size 嘗試都失敗):
  - class: {lab}
  - n_rows: {n}
  - epochs: {e}
  - tried batch_sizes: {batch_sizes_to_try}
  - error: {str(last_error) if last_error else 'Unknown'}
"""
            raise RuntimeError(error_details)
        
        synth_models[lab] = synthesizer
        loss_history[str(lab)] = {
            "n_rows": n, 
            "epochs": e, 
            "batch_size": actual_bs,
            "discriminator_steps": actual_disc_steps,
            "generator_dim": list(gen_dims),
            "discriminator_dim": list(disc_dims),
            "embedding_dim": emb_dim,
            "paper_reference": "CTGAN NeurIPS 2019"
        }

    syn_parts = []
    for lab, synthesizer in synth_models.items():
        k = int(allocations.get(lab, 0))
        if k <= 0:
            continue
        syn_lab = synthesizer.sample(num_rows=k)
        syn_lab[label_column] = lab
        syn_parts.append(syn_lab)

    if syn_parts:
        synthetic_df = pd.concat(syn_parts, ignore_index=True)
    else:
        synthetic_df = pd.DataFrame(columns=df.columns)

    if original_labels_map:
        synthetic_df[label_column] = synthetic_df[label_column].map(
            lambda x: original_labels_map.get(str(x), x)
        )

    if skipped_classes:
        loss_history["_skipped_classes"] = skipped_classes
    
    if not synth_models:
        raise ValueError(f"所有類別都因樣本數不足而被跳過。跳過的類別: {skipped_classes}")

    return synthetic_df, synth_models, loss_history


# ============================================================
# 分群訓練 TVAE（對照論文 TVAE_pos.ipynb / TVAE_neg.ipynb）
# ============================================================


def train_grouped_tvae(
    df,
    label_column,
    metadata_dict,
    epochs,
    batch_size,
    compress_dims,
    decompress_dims,
    embedding_dim,
    num_to_generate,
    *,
    use_auto_params: bool = True,
    epochs_by_label: dict = None,
    l2scale: float = 1e-5,
    loss_factor: float = 2.0,
):
    """
    分群訓練 TVAE（泛化版：支援 2~N 類別）。
    
    對照論文 TVAE_pos.ipynb / TVAE_neg.ipynb：
      - epochs=5000
      - embedding_dim=64
      - compress_dims=(64, 64)
      - decompress_dims=(64, 64)
    
    每個 label 各訓練一個 TVAE（移除 label 欄位後訓練），
    生成時依照真實資料的類別比例分配生成數量。
    """
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import Metadata

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    df = df.copy()
    label_dtype = df[label_column].dtype
    original_labels_map = {}
    
    if pd.api.types.is_float_dtype(label_dtype):
        df[label_column] = df[label_column].astype(str)
        for orig_val in df[label_column].unique():
            try:
                original_labels_map[orig_val] = float(orig_val)
            except:
                original_labels_map[orig_val] = orig_val

    labels = list(pd.Series(df[label_column]).dropna().unique())
    if len(labels) < 2:
        raise ValueError(f"grouped 模式至少需要 2 個類別，目前僅 {len(labels)} 個")

    # 依比例分配生成數量
    vc = df[label_column].value_counts(dropna=False).to_dict()
    total = int(sum(vc.values()))
    allocations = {}
    remain = int(num_to_generate)
    for i, lab in enumerate(labels):
        n_lab = int(vc.get(lab, 0))
        if i < len(labels) - 1:
            k = int(round(num_to_generate * (n_lab / max(total, 1))))
            k = max(0, k)
            allocations[lab] = k
            remain -= k
        else:
            allocations[lab] = max(0, remain)

    synth_models = {}
    loss_history = {}

    # ============================================================
    # TVAE 自動參數調整 (VAE 訓練，收斂較慢)
    # ============================================================
    # SDV 官方預設: epochs=300, batch_size=500, dims=(128,128)
    # 論文驗證: TVAE 需要更多 epochs，小數據集用較小網路
    # ============================================================
    # 自動參數調整 (基於 NeurIPS 2019 論文和 SDV 官方最佳實踐)
    # ============================================================
    # 論文: "Modeling Tabular data using Conditional GAN" (包含 TVAE)
    # - Xu et al., NeurIPS 2019
    # 
    # SDV 官方預設: epochs=300, batch_size=500, dims=(128,128)
    # 
    # 【關鍵發現】TVAE 對以下參數敏感:
    # - l2scale: 1e-5 (正則化強度)
    # - loss_factor: 2 (重建損失權重)
    # - embedding_dim: 應與網路維度匹配
    # ============================================================
    
    # ------------------------------------------------------------
    # 自動參數調整：本函數內部呼叫處已使用 _auto_epochs_tvae(n) 等 nested name，
    # 故此處保留同名薄包裝函數，內部轉發至 module-level 實作。
    # ------------------------------------------------------------
    def _auto_epochs_tvae_local(n: int) -> int:
        return _auto_epochs_tvae(n, epochs, use_auto_params)

    def _auto_dims_tvae_local(n: int) -> tuple:
        return _auto_dims_tvae(n, compress_dims, use_auto_params)

    def _auto_batch_tvae_local(n: int) -> int:
        return _auto_batch_tvae(n, batch_size)

    def _get_batch_sizes_to_try_tvae_local(n: int, preferred_bs: int) -> list:
        return _get_batch_sizes_to_try_tvae(n, preferred_bs)

    skipped_classes = {}
    MIN_SAMPLES_FOR_TVAE = 10
    
    for lab in labels:
        df_lab = df[df[label_column] == lab].drop(columns=[label_column]).reset_index(drop=True)
        n = int(len(df_lab))
        if n <= 0:
            continue
        
        if n < MIN_SAMPLES_FOR_TVAE:
            skipped_classes[str(lab)] = {
                "reason": f"樣本數不足 (n={n}, 最少需要 {MIN_SAMPLES_FOR_TVAE})",
                "n_rows": n
            }
            continue
        
        # 資料清理
        df_lab = df_lab.dropna(axis=1, how='all')
        
        for col in df_lab.select_dtypes(include=[np.number]).columns:
            df_lab[col] = df_lab[col].replace([np.inf, -np.inf], np.nan)
            if df_lab[col].isna().any():
                df_lab[col] = df_lab[col].fillna(df_lab[col].median())
        
        for col in df_lab.select_dtypes(include=['object', 'category']).columns:
            if df_lab[col].isna().any():
                df_lab[col] = df_lab[col].fillna(df_lab[col].mode().iloc[0] if len(df_lab[col].mode()) > 0 else 'Unknown')
        
        if df_lab.empty or len(df_lab.columns) == 0:
            skipped_classes[str(lab)] = {"reason": "資料清理後為空", "n_rows": 0}
            continue

        # 自動參數調整
        e = int(epochs_by_label.get(lab, epochs)) if isinstance(epochs_by_label, dict) else int(epochs)
        e = _auto_epochs_tvae_local(n) if (use_auto_params and not (isinstance(epochs_by_label, dict) and lab in epochs_by_label)) else e
        bs = _auto_batch_tvae_local(n)
        auto_dims = _auto_dims_tvae_local(n)

        metadata_lab = Metadata.detect_from_dataframe(data=df_lab)

        # GPU 偵測
        use_gpu, gpu_info = detect_gpu()
        if lab == labels[0]:  # 只在第一個類別輸出
            log_gpu_status("TVAE-grouped", use_gpu, gpu_info)

        synthesizer = None
        last_error = None
        actual_bs = bs
        
        # 獲取安全的 batch_size 嘗試列表
        batch_sizes_to_try = _get_batch_sizes_to_try_tvae_local(n, bs)
        
        for try_bs in batch_sizes_to_try:
            try:
                # 使用自動調整的維度
                comp_dims = auto_dims
                decomp_dims = auto_dims
                emb_dim = auto_dims[0]
                
                # 如果未啟用自動參數，使用用戶指定的維度
                if not use_auto_params:
                    if isinstance(compress_dims, (list, tuple)):
                        comp_dims = tuple(compress_dims)
                    else:
                        comp_dims = (int(compress_dims), int(compress_dims))
                    if isinstance(decompress_dims, (list, tuple)):
                        decomp_dims = tuple(decompress_dims)
                    else:
                        decomp_dims = (int(decompress_dims), int(decompress_dims))
                    if isinstance(embedding_dim, (list, tuple)):
                        emb_dim = int(embedding_dim[0])
                    else:
                        emb_dim = int(embedding_dim)
                
                synthesizer = TVAESynthesizer(
                    metadata=metadata_lab,
                    epochs=e,
                    batch_size=try_bs,
                    compress_dims=comp_dims,
                    decompress_dims=decomp_dims,
                    embedding_dim=emb_dim,
                    l2scale=float(l2scale),
                    loss_factor=float(loss_factor),
                    enforce_min_max_values=True,
                    enforce_rounding=True,
                    verbose=False,
                    cuda=use_gpu  # 啟用 GPU
                )
                synthesizer.fit(df_lab)
                actual_bs = try_bs
                break
            except (AssertionError, ValueError) as err:
                last_error = err
                synthesizer = None
                continue
            except TypeError:
                # 某些版本可能不支援某些參數
                try:
                    synthesizer = TVAESynthesizer(
                        metadata=metadata_lab,
                        epochs=e,
                        batch_size=try_bs,
                        verbose=False,
                        cuda=use_gpu  # 啟用 GPU
                    )
                    synthesizer.fit(df_lab)
                    actual_bs = try_bs
                    break
                except Exception as err:
                    last_error = err
                    synthesizer = None
                    continue
        
        if synthesizer is None:
            error_details = f"""
TVAE fit 失敗 (所有 batch_size 嘗試都失敗):
  - class: {lab}
  - n_rows: {n}
  - epochs: {e}
  - compress_dims: {comp_dims}
  - decompress_dims: {decomp_dims}
  - tried batch_sizes: {batch_sizes_to_try}
  - error: {str(last_error) if last_error else 'Unknown'}
"""
            raise RuntimeError(error_details)
        
        synth_models[lab] = synthesizer
        loss_history[str(lab)] = {
            "n_rows": n, 
            "epochs": e, 
            "batch_size": actual_bs,
            "compress_dims": list(comp_dims),
            "decompress_dims": list(decomp_dims),
            "embedding_dim": emb_dim,
            "l2scale": float(l2scale),
            "loss_factor": float(loss_factor),
            "paper_reference": "TVAE NeurIPS 2019"
        }

    # 生成資料
    syn_parts = []
    for lab, synthesizer in synth_models.items():
        k = int(allocations.get(lab, 0))
        if k <= 0:
            continue
        syn_lab = synthesizer.sample(num_rows=k)
        syn_lab[label_column] = lab
        syn_parts.append(syn_lab)

    if syn_parts:
        synthetic_df = pd.concat(syn_parts, ignore_index=True)
    else:
        synthetic_df = pd.DataFrame(columns=df.columns)

    # 還原標籤類型
    if original_labels_map:
        synthetic_df[label_column] = synthetic_df[label_column].map(
            lambda x: original_labels_map.get(str(x), x)
        )

    if skipped_classes:
        loss_history["_skipped_classes"] = skipped_classes
    
    if not synth_models:
        raise ValueError(f"所有類別都因樣本數不足而被跳過。跳過的類別: {skipped_classes}")

    return synthetic_df, synth_models, loss_history


# ============================================================
# 混合訓練 CTGAN
# ============================================================


def train_mixed_ctgan(df, metadata, epochs, batch_size, discriminator_steps,
                      generator_dim, discriminator_dim, embedding_dim,
                      generator_lr, discriminator_lr, num_to_generate,
                      *, use_auto_params: bool = True):
    """混合訓練 CTGAN。

    與 grouped 模式行為對齊：
      - 支援 use_auto_params 自動參數調整（依資料量 n=len(df)）
      - 支援 batch_size 退避機制（fit 失敗時自動嘗試較小批次）
    """
    from sdv.single_table import CTGANSynthesizer

    n = int(len(df))

    # GPU 偵測
    use_gpu, gpu_info = detect_gpu()
    log_gpu_status("CTGAN-mixed", use_gpu, gpu_info)
    loss_history = {'device': 'cuda' if use_gpu else 'cpu'}

    # ------------------------------------------------------------
    # 自動參數調整（與 grouped 模式採用相同 helper 實作）
    # ------------------------------------------------------------
    if use_auto_params:
        e = _auto_epochs_ctgan(n, epochs, True)
        gen_dims = _auto_dims_ctgan(n, generator_dim, True)
        disc_dims = gen_dims  # 對齊 grouped：自動模式下 disc 與 gen 維度同步
        emb_dim = gen_dims[0]
        disc_steps = _auto_discriminator_steps_ctgan(n, discriminator_steps, True)
        bs = _auto_batch_ctgan(n, batch_size)
    else:
        e = int(epochs)
        if isinstance(generator_dim, (list, tuple)):
            gen_dims = tuple(int(d) for d in generator_dim)
        else:
            gen_dims = (int(generator_dim), int(generator_dim))
        if isinstance(discriminator_dim, (list, tuple)):
            disc_dims = tuple(int(d) for d in discriminator_dim)
        else:
            disc_dims = (int(discriminator_dim), int(discriminator_dim))
        if isinstance(embedding_dim, (list, tuple)):
            emb_dim = int(embedding_dim[0])
        else:
            emb_dim = int(embedding_dim)
        disc_steps = int(discriminator_steps)
        bs = int(batch_size)

    # ------------------------------------------------------------
    # batch_size 退避機制
    # ------------------------------------------------------------
    batch_sizes_to_try = _get_batch_sizes_to_try(n, bs)
    synthesizer = None
    last_error = None
    actual_bs = bs

    for try_bs in batch_sizes_to_try:
        try:
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=e,
                batch_size=try_bs,
                discriminator_steps=disc_steps,
                generator_dim=gen_dims,
                discriminator_dim=disc_dims,
                embedding_dim=emb_dim,
                generator_lr=float(generator_lr),
                discriminator_lr=float(discriminator_lr),
                verbose=False,
                cuda=use_gpu
            )
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    synthesizer.fit(df)
            actual_bs = try_bs
            print(f"[CTGAN-mixed] 訓練成功 (n={n}, batch_size={try_bs}, epochs={e})", file=sys.stderr)
            break
        except (AssertionError, ValueError, RuntimeError) as err:
            last_error = err
            print(f"[CTGAN-mixed][WARN] batch_size={try_bs} 失敗: {err}; 嘗試下一個", file=sys.stderr)
            synthesizer = None
            continue

    if synthesizer is None:
        raise RuntimeError(
            f"CTGAN-mixed 所有 batch_size 嘗試皆失敗 (最後錯誤: {last_error})"
        )

    loss_history['actual_batch_size'] = actual_bs
    loss_history['actual_epochs'] = e

    try:
        if hasattr(synthesizer, '_model') and hasattr(synthesizer._model, 'loss_values'):
            loss_values = synthesizer._model.loss_values
            loss_history['generator_loss'] = loss_values.get('Generator', [])
            loss_history['discriminator_loss'] = loss_values.get('Discriminator', [])
    except Exception:
        pass

    synthetic_df = synthesizer.sample(num_rows=num_to_generate)

    return synthetic_df, synthesizer, loss_history


# ============================================================
# 混合訓練 TVAE
# ============================================================


def train_mixed_tvae(df, metadata, epochs, batch_size, 
                     compress_dims, decompress_dims, embedding_dim,
                     num_to_generate, l2scale=1e-5, loss_factor=2.0,
                     *, use_auto_params: bool = True):
    """混合訓練 TVAE。

    與 grouped 模式行為對齊：
      - 支援 use_auto_params 自動參數調整（依資料量 n=len(df)）
      - 支援 batch_size 退避機制（fit 失敗時自動嘗試較小批次）
    """
    from sdv.single_table import TVAESynthesizer

    n = int(len(df))

    # GPU 偵測
    use_gpu, gpu_info = detect_gpu()
    log_gpu_status("TVAE-mixed", use_gpu, gpu_info)
    loss_history = {'device': 'cuda' if use_gpu else 'cpu'}

    # ------------------------------------------------------------
    # 自動參數調整
    # ------------------------------------------------------------
    if use_auto_params:
        e = _auto_epochs_tvae(n, epochs, True)
        comp_dims = _auto_dims_tvae(n, compress_dims, True)
        decomp_dims = comp_dims
        emb_dim = comp_dims[0]
        bs = _auto_batch_tvae(n, batch_size)
    else:
        e = int(epochs)
        if isinstance(compress_dims, (list, tuple)):
            comp_dims = tuple(int(d) for d in compress_dims)
        else:
            comp_dims = (int(compress_dims), int(compress_dims))
        if isinstance(decompress_dims, (list, tuple)):
            decomp_dims = tuple(int(d) for d in decompress_dims)
        else:
            decomp_dims = (int(decompress_dims), int(decompress_dims))
        if isinstance(embedding_dim, (list, tuple)):
            emb_dim = int(embedding_dim[0])
        else:
            emb_dim = int(embedding_dim)
        bs = int(batch_size)

    # ------------------------------------------------------------
    # batch_size 退避機制
    # ------------------------------------------------------------
    batch_sizes_to_try = _get_batch_sizes_to_try_tvae(n, bs)
    synthesizer = None
    last_error = None
    actual_bs = bs

    for try_bs in batch_sizes_to_try:
        try:
            synthesizer = TVAESynthesizer(
                metadata,
                epochs=e,
                batch_size=try_bs,
                compress_dims=comp_dims,
                decompress_dims=decomp_dims,
                embedding_dim=emb_dim,
                l2scale=float(l2scale),
                loss_factor=float(loss_factor),
                verbose=False,
                cuda=use_gpu
            )
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    synthesizer.fit(df)
            actual_bs = try_bs
            print(f"[TVAE-mixed] 訓練成功 (n={n}, batch_size={try_bs}, epochs={e})", file=sys.stderr)
            break
        except (AssertionError, ValueError, RuntimeError) as err:
            last_error = err
            print(f"[TVAE-mixed][WARN] batch_size={try_bs} 失敗: {err}; 嘗試下一個", file=sys.stderr)
            synthesizer = None
            continue

    if synthesizer is None:
        raise RuntimeError(
            f"TVAE-mixed 所有 batch_size 嘗試皆失敗 (最後錯誤: {last_error})"
        )

    loss_history['actual_batch_size'] = actual_bs
    loss_history['actual_epochs'] = e

    synthetic_df = synthesizer.sample(num_rows=num_to_generate)

    return synthetic_df, synthesizer, loss_history


# ============================================================
# TabDDPM 訓練函數（使用 synthcity 庫）
# ============================================================
# TabDDPM 論文: "TabDDPM: Modelling Tabular Data with Diffusion Models" (ICML 2023)
# 參考: https://github.com/yandex-research/tab-ddpm
# synthcity 實現: Plugins().get("ddpm")
# ============================================================


def train_grouped_tabddpm(
    df,
    label_column,
    metadata_dict,
    n_iter,
    batch_size,
    num_timesteps,
    num_to_generate,
    *,
    use_auto_params: bool = True,
    n_iter_by_label: dict = None,
):
    """
    分群訓練 TabDDPM（泛化版：支援 2~N 類別）。
    
    TabDDPM 使用擴散模型生成表格數據，相比 GAN/VAE 有更好的穩定性。
    
    參數:
        n_iter: 訓練迭代次數（類似 epochs）
        batch_size: 批次大小
        num_timesteps: 擴散步數（預設 1000）
    """
    try:
        # 抑制 synthcity/KeOps 的警告訊息
        import os
        import warnings
        os.environ['KEOPS_VERBOSE'] = '0'
        os.environ['USE_KEOPS'] = '0'
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # 抑制 loguru 的 CRITICAL 訊息
        import logging
        logging.getLogger('synthcity').setLevel(logging.ERROR)
        
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
    except ImportError:
        raise ImportError(
            "TabDDPM 需要安裝 synthcity 庫。請執行: pip install synthcity"
        )

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    df = df.copy()
    label_dtype = df[label_column].dtype
    original_labels_map = {}
    
    if pd.api.types.is_float_dtype(label_dtype):
        df[label_column] = df[label_column].astype(str)
        for orig_val in df[label_column].unique():
            try:
                original_labels_map[orig_val] = float(orig_val)
            except:
                original_labels_map[orig_val] = orig_val

    labels = list(pd.Series(df[label_column]).dropna().unique())
    if len(labels) < 2:
        raise ValueError(f"grouped 模式至少需要 2 個類別，目前僅 {len(labels)} 個")

    vc = df[label_column].value_counts(dropna=False).to_dict()
    total = int(sum(vc.values()))
    allocations = {}
    remain = int(num_to_generate)
    for i, lab in enumerate(labels):
        n_lab = int(vc.get(lab, 0))
        if i < len(labels) - 1:
            k = int(round(num_to_generate * (n_lab / max(total, 1))))
            k = max(0, k)
            allocations[lab] = k
            remain -= k
        else:
            allocations[lab] = max(0, remain)

    synth_models = {}
    loss_history = {}

    # ============================================================
    # TabDDPM GPU 記憶體和資料大小檢測
    # ============================================================
    n_features = len(df.columns)
    n_samples = len(df)
    
    # 估算 GPU 記憶體需求 (粗略估計)
    # TabDDPM 記憶體需求 ≈ batch_size × n_features × model_width × 4 bytes × 多層
    estimated_memory_gb = (n_samples * n_features * 4 * 8) / (1024**3)
    
    # 檢測 GPU 記憶體
    gpu_memory_gb = 0
    force_cpu = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # 如果資料太大或 GPU 記憶體不足，強制使用 CPU
            if n_features > 80 and n_samples > 3000:
                print(f"[TabDDPM] 警告: 大數據集 ({n_samples} 筆 × {n_features} 欄位)，可能需要較長時間", file=sys.stderr)
                if gpu_memory_gb < 10:  # 10GB 以下的 GPU
                    print(f"[TabDDPM] GPU 記憶體 ({gpu_memory_gb:.1f}GB) 可能不足，將使用較小的 batch_size", file=sys.stderr)
    except:
        pass
    
    loss_history['_data_info'] = {
        'n_samples': n_samples,
        'n_features': n_features,
        'gpu_memory_gb': round(gpu_memory_gb, 1),
        'estimated_memory_gb': round(estimated_memory_gb, 2)
    }

    # ============================================================
    # TabDDPM 自動參數調整（依據 ICML 2023 論文 Table 1）
    # ============================================================
    # 論文來源: Kotelnikov et al., "TabDDPM: Modelling Tabular Data 
    #           with Diffusion Models", ICML 2023
    # GitHub: https://github.com/yandex-research/tab-ddpm
    # 
    # 論文 Table 1 建議的超參數搜索空間:
    # - Learning rate: LogUniform[0.00001, 0.003]
    # - Batch size: Cat{256, 4096}
    # - Diffusion timesteps: Cat{100, 1000}
    # - Training iterations: Cat{5000, 10000, 20000}  <-- 關鍵！
    # - # MLP layers: Int{2, 4, 6, 8}
    # - MLP width of layers: Int{128, 256, 512, 1024}
    # - Dropout: 0.0
    # - Scheduler: cosine
    # ============================================================
    
    # ------------------------------------------------------------
    # 自動參數調整：本函數內部呼叫處已使用 _auto_n_iter_tabddpm(n) 等 nested name，
    # 故此處保留同名薄包裝函數，內部轉發至 module-level 實作。
    # ------------------------------------------------------------
    def _auto_n_iter_tabddpm_local(n: int) -> int:
        return _auto_n_iter_tabddpm(n, n_iter, use_auto_params)

    def _auto_batch_tabddpm_local(n: int) -> int:
        return _auto_batch_tabddpm(n, batch_size, n_features, use_auto_params)

    def _auto_timesteps_tabddpm_local(user_timesteps: int) -> int:
        return _auto_timesteps_tabddpm(user_timesteps, use_auto_params)

    # ============================================================
    # 手動模式偏離警告（只在 use_auto_params=False 時輸出一次）
    # ============================================================
    if not use_auto_params:
        user_n_iter = int(n_iter)
        user_batch = int(batch_size)
        user_timesteps_val = int(num_timesteps)
        grouped_warnings = []
        if user_n_iter < 5000:
            grouped_warnings.append(
                f"n_iter={user_n_iter} 低於論文建議下限 5000 (Table 1: 5000/10000/20000)"
            )
        if user_batch not in [256, 4096]:
            grouped_warnings.append(
                f"batch_size={user_batch} 不是論文建議值 (Table 1: 256 或 4096)"
            )
        if user_timesteps_val not in [100, 1000]:
            grouped_warnings.append(
                f"num_timesteps={user_timesteps_val} 不是論文建議值 (Table 1: 100 或 1000)"
            )
        for w in grouped_warnings:
            print(f"[TabDDPM][WARN] {w}", file=sys.stderr)
        if grouped_warnings:
            print(
                "[TabDDPM][WARN] 使用者關閉了自動參數 (use_auto_params=False)，後端尊重使用者輸入。"
                "若品質不佳請將偏離值調整為論文建議值。",
                file=sys.stderr
            )

    synthetic_parts = []

    for lab in labels:
        sub_df = df[df[label_column] == lab].copy()
        n_sub = len(sub_df)
        k_gen = int(allocations.get(lab, 0))

        if n_sub == 0 or k_gen == 0:
            continue

        # 決定該類別的迭代次數
        if n_iter_by_label and lab in n_iter_by_label:
            actual_n_iter = int(n_iter_by_label[lab])
        else:
            actual_n_iter = _auto_n_iter_tabddpm_local(n_sub)
        
        actual_batch = _auto_batch_tabddpm_local(n_sub)
        
        # 【論文依據】使用論文建議的 timesteps: 100 或 1000
        actual_timesteps = _auto_timesteps_tabddpm_local(int(num_timesteps))

        loss_history[str(lab)] = {
            'n_samples': n_sub,
            'n_iter': actual_n_iter,
            'batch_size': actual_batch,
            'num_timesteps': actual_timesteps,
            'paper_reference': 'TabDDPM ICML 2023, Table 1'
        }

        try:
            # 偵測 GPU 並清理記憶體
            import torch
            has_cuda = torch.cuda.is_available()
            
            if has_cuda:
                # 在每個類別訓練前清理 GPU 記憶體
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 只在第一個類別輸出 GPU 狀態和參數
            if lab == labels[0]:
                gpu_available, gpu_info = detect_gpu()
                log_gpu_status("TabDDPM-grouped", gpu_available, gpu_info)
                loss_history['device'] = "cuda" if has_cuda else "cpu"
                
                # 警告大數據集
                if n_features > 80:
                    print(f"[TabDDPM] 注意: 高維資料 ({n_features} 欄位)，已自動減少 batch_size", file=sys.stderr)
            
            # 輸出訓練參數（標註論文依據）
            print(f"[TabDDPM-ICML2023] 類別 {lab}: n_samples={n_sub}, n_iter={actual_n_iter}, batch={actual_batch}, timesteps={actual_timesteps}", file=sys.stderr)
            
            # ============================================================
            # 創建 TabDDPM 模型（只使用基本參數）
            # synthcity ddpm 可能只支援: n_iter, batch_size, num_timesteps
            # ============================================================
            # 設定 device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            plugin = Plugins().get(
                "ddpm",
                n_iter=actual_n_iter,           # 論文: 5000, 10000, 20000
                batch_size=actual_batch,         # 論文: 256, 4096
                num_timesteps=actual_timesteps,  # 論文: 100, 1000
                device=device,                   # 使用 GPU
            )
            
            # 訓練
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    with contextlib.redirect_stderr(devnull):
                        plugin.fit(sub_df)
            
            # 生成
            synthetic_sub = plugin.generate(count=k_gen).dataframe()
            
            # 確保 label 列正確
            if label_column not in synthetic_sub.columns:
                synthetic_sub[label_column] = lab
            
            synthetic_parts.append(synthetic_sub)
            synth_models[lab] = plugin
            
        except Exception as e:
            error_msg = str(e)
            print(f"[WARNING] TabDDPM 訓練類別 {lab} 失敗: {error_msg}", file=sys.stderr)
            
            # ============================================================
            # CUDA 錯誤處理: 清理記憶體並嘗試回退策略
            # ============================================================
            is_cuda_error = "CUDA" in error_msg or "cuda" in error_msg or "device" in error_msg.lower()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            
            # 嘗試使用更小的參數重試
            try:
                # 【重試策略 1】使用更小的 batch_size
                retry_n_iter = max(5000, actual_n_iter // 2)
                retry_batch = min(64, actual_batch // 4)  # 大幅減少 batch_size
                retry_timesteps = 100
                
                print(f"[TabDDPM] 重試類別 {lab}: n_iter={retry_n_iter}, batch={retry_batch}, timesteps={retry_timesteps}", file=sys.stderr)
                
                plugin = Plugins().get(
                    "ddpm",
                    n_iter=retry_n_iter,
                    batch_size=retry_batch,
                    num_timesteps=retry_timesteps,
                    device=device,  # 使用 GPU
                )
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        with contextlib.redirect_stderr(devnull):
                            plugin.fit(sub_df)
                synthetic_sub = plugin.generate(count=k_gen).dataframe()
                if label_column not in synthetic_sub.columns:
                    synthetic_sub[label_column] = lab
                synthetic_parts.append(synthetic_sub)
                synth_models[lab] = plugin
                print(f"[TabDDPM] 類別 {lab} 重試成功", file=sys.stderr)
                
            except Exception as e2:
                print(f"[ERROR] TabDDPM 重試失敗: {e2}", file=sys.stderr)
                
                # ============================================================
                # 【重試策略 2】如果是 CUDA 錯誤，嘗試強制使用 CPU
                # ============================================================
                if is_cuda_error:
                    try:
                        print(f"[TabDDPM] 嘗試使用 CPU 訓練類別 {lab}...", file=sys.stderr)
                        
                        import torch
                        # 清理 CUDA 狀態
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 暫時設置環境變數禁用 CUDA
                        import os
                        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        
                        try:
                            cpu_device = torch.device('cpu')
                            cpu_plugin = Plugins().get(
                                "ddpm",
                                n_iter=max(3000, retry_n_iter // 2),  # 更少的迭代
                                batch_size=32,  # 最小 batch
                                num_timesteps=100,
                                device=cpu_device,  # 強制使用 CPU
                            )
                            with open(os.devnull, 'w') as devnull:
                                with contextlib.redirect_stdout(devnull):
                                    with contextlib.redirect_stderr(devnull):
                                        cpu_plugin.fit(sub_df)
                            synthetic_sub = cpu_plugin.generate(count=k_gen).dataframe()
                            if label_column not in synthetic_sub.columns:
                                synthetic_sub[label_column] = lab
                            synthetic_parts.append(synthetic_sub)
                            synth_models[lab] = cpu_plugin
                            print(f"[TabDDPM] 類別 {lab} CPU 訓練成功", file=sys.stderr)
                        finally:
                            # 恢復 CUDA 設置
                            if old_cuda_visible is not None:
                                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
                            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                                del os.environ['CUDA_VISIBLE_DEVICES']
                                
                    except Exception as e3:
                        print(f"[ERROR] TabDDPM CPU 回退也失敗: {e3}", file=sys.stderr)
                        # 記錄失敗的類別，但繼續處理其他類別
                        loss_history[str(lab)]['error'] = str(e3)
                        continue
                else:
                    # 非 CUDA 錯誤，記錄並繼續
                    loss_history[str(lab)]['error'] = str(e2)
                    continue

    if not synthetic_parts:
        raise RuntimeError("TabDDPM 所有類別訓練均失敗")

    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)

    # 還原 float label
    if original_labels_map:
        synthetic_df[label_column] = synthetic_df[label_column].map(
            lambda x: original_labels_map.get(x, x)
        )

    return synthetic_df, synth_models, loss_history


def train_mixed_tabddpm(df, metadata, n_iter, batch_size, num_timesteps, num_to_generate,
                         use_auto_params=True):
    """混合訓練 TabDDPM（不分群）。

    與 grouped 模式行為對齊：
      - 支援 use_auto_params 自動參數調整（依資料量 n=len(df) 與特徵數）
      - 支援三層退避機制（GPU 較小批次 → CPU → 拋出錯誤）
    """
    try:
        import os
        import warnings
        os.environ['KEOPS_VERBOSE'] = '0'
        os.environ['USE_KEOPS'] = '0'
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        import logging
        logging.getLogger('synthcity').setLevel(logging.ERROR)

        from synthcity.plugins import Plugins
    except ImportError:
        raise ImportError(
            "TabDDPM 需要安裝 synthcity 庫。請執行: pip install synthcity"
        )

    import torch
    has_cuda = torch.cuda.is_available()

    n = int(len(df))
    n_features = int(df.shape[1])

    # GPU 狀態
    gpu_available, gpu_info = detect_gpu()
    log_gpu_status("TabDDPM-mixed", gpu_available, gpu_info)

    # ------------------------------------------------------------
    # 自動參數調整（與 grouped 模式採用相同 helper 實作）
    # ------------------------------------------------------------
    if use_auto_params:
        actual_n_iter = _auto_n_iter_tabddpm(n, n_iter, True)
        actual_batch = _auto_batch_tabddpm(n, batch_size, n_features, True)
        actual_timesteps = _auto_timesteps_tabddpm(int(num_timesteps), True)
    else:
        actual_n_iter = int(n_iter)
        actual_batch = int(batch_size)
        actual_timesteps = int(num_timesteps)

        warnings_list = []
        if actual_n_iter < 5000:
            warnings_list.append(
                f"n_iter={actual_n_iter} 低於論文建議下限 5000 (Table 1: 5000/10000/20000)"
            )
        if actual_batch not in [256, 4096]:
            warnings_list.append(
                f"batch_size={actual_batch} 不是論文建議值 (Table 1: 256 或 4096)"
            )
        if actual_timesteps not in [100, 1000]:
            warnings_list.append(
                f"num_timesteps={actual_timesteps} 不是論文建議值 (Table 1: 100 或 1000)"
            )
        for w in warnings_list:
            print(f"[TabDDPM][WARN] {w}", file=sys.stderr)
        if warnings_list:
            print(
                "[TabDDPM][WARN] 使用者關閉了自動參數 (use_auto_params=False)，後端尊重使用者輸入。"
                "若品質不佳請將偏離值調整為論文建議值。",
                file=sys.stderr
            )

    loss_history = {
        'n_iter': actual_n_iter,
        'batch_size': actual_batch,
        'num_timesteps': actual_timesteps,
        'device': "cuda" if has_cuda else "cpu",
        'paper_reference': 'TabDDPM ICML 2023, Table 1'
    }

    print(f"[TabDDPM-ICML2023] Mixed mode: n_iter={actual_n_iter}, batch={actual_batch}, timesteps={actual_timesteps}", file=sys.stderr)

    # ------------------------------------------------------------
    # 三層退避機制
    #   策略 1：原始參數 + GPU
    #   策略 2：較小參數 + GPU（n_iter/2 下限 5000、batch/4 下限 64、timesteps=100）
    #   策略 3：較小參數 + CPU
    # ------------------------------------------------------------
    plugin = None
    last_error = None
    synthetic_df = None

    # 清理 GPU 記憶體
    if has_cuda:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    # 策略 1：原始參數 + GPU
    try:
        device = torch.device('cuda' if has_cuda else 'cpu')
        plugin = Plugins().get(
            "ddpm",
            n_iter=actual_n_iter,
            batch_size=actual_batch,
            num_timesteps=actual_timesteps,
            device=device,
        )
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    plugin.fit(df)
        synthetic_df = plugin.generate(count=num_to_generate).dataframe()
        print(f"[TabDDPM-mixed] 訓練成功 (策略 1: 原始參數)", file=sys.stderr)
    except Exception as err1:
        last_error = err1
        is_cuda_error = ("CUDA" in str(err1)) or ("cuda" in str(err1)) or ("device" in str(err1).lower())
        print(f"[TabDDPM-mixed][WARN] 策略 1 失敗: {err1}", file=sys.stderr)
        plugin = None

        # 釋放 GPU 記憶體
        if has_cuda:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        # 策略 2：較小參數 + GPU
        retry_n_iter = max(5000, actual_n_iter // 2)
        retry_batch = min(64, max(16, actual_batch // 4))
        retry_timesteps = 100
        try:
            device = torch.device('cuda' if has_cuda else 'cpu')
            print(f"[TabDDPM-mixed] 重試策略 2: GPU + n_iter={retry_n_iter}, batch={retry_batch}, timesteps={retry_timesteps}", file=sys.stderr)
            plugin = Plugins().get(
                "ddpm",
                n_iter=retry_n_iter,
                batch_size=retry_batch,
                num_timesteps=retry_timesteps,
                device=device,
            )
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    with contextlib.redirect_stderr(devnull):
                        plugin.fit(df)
            synthetic_df = plugin.generate(count=num_to_generate).dataframe()
            actual_n_iter, actual_batch, actual_timesteps = retry_n_iter, retry_batch, retry_timesteps
            loss_history.update({
                'n_iter': actual_n_iter,
                'batch_size': actual_batch,
                'num_timesteps': actual_timesteps,
                'fallback': 'strategy_2_smaller_gpu',
            })
            print(f"[TabDDPM-mixed] 訓練成功 (策略 2: 較小參數 + GPU)", file=sys.stderr)
        except Exception as err2:
            last_error = err2
            print(f"[TabDDPM-mixed][WARN] 策略 2 失敗: {err2}", file=sys.stderr)
            plugin = None

            # 策略 3：較小參數 + CPU（僅在 CUDA 錯誤時）
            if is_cuda_error or has_cuda:
                try:
                    print(f"[TabDDPM-mixed] 重試策略 3: CPU + n_iter={retry_n_iter}, batch={retry_batch}, timesteps={retry_timesteps}", file=sys.stderr)
                    plugin = Plugins().get(
                        "ddpm",
                        n_iter=retry_n_iter,
                        batch_size=retry_batch,
                        num_timesteps=retry_timesteps,
                        device=torch.device('cpu'),
                    )
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            with contextlib.redirect_stderr(devnull):
                                plugin.fit(df)
                    synthetic_df = plugin.generate(count=num_to_generate).dataframe()
                    actual_n_iter, actual_batch, actual_timesteps = retry_n_iter, retry_batch, retry_timesteps
                    loss_history.update({
                        'n_iter': actual_n_iter,
                        'batch_size': actual_batch,
                        'num_timesteps': actual_timesteps,
                        'device': 'cpu',
                        'fallback': 'strategy_3_smaller_cpu',
                    })
                    print(f"[TabDDPM-mixed] 訓練成功 (策略 3: 較小參數 + CPU)", file=sys.stderr)
                except Exception as err3:
                    last_error = err3
                    print(f"[TabDDPM-mixed][ERROR] 策略 3 亦失敗: {err3}", file=sys.stderr)
                    plugin = None

    if synthetic_df is None or plugin is None:
        raise RuntimeError(
            f"TabDDPM-mixed 全部三層退避皆失敗 (最後錯誤: {last_error})"
        )

    return synthetic_df, plugin, loss_history


# ============================================================
# 欄位約束系統 (Column Constraints)
# ============================================================
# 支援使用者在生成合成資料時指定每個欄位的生成範圍或選項。
#
# 技術方案（混合策略）:
#   CTGAN / TVAE (SDV):
#     - Pre-filter: 訓練前過濾原始資料，讓模型只學習約束範圍內的分布
#     - enforce_min_max_values=True (SDV 預設) 自動 clip 到 pre-filtered 範圍
#     - Post-filter: 生成後再次過濾保證 100% 合規
#     來源: https://docs.sdv.dev/sdv/concepts/constraint-augmented-generation-cag
#
#   TabDDPM (synthcity):
#     - Pre-filter: 同上
#     - Post-filter: 同上（等效於 synthcity Constraints 的 reject sampling）
#     來源: https://synthcity.readthedocs.io/en/latest/
#
# 約束格式 (JSON):
#   {
#     "age":    {"type": "range",   "min": 18, "max": 65},
#     "gender": {"type": "options", "values": ["M", "F"]},
#     "label":  {"type": "fixed",   "value": 1},
#     "income": {"type": "min",     "value": 30000},
#     "risk":   {"type": "max",     "value": 0.8}
#   }
# ============================================================
