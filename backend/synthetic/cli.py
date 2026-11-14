#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.cli

合成資料生成之命令列進入點（main）。由後端 app.py 以 subprocess 呼叫，
解析 argv 參數、分派至各生成器與 TGANT／品質評估流程，並將結果以 JSON 輸出至 stdout。
"""

# 在導入任何模組前，先設置環境變數來抑制警告
import os
os.environ['KEOPS_VERBOSE'] = '0'
os.environ['USE_KEOPS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
import json
import time
import warnings
import contextlib
import re
from pathlib import Path

warnings.filterwarnings('ignore')

import logging
logging.getLogger('synthcity').setLevel(logging.CRITICAL + 1)
logging.getLogger('keops').setLevel(logging.CRITICAL + 1)

import numpy as np
import pandas as pd

# --- 各模組匯入 ---
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
from .metrics import (
    compute_js_divergence, compute_kl_divergence, divergence_for_column,
    evaluate_fidelity, evaluate_privacy, evaluate_ml_efficacy,
    evaluate_discriminator_measure,
)
from .constraints import (
    parse_column_constraints, prefilter_data_by_constraints,
    postfilter_synthetic_by_constraints,
)
from .tgant import tgant_distill_filter
from .generators import (
    train_grouped_ctgan, train_grouped_tvae,
    train_mixed_ctgan, train_mixed_tvae,
    train_grouped_tabddpm, train_mixed_tabddpm,
)


def main():
    # ============================================================
    # 抑制第三方庫的 stdout 輸出（pyKeOps, synthcity 等）
    # 確保只有 JSON 輸出到 stdout
    # ============================================================
    import os
    import io
    import contextlib
    
    # 設置環境變數抑制 KeOps 輸出
    os.environ['KEOPS_VERBOSE'] = '0'
    os.environ['USE_KEOPS'] = '0'
    
    # 抑制 synthcity 的所有輸出
    import logging
    logging.getLogger('synthcity').setLevel(logging.CRITICAL)
    logging.getLogger('pykeops').setLevel(logging.CRITICAL)
    logging.getLogger('keopscore').setLevel(logging.CRITICAL)
    
    # 暫存原始 stdout
    original_stdout = sys.stdout
    
    # 捕獲初始化時的 stdout 輸出（防止污染 JSON）
    init_capture = io.StringIO()
    
    # 導入可能產生輸出的模組時重定向 stdout
    try:
        sys.stdout = init_capture
        # 預先導入會產生輸出的模組
        try:
            import warnings
            warnings.filterwarnings('ignore')
            from synthcity.plugins import Plugins
            _ = Plugins().list()  # 觸發初始化
        except:
            pass
    finally:
        sys.stdout = original_stdout
    
    # 將捕獲的初始化輸出寫入 stderr（僅用於調試）
    init_output = init_capture.getvalue()
    if init_output.strip():
        print(f"[init] {init_output.strip()}", file=sys.stderr)
    
    if len(sys.argv) < 12:
        print(json.dumps({
            "status": "error",
            "message": "參數不足。用法: python run_synthetic.py <file_path> <label_column> <generator_type> <num_samples> <epochs> <batch_size> <disc_steps> <gen_dim> <disc_dim> <embed_dim> <task_dir> [train_mode] [gen_lr] [disc_lr] [use_auto_params] [enable_tgant] [generate_multiplier] [tgant_threshold] [tgant_max_gen] [tgant_metric]"
        }))
        sys.exit(1)

    # 基本參數
    file_path = sys.argv[1]
    label_column = sys.argv[2]
    generator_type = sys.argv[3].lower()  # ctgan 或 tvae
    num_samples = sys.argv[4]
    epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    discriminator_steps = int(sys.argv[7])  # TVAE 不使用，但保持參數一致性
    generator_dim = int(sys.argv[8])  # TVAE 用作 compress_dims
    discriminator_dim = int(sys.argv[9])  # TVAE 用作 decompress_dims
    embedding_dim = int(sys.argv[10])
    task_dir = sys.argv[11]
    
    # 進階參數
    train_mode = sys.argv[12] if len(sys.argv) > 12 else "grouped"
    generator_lr = float(sys.argv[13]) if len(sys.argv) > 13 else 0.0002
    discriminator_lr = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0002

    # ------------------------------------------------------------
    # 進階參數預設值
    # 註：tgant_metric 預設統一為 "min"（兩個分支皆同），
    # 對應 grouped 模式下少數類別保護之預設策略。
    # ------------------------------------------------------------
    use_auto_params = False
    pos_epochs = epochs
    neg_epochs = epochs
    enable_tgant = False
    generate_multiplier = 1
    tgant_threshold = 0.95
    tgant_max_gen = 10
    tgant_metric = "min"

    if len(sys.argv) > 15:
        v15 = str(sys.argv[15]).lower()
        if v15 in ("true", "false"):
            # ----------------------------------------------------
            # 模式 A：sys.argv[15] = use_auto_params（true/false）
            #   位置: 15=use_auto_params, 16=enable_tgant, 17=multiplier,
            #         18=tgant_threshold, 19=tgant_max_gen, 20=tgant_metric
            # ----------------------------------------------------
            use_auto_params = (v15 == "true")
            enable_tgant = (str(sys.argv[16]).lower() == "true") if len(sys.argv) > 16 else False
            generate_multiplier = int(sys.argv[17]) if len(sys.argv) > 17 else 1
            tgant_threshold = float(sys.argv[18]) if len(sys.argv) > 18 else 0.95
            tgant_max_gen = int(sys.argv[19]) if len(sys.argv) > 19 else 10
            tgant_metric = str(sys.argv[20]).lower() if len(sys.argv) > 20 else "min"
            pos_epochs = epochs
            neg_epochs = epochs
        else:
            # ----------------------------------------------------
            # 模式 B：sys.argv[15] = pos_epochs（整數，舊版相容介面）
            #   位置: 15=pos_epochs, 16=neg_epochs, 17=enable_tgant,
            #         18=multiplier, 19=tgant_threshold, 20=tgant_max_gen
            #   註：此模式下 tgant_metric 沿用預設 "min"
            # ----------------------------------------------------
            pos_epochs = int(sys.argv[15])
            neg_epochs = int(sys.argv[16]) if len(sys.argv) > 16 else epochs
            enable_tgant = (str(sys.argv[17]).lower() == "true") if len(sys.argv) > 17 else False
            generate_multiplier = int(sys.argv[18]) if len(sys.argv) > 18 else 1
            tgant_threshold = float(sys.argv[19]) if len(sys.argv) > 19 else 0.95
            tgant_max_gen = int(sys.argv[20]) if len(sys.argv) > 20 else 10
            # tgant_metric 沿用預設 "min"

    # 第 21 個參數：欄位約束 JSON（可選）
    column_constraints_raw = sys.argv[21] if len(sys.argv) > 21 else "none"
    column_constraints = parse_column_constraints(column_constraints_raw)


    start_time = time.time()

    try:
        # 1. 載入資料
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print(json.dumps({"status": "error", "message": f"不支援的檔案格式: {file_path}"}))
            sys.exit(1)

        original_rows = len(df)
        original_columns = list(df.columns)

        # 2. 計算生成數量
        if num_samples.endswith('x'):
            multiplier = float(num_samples[:-1])
            num_to_generate = int(original_rows * multiplier)
        else:
            num_to_generate = int(num_samples)

        # 若啟用 TGANT，通常需要較大倍數
        if enable_tgant:
            desired_mult = generate_multiplier if generate_multiplier and generate_multiplier > 1 else None
            current_mult = multiplier if num_samples.endswith('x') else None
            if desired_mult is not None and (current_mult is None or current_mult < desired_mult):
                num_samples = f"{int(desired_mult)}x"
                num_to_generate = int(original_rows * desired_mult)

        # 3. 偵測 Metadata
        metadata, metadata_dict = detect_metadata(df)

        # 3.5 欄位約束 Pre-filter（訓練前過濾原始資料）
        # 在 grouped 模式下，傳入 label_column 讓 pre-filter 偵測類別流失並降級
        constraints_prefilter_report = {}
        if column_constraints:
            prefilter_label_col = label_column if (train_mode == "grouped" and label_column != 'none') else None
            df, constraints_prefilter_report = prefilter_data_by_constraints(
                df, column_constraints, label_column=prefilter_label_col
            )
            # 處理回退警告（資料太少 或 label 類別流失）
            if constraints_prefilter_report.get('_warning'):
                print(
                    f"[Constraints][WARN] Pre-filter fallback triggered: "
                    f"{constraints_prefilter_report['_warning']}",
                    file=sys.stderr
                )
                if constraints_prefilter_report.get('_label_fallback'):
                    lf = constraints_prefilter_report['_label_fallback']
                    print(
                        f"[Constraints][WARN] Label diversity check failed. "
                        f"Original labels: {lf['original_labels']}, "
                        f"after filter: {lf['filtered_labels']}, "
                        f"missing: {lf['missing_labels']}. "
                        f"Using original data; post-filter will still enforce constraints.",
                        file=sys.stderr
                    )
            elif constraints_prefilter_report.get('_summary', {}).get('rows_removed', 0) > 0:
                # 正常過濾情況，需要重新偵測 Metadata
                metadata, metadata_dict = detect_metadata(df)
                print(
                    f"[Constraints] Pre-filter: {constraints_prefilter_report['_summary']['original_rows']} → "
                    f"{constraints_prefilter_report['_summary']['filtered_rows']} rows",
                    file=sys.stderr
                )

        # 4. 訓練生成器
        loss_history = {}
        synthesizer = None
        synth_models = None
        
        if generator_type == 'ctgan':
            if train_mode == "grouped":
                if label_column == 'none' or label_column not in df.columns:
                    raise ValueError('train_mode="grouped" 需要有效的 label_column')

                synthetic_df, synth_models, loss_history = train_grouped_ctgan(
                    df, label_column, metadata_dict,
                    epochs, batch_size, discriminator_steps,
                    generator_dim, discriminator_dim, embedding_dim,
                    generator_lr, discriminator_lr,
                    num_to_generate,
                    use_auto_params=use_auto_params,
                    epochs_by_label={1: pos_epochs, 0: neg_epochs} if (len(df[label_column].dropna().unique())==2 and set(df[label_column].dropna().unique())=={0,1}) else None,
                )
            else:
                synthetic_df, synthesizer, loss_history = train_mixed_ctgan(
                    df, metadata, epochs, batch_size, discriminator_steps,
                    generator_dim, discriminator_dim, embedding_dim,
                    generator_lr, discriminator_lr, num_to_generate,
                    use_auto_params=use_auto_params
                )
                
        elif generator_type == 'tvae':
            # TVAE 使用 generator_dim 作為 compress_dims，discriminator_dim 作為 decompress_dims
            compress_dims = (generator_dim, generator_dim)
            decompress_dims = (discriminator_dim, discriminator_dim)
            
            if train_mode == "grouped":
                if label_column == 'none' or label_column not in df.columns:
                    raise ValueError('train_mode="grouped" 需要有效的 label_column')

                synthetic_df, synth_models, loss_history = train_grouped_tvae(
                    df, label_column, metadata_dict,
                    epochs, batch_size,
                    compress_dims, decompress_dims, embedding_dim,
                    num_to_generate,
                    use_auto_params=use_auto_params,
                    epochs_by_label={1: pos_epochs, 0: neg_epochs} if (len(df[label_column].dropna().unique())==2 and set(df[label_column].dropna().unique())=={0,1}) else None,
                )
            else:
                synthetic_df, synthesizer, loss_history = train_mixed_tvae(
                    df, metadata, epochs, batch_size,
                    compress_dims, decompress_dims, embedding_dim,
                    num_to_generate,
                    use_auto_params=use_auto_params
                )
        
        elif generator_type == 'tabddpm':
            # TabDDPM 使用擴散模型，參數映射：
            # - epochs -> n_iter (迭代次數)
            # - generator_dim -> num_timesteps (擴散步數)
            n_iter = epochs  # 對應 epochs 參數
            
            # num_timesteps: 擴散步數，影響生成品質和速度
            n_rows = len(df)
            if use_auto_params:
                # 自動參數模式：根據資料量調整 timesteps
                if n_rows < 100:
                    num_timesteps = 100
                elif n_rows < 500:
                    num_timesteps = 200
                elif n_rows < 1000:
                    num_timesteps = 300
                else:
                    num_timesteps = 500
            elif 50 < generator_dim <= 2000:
                # 手動模式：使用用戶指定的 timesteps
                num_timesteps = generator_dim
            else:
                # 無效值：使用預設
                num_timesteps = 500
            
            if train_mode == "grouped":
                if label_column == 'none' or label_column not in df.columns:
                    raise ValueError('train_mode="grouped" 需要有效的 label_column')

                synthetic_df, synth_models, loss_history = train_grouped_tabddpm(
                    df, label_column, metadata_dict,
                    n_iter, batch_size, num_timesteps,
                    num_to_generate,
                    use_auto_params=use_auto_params,
                    n_iter_by_label={1: pos_epochs, 0: neg_epochs} if (len(df[label_column].dropna().unique())==2 and set(df[label_column].dropna().unique())=={0,1}) else None,
                )
            else:
                synthetic_df, synthesizer, loss_history = train_mixed_tabddpm(
                    df, metadata, n_iter, batch_size, num_timesteps,
                    num_to_generate,
                    use_auto_params=use_auto_params
                )
        else:
            print(json.dumps({"status": "error", "message": f"不支援的生成器類型: {generator_type}（支援: ctgan, tvae, tabddpm）"}))
            sys.exit(1)


        # 4.5 欄位約束 Post-filter（生成後過濾合成資料）
        constraints_postfilter_report = {}
        if column_constraints:
            synthetic_df, constraints_postfilter_report = postfilter_synthetic_by_constraints(
                synthetic_df, column_constraints
            )
            if constraints_postfilter_report.get('_summary', {}).get('rows_removed', 0) > 0:
                print(
                    f"[Constraints] Post-filter: {constraints_postfilter_report['_summary']['original_rows']} → "
                    f"{constraints_postfilter_report['_summary']['final_rows']} rows",
                    file=sys.stderr
                )

        # 4.6 儲存未過濾合成資料；若啟用 TGANT，再蒸餾
        synthetic_df_unfiltered = synthetic_df.copy()
        
        # 確保欄位順序與原始資料一致
        original_cols = list(df.columns)
        synth_cols = [c for c in original_cols if c in synthetic_df_unfiltered.columns]
        extra_cols = [c for c in synthetic_df_unfiltered.columns if c not in original_cols]
        synthetic_df_unfiltered = synthetic_df_unfiltered[synth_cols + extra_cols]
        synthetic_df = synthetic_df_unfiltered.copy()

        os.makedirs(task_dir, exist_ok=True)

        unfiltered_filename = _make_synthetic_filename(file_path, generator_type, num_samples, num_to_generate)
        unfiltered_path = os.path.join(task_dir, unfiltered_filename)
        synthetic_df_unfiltered.to_csv(unfiltered_path, index=False, encoding='utf-8-sig')

        tgant_report = None
        synthetic_filename = unfiltered_filename
        synthetic_path = unfiltered_path

        if enable_tgant:
            if label_column == 'none' or label_column not in df.columns:
                raise ValueError('enable_tgant=true 需要有效的 label_column')

            core_df, core_real_df, history, summary = tgant_distill_filter(
                real_df=df,
                synthetic_df=synthetic_df_unfiltered,
                label_column=label_column,
                recall_threshold=tgant_threshold,
                test_size=0.2,
                max_generations=tgant_max_gen,
                random_state=42,
                metric=tgant_metric,
            )
            # 向下相容：舊版 history 為 list（僅第一階段紀錄），
            # 新版 history 為 dict {'stage1': [...], 'stage2': [...]}。
            # 若前端僅讀取 list，提供平坦化之第一階段歷程於 'history' 欄位。
            tgant_report = {
                "history": history.get('stage1', []) if isinstance(history, dict) else history,
                "stage2_history": history.get('stage2', []) if isinstance(history, dict) else [],
                "summary": summary
            }

            synthetic_df = core_df
            
            # 確保欄位順序與原始資料一致
            original_cols = list(df.columns)
            synth_cols = [c for c in original_cols if c in synthetic_df.columns]
            extra_cols = [c for c in synthetic_df.columns if c not in original_cols]
            synthetic_df = synthetic_df[synth_cols + extra_cols]
            
            synthetic_filename = _make_core_filename(file_path, generator_type)
            synthetic_path = os.path.join(task_dir, synthetic_filename)
            synthetic_df.to_csv(synthetic_path, index=False, encoding='utf-8-sig')

        # 5. 品質評估
        evaluation_results = {}
        grouped_eval = (train_mode == "grouped" and label_column != 'none')
        
        evaluation_results['fidelity'] = evaluate_fidelity(df, synthetic_df, metadata)
        
        evaluation_results['privacy'] = evaluate_privacy(
            df, synthetic_df, metadata_dict, 
            label_column if label_column != 'none' else None,
            grouped=grouped_eval
        )
        
        if label_column != 'none' and label_column in df.columns:
            evaluation_results['utility'] = {
                'ml_efficacy': evaluate_ml_efficacy(df, synthetic_df, label_column),
                'discriminator_measure': evaluate_discriminator_measure(
                    df, synthetic_df, label_column, grouped=grouped_eval
                )
            }
        else:
            evaluation_results['utility'] = {
                'ml_efficacy': {'error': 'No label column specified'},
                'discriminator_measure': evaluate_discriminator_measure(df, synthetic_df)
            }

        # 6. 儲存檔案
        model_dir = os.path.join(task_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        # 定義保存函數（區分 SDV 和 synthcity 的保存方式）
        def save_model(model, path, gen_type):
            """保存模型，根據生成器類型選擇保存方式"""
            if gen_type == 'tabddpm':
                # synthcity 方式：save() 返回 bytes
                buff = model.save()
                with open(path, 'wb') as f:
                    f.write(buff)
            else:
                # SDV 方式：save(path)
                model.save(path)

        if train_mode == "grouped" and synth_models:
            for lab, synth in synth_models.items():
                safe_key = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(lab))
                model_path = os.path.join(model_dir, f"{generator_type}_{safe_key}.pkl")
                save_model(synth, model_path, generator_type)

            labs = set(synth_models.keys())
            if labs == {0, 1}:
                save_model(synth_models[1], os.path.join(model_dir, f"{generator_type}_pos.pkl"), generator_type)
                save_model(synth_models[0], os.path.join(model_dir, f"{generator_type}_neg.pkl"), generator_type)
        elif synthesizer:
            model_path = os.path.join(model_dir, f"{generator_type}.pkl")
            save_model(synthesizer, model_path, generator_type)

        # 保存評估報告（包含 tgant 資訊）
        report_filename = "quality_report.json"
        report_path = os.path.join(task_dir, report_filename)
        
        # 組合完整報告
        full_report = evaluation_results.copy()
        if tgant_report:
            full_report['tgant'] = tgant_report
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)

        # 7. 計算訓練時間
        end_time = time.time()
        training_time = round(end_time - start_time, 2)

        # 8. 輸出結果
        success_message = "合成資料生成完成"
        if loss_history.get("_skipped_classes"):
            skipped_info = loss_history["_skipped_classes"]
            skipped_names = list(skipped_info.keys())
            success_message += f"（注意：{len(skipped_names)} 個類別因樣本數不足被跳過: {skipped_names}）"
        
        result = {
            "status": "success",
            "message": success_message,
            "generator_type": generator_type,
            "train_mode": train_mode,
            "original_rows": original_rows,
            "synthetic_rows": len(synthetic_df),
            "synthetic_rows_generated": num_to_generate,
            "columns": original_columns,
            "training_time_seconds": training_time,
            "outputs": {
                "synthetic_csv": synthetic_path,
                "synthetic_csv_unfiltered": unfiltered_path if enable_tgant else synthetic_path,
                "quality_report": report_path,
                "model_dir": model_dir,
            },
            "parameters": {
                "epochs": epochs,
                "pos_epochs": pos_epochs if train_mode == "grouped" else None,
                "neg_epochs": neg_epochs if train_mode == "grouped" else None,
                "batch_size": batch_size,
                "discriminator_steps": discriminator_steps if generator_type == "ctgan" else None,
                "generator_dim": generator_dim,
                "discriminator_dim": discriminator_dim,
                "embedding_dim": embedding_dim,
                "generator_lr": generator_lr if generator_type == "ctgan" else None,
                "discriminator_lr": discriminator_lr if generator_type == "ctgan" else None,
                "compress_dims": [generator_dim, generator_dim] if generator_type == "tvae" else None,
                "decompress_dims": [discriminator_dim, discriminator_dim] if generator_type == "tvae" else None,
                "label_column": label_column,
                "enable_tgant": enable_tgant,
                "generate_multiplier": generate_multiplier,
                "column_constraints": column_constraints if column_constraints else None
            },
            "evaluation": evaluation_results,
            "synthetic_file": synthetic_filename,
            "synthetic_file_unfiltered": unfiltered_filename if enable_tgant else synthetic_filename,
            "tgant": tgant_report,
            "training_details": loss_history,
            "synthetic_preview": synthetic_df.head(10).to_dict(orient='records'),
            "column_constraints": {
                "enabled": bool(column_constraints),
                "rules": column_constraints if column_constraints else None,
                "prefilter_report": constraints_prefilter_report if constraints_prefilter_report else None,
                "postfilter_report": constraints_postfilter_report if constraints_postfilter_report else None,
            } if column_constraints else None
        }

        # 清理 NaN/Infinity/Timestamp 值（JSON 不支援這些值）
        def clean_for_json(obj):
            """遞迴清理字典/列表中的 NaN、Infinity 和不可序列化的值"""
            import math
            import numpy as np
            import pandas as pd
            from datetime import datetime, date
            
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (float, np.floating)):
                if math.isnan(float(obj)) or math.isinf(float(obj)):
                    return None  # 將 NaN/Infinity 轉換為 null
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return clean_for_json(obj.tolist())
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()  # 轉換為 ISO 格式字串
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, pd.Timedelta):
                return str(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        result = clean_for_json(result)
        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        import traceback
        print(json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
