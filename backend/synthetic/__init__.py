#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic —— 合成資料生成套件

各模組職責如下：

    utils        工具（GPU／檔名／前處理／metadata）
    auto_params  依樣本規模之自動參數機制
    metrics      品質評估（Fidelity／Privacy／Utility／Discriminator）
    constraints  欄位約束（解析＋前置／後置過濾）
    tgant        TGANT 數據蒸餾
    generators   CTGAN／TVAE／TabDDPM 之 grouped／mixed 訓練
    cli          命令列進入點 main()

命令列用法（由 app.py 以 subprocess 呼叫）：
    python -m synthetic <file_path> <label_column> <generator_type> ...
或經相容轉接檔：
    python run_synthetic.py <file_path> ...
"""

from .cli import main

__all__ = ["main"]
