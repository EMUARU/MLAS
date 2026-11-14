#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_synthetic.py —— 合成資料生成之命令列進入點（thin entry）

實際邏輯位於 synthetic/ 套件；本檔僅作為 subprocess 進入點，
供後端 app.py 以 `python run_synthetic.py <args>` 呼叫。
等價於 `python -m synthetic <args>`。
"""

from synthetic.cli import main

if __name__ == "__main__":
    main()
