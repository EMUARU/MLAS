# synthetic - 合成資料生成套件

MLAS 的合成資料功能都在這個套件裡。後端 app.py 用 subprocess 呼叫
run_synthetic.py，run_synthetic.py 轉呼叫這裡的 cli.main()。

## 模組

    utils.py         GPU 偵測、型別轉換、ML 特徵準備、輸出檔名、metadata 偵測
    auto_params.py   依樣本數自動決定訓練參數（CTGAN/TVAE/TabDDPM 各一套分段規則）
    metrics.py       品質評估: fidelity (KL/JS)、privacy (DCR)、
                     utility (TSTR/TRTR/TRTS)、discriminator measure
    constraints.py   欄位約束: 單欄位約束 + IF-THEN 條件規則，前置與後置雙階段過濾
    tgant.py         TGANT 蒸餾，雙向 recall 都 >= 門檻才收斂（演算法沿用張正昕 (2025)）
    generators.py    三種生成器的訓練，各有 grouped（依標籤分群）和 mixed（混合）兩種模式
    cli.py           main()，解析 argv、跑流程、輸出 JSON

依賴方向是單向的: utils 最底層，generators 用到 utils 和 auto_params，
metrics 和 tgant 用到 utils，cli 收整全部。修改時不要反向 import。

## 執行

    python run_synthetic.py <參數...>     # app.py 用這個
    python -m synthetic <參數...>         # 等價

## 參數

位置參數，1-11 必填:

     1  file_path       訓練資料路徑（CSV 或 Excel）
     2  label_column    標籤欄位名，none 表示無標籤（只能用 mixed 模式）
     3  generator_type  ctgan / tvae / tabddpm
     4  num_samples     生成筆數；也可寫成倍數，例如 2x 表示原資料的 2 倍。
                        TGANT 開啟時不用這個值，改由倍數和蒸餾流程決定
     5  epochs          訓練輪數，TabDDPM 對應 n_iter。自動參數開啟時會被蓋掉
     6  batch_size      批次大小。自動參數開啟時會被蓋掉
     7  disc_steps      CTGAN discriminator 每輪步數。TVAE/TabDDPM 不用但要佔位
     8  gen_dim         CTGAN generator 維度，TVAE 拿去當 compress_dims
     9  disc_dim        CTGAN discriminator 維度，TVAE 拿去當 decompress_dims
    10  embed_dim       嵌入維度
    11  task_dir        任務輸出目錄，結果和 status.json 都寫這裡

    12  train_mode           grouped（預設）/ mixed
    13  gen_lr               預設 0.0002
    14  disc_lr              預設 0.0002
    15  use_auto_params      true/false，預設 false。
                             這一位如果傳整數會走舊版介面 (pos_epochs/neg_epochs)
    16  enable_tgant         true 開 TGANT，預設 false
    17  generate_multiplier  每代生成倍數，預設 1
    18  tgant_threshold      收斂門檻，預設 0.95
    19  tgant_max_gen        最大代數，預設 10
    20  tgant_metric         少數類保護的指標選擇 min/pos，預設 min
    21  column_constraints   欄位約束 JSON，預設 none（不套用約束）

第 15 位傳整數時（舊版介面），16 為 neg_epochs，17–20 依序為 enable_tgant、
generate_multiplier、tgant_threshold、tgant_max_gen，tgant_metric 固定為 min；
第 21 位欄位約束不受影響。

## 輸出

- 合成資料 CSV 和品質評估報告 (JSON) 寫進 task_dir
- 進度寫 task_dir/status.json 給後端輪詢
- 程序結束時 stdout 印最終 JSON，status 是 success 或 error
