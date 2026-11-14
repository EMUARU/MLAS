# MLAS 擴充功能測試說明

本資料夾是針對本研究擴充功能（合成資料生成、欄位約束、TGANT 蒸餾、TabPFN 整合）的
自動化測試，使用 pytest。原系統各模型的回歸測試另見 backend/test_all.bat。

## 測試總數

| 類型 | 數量 | 執行時間（開發環境實測） |
| --- | --- | --- |
| 單元測試 | 197 | 約 20 秒（與快速整合測試合計） |
| 整合測試 | 25 | 快速 14 個；需實際訓練的 11 個約 5–6 分鐘 |
| 合計 | 222 | 2026-11-15 於開發環境（Windows、Python 3.11）全部通過 |

另有原系統回歸測試 test_all.bat：81 個案例。

數量說明：pytest 以「參數化」把每個邊界值、每種運算子各算一個案例，
例如自動參數的 12 個分段邊界由同一個測試函式產生 12 個案例。

## 各檔案內容

### 單元測試（unit/，197 個）

| 檔案 | 數量 | 測什麼 |
| --- | --- | --- |
| test_constraints.py | 72 | 約束 JSON 解析（5 種單欄位約束、9 種條件運算子、無效規則）；when/then 判斷；前置過濾（含資料過少、類別消失時改用原始資料）；後置過濾（數值裁切、類別與條件規則刪列、結果 100% 合規） |
| test_auto_params.py | 74 | CTGAN/TVAE/TabDDPM 依樣本數自動調整 epochs、網路維度、batch size 的每個分段邊界；關閉自動參數時回傳原值；batch 為 10 的倍數且不超過資料量 |
| test_tabpfn.py | 21 | TabPFN 整合層：裝置選擇、ensemble 數量防呆（0、負數、非數字、上限 32）、權重版本（2.5 / v2）、模型存檔、門檻掃描曲線、錯誤處理 |
| test_metrics.py | 12 | JS/KL 散度的數學性質；ML Efficacy；Discriminator Measure（同分布約 50%、明顯假資料 > 95%）；Fidelity 報告結構 |
| test_tgant.py | 9 | 標籤錯誤時報錯；輸出筆數與摘要一致；混入標籤翻轉雜訊後雜訊比例至少減半；收斂、最大代數、第二階段開關；多類別 recall 計算 |
| test_utils.py | 6 | 日期轉換、缺值與無限值補值、類別編碼、輸出檔名 |
| test_cli.py | 3 | 參數不足、檔案格式不支援、生成器不支援時回傳錯誤 |

### 整合測試（integration/，25 個）

| 檔案 | 數量 | 測什麼 |
| --- | --- | --- |
| test_api_routes.py | 14 | Flask API：/run-synthetic 的 21 個參數順序與預設值、約束 JSON、從套件雜訊中找出結果 JSON、失敗與終止處理、status.json 更新；/run-train-tabpfn 的選用參數；/download-synthetic |
| test_synthetic_pipeline.py | 9 | 實際執行 run_synthetic.py：CTGAN/TVAE 混合模式、分群模式、倍數寫法、欄位約束 100% 合規、TGANT 蒸餾、無標籤欄位、錯誤處理、TabDDPM |
| test_tabpfn_pipeline.py | 2 | 實際執行 train_tabpfn.py（train_test_split、k_fold），需下載權重 |

## 測試規範

1. 單元測試不需要 GPU，數秒內完成。
2. 測「行為與性質」，不比對生成模型的精確數值（例如約束一定被遵守、雜訊比例下降）。
3. 固定亂數種子，失敗可重現。
4. 每個分段邊界、空值、類別消失、錯誤輸入都要測。
5. 外部相依（TabPFN 權重、訓練子程序）在單元測試中以替身取代，只測 MLAS 自己的邏輯。
6. 需要實際訓練的測試標記為 slow；缺少 synthcity、tabpfn 時自動跳過。

測試只確認功能與流程正確，不評估生成品質；生成品質由論文的消融實驗驗證。

## 執行方式（在 backend 目錄，PowerShell）

    pip install -r requirements-test.txt      # 已有完整開發環境則只需 pip install pytest
    pytest -m "not slow"                       # 單元 + API 整合測試
    pytest -m slow                             # 需實際訓練的整合測試
    $env:MLAS_RUN_TABPFN="1"; pytest           # 全部，含 TabPFN 完整訓練

## 檔案結構

    backend/
      pytest.ini              pytest 設定（測試路徑、slow 標記）
      requirements-test.txt   執行測試所需的最小相依
      tests/
        README.md             本說明
        conftest.py           共用測試資料（fixture）與自動跳過規則
        fakes.py              替身：假的 TabPFN
        unit/                 單元測試
        integration/          整合測試
