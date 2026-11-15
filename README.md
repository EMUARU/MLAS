# MLAS - 機器學習分析系統

網頁式的表格資料機器學習平台。使用者上傳 CSV 後可以做前處理、訓練模型、
堆疊學習、預測、看 XAI 解釋，也能生成合成資料並產出品質評估報告。
不用寫程式，全部在瀏覽器操作。

本專案以賴彥廷學長開發的 MLAS 為基礎進行擴充（碩士論文〈表格式資料機器學習分析系統之擴充：
合成資料生成與基礎模型整合〉，國立中山大學資訊工程學系，2026）。
版本紀錄中第一個 commit 為擴充前的原系統，之後為擴充內容，可直接比對差異。
TGANT 蒸餾沿用張正昕學長 (2025) 提出的演算法。

## 系統架構

- 前端: Vue.js SPA，開發時跑 8080
- 後端: Flask (app.py)，跑 5000。每個訓練/預測任務都是用 subprocess
  另起一個 python 腳本執行，任務進度寫在各自的 status.json，前端輪詢
- 正式站: Flask 直接 serve 前端 build 出來的靜態檔，所以只開一個 port
- 儲存: 依使用者分目錄，上傳檔在 upload/{user}/，訓練結果在 model/{user}/{timestamp}/
- upload/ 和 model/ 內的檔案會由 cleaner.py 背景執行緒定期清掉，別把重要東西只放在伺服器上

## 功能總覽

原系統的部分:

- 資料上傳: 本機上傳、Samba 分享、伺服器端複製三種來源，可預覽、刪列、刪檔
- 前處理: 規則式缺值處理（mean/median 填補、刪列等），前端圖形化設定規則
- 模型訓練: xgb、lgbm、rf、lr、tabnet、mlp、catboost、adaboost、svm 九種，
  各自一個 train_*.py，參數由前端傳入
- 堆疊學習 (stacking): train_stacking.py 負責整個流程，K-fold 產 OOF 特徵
  訓練 meta model，base model 再用全資料 retrain 後存檔。各 base model 的
  邏輯在對應的 stacking_*.py
- 評估: evaluate.py，輸出 accuracy / precision / recall / f1 / roc
- 預測: predict.py，載入訓練好的模型對新資料推論
- XAI: 各模型的解釋方法寫在自己的訓練腳本裡（SHAP/LIME 等，依模型而異）

擴充的部分:

- TabPFN: train_tabpfn.py，表格基礎模型，可選 2.5 或 v2 權重；
  堆疊學習也可選 TabPFN 作為 base model（stacking_tabpfn.py）
- 合成資料生成: run_synthetic.py 進入點，實作在 synthetic/ 套件。
  三種生成器 (CTGAN/TVAE/TabDDPM)、欄位約束、TGANT 蒸餾、
  品質評估報告。前端頁面為 SyntheticView.vue。細節看 backend/synthetic/README.md

## 環境需求

- Python 3.11。別的版本裝 tabpfn 或 synthcity 可能會失敗
- Node.js 20
- 建議使用 GPU (CUDA 12.x)。沒有 GPU 時 TabPFN 與 TabDDPM 會改用 CPU，
  但大資料集會非常慢，正式站請使用 GPU
- Docker 部署的主機要先裝 nvidia-container-toolkit

## 首次安裝（本機開發）

後端:

    cd backend
    python -m venv myenv
    myenv\Scripts\activate          (Linux 用 source myenv/bin/activate)

    # torch 要從 pytorch 官方 index 裝，直接 pip install torch 抓不到 cu121 版
    pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    pip install -r requirements.txt
    pip install synthcity==0.2.12

torch 釘 2.2.2、networkx 釘 2.8.8 的原因: synthcity 0.2.12 要求 torch<2.3
和 networkx<3.0。requirements.txt 已經照這個對齊，照順序裝就不會被降版。
synthcity 相依很多所以最後單獨裝。

前端:

    cd frontend
    npm install
    npm run serve

## 開發

start.bat 會開兩個終端機分別跑前後端，好了會自動開瀏覽器。
前提是 backend/myenv 已建好、frontend 已 npm install 過。

## 測試

### 擴充功能的自動化測試（pytest）

位於 backend/tests/，涵蓋本研究擴充的合成資料、欄位約束、TGANT 與 TabPFN:

    backend/tests/
      unit/                         單元測試（不需 GPU，數秒內跑完）
        test_constraints.py         約束解析（5 種單欄位約束、9 種條件運算子）、前置／後置過濾、回退機制
        test_auto_params.py         依樣本數自動調整參數的各分段邊界
        test_metrics.py             JS/KL 散度性質、ML Efficacy、Discriminator Measure、Fidelity
        test_tgant.py               TGANT 蒸餾：濾除標籤雜訊、收斂、最大代數、第二階段開關
        test_tabpfn.py              TabPFN 整合層：裝置選擇、ensemble 數量防呆、權重版本、錯誤處理
        test_utils.py / test_cli.py 資料前處理、輸出檔名、命令列參數錯誤
      integration/                  整合測試
        test_api_routes.py          Flask API：前端請求 → 組參數 → 啟動腳本 → 解析輸出 → 任務狀態
        test_synthetic_pipeline.py  實際執行合成資料完整流程（CTGAN/TVAE/TabDDPM、分群、約束、TGANT）
        test_tabpfn_pipeline.py     實際執行 TabPFN 訓練（需權重，設 MLAS_RUN_TABPFN=1 才執行）
        test_regression_models.py   原系統回歸：讀取 test_all.bat 的 81 個案例執行（需 GPU）

執行方式（在 backend 目錄）:

    pip install -r requirements-test.txt
    pytest -m "not slow"     # 單元測試 + API 整合測試
    pytest -m slow           # 實際訓練生成器的整合測試（較慢）
    pytest                   # 全部

需要 synthcity、tabpfn 的測試在未安裝時會自動跳過。
每次 push 會由 GitHub Actions（.github/workflows/tests.yml）自動執行。

測試只確認功能與流程正確，不評估生成品質；生成品質由論文中的消融實驗驗證。

### 原系統的回歸測試（test_all.bat，已整合進 pytest）

backend/test_all.bat 是 81 個回歸案例的清單。tests/integration/test_regression_models.py
會讀取它並以 pytest 執行（另外檢查腳本回報的錯誤），也可以在 Windows 上直接執行 .bat 只看 OK / FAIL。
案例內容:

- 訓練（train_test_split）: 10 種模型（九種傳統模型 + TabPFN）× 二元／多類別
- 訓練（k_fold）、評估、檔案預測: 各 10 個
- 手動輸入預測: 5 個
- 堆疊學習: 單一 base model、兩兩組合、三個以上組合、不同 meta model

整合時發現 5 個原系統問題（train_mlp.py 多類別、train_adaboost.py 多類別 k_fold、
CatBoost 多類別模型在評估與預測時的 SHAP 崩潰），已修正，詳見 backend/tests/README.md。

## 部署（正式站）

先設定環境變數 HF_TOKEN（Hugging Face token，下載 TabPFN 權重用），再執行 deploy.py。換機器只要改檔案開頭的 BASE_DIR、IMAGE_NAME、PORT_OUT 三個值，
腳本會停舊容器、重 build image、帶 --gpus all 起新容器，並 mount
upload/、model/、listWhite.txt。

要驗證能從零重建時把 NO_CACHE 改 True。

除錯:

    docker ps                              # 容器有沒有起來
    docker logs <IMAGE_NAME>_container     # 看 log
    docker restart <IMAGE_NAME>_container  # 重啟

注意: 重啟容器後，跑到一半的任務會變殭屍狀態（前端顯示執行中但其實沒了），
勾選後用刪除鈕清掉即可。

## 檔案結構

    backend/
      app.py                Flask 主程式，所有 API 路由
      run_synthetic.py      合成資料生成進入點
      synthetic/            合成資料套件（有自己的 README）
      train_*.py            各模型訓練腳本（含 train_tabpfn.py、train_stacking.py）
      stacking_*.py         堆疊學習的各 base model 邏輯
      evaluate.py           評估
      predict.py            預測
      preprocess.py         前處理
      cleaner.py            定期清 upload/ 和 model/ 的背景執行緒
      tool_*.py             開發工具（版本檢查、xlsx 合併等），非執行必需
      listWhite.txt         登入白名單（不進版控，參考 listWhite.example.txt 建立）
      tests/                pytest 單元測試與整合測試
      test_all.bat          原系統各模型的回歸測試
      upload/{user}/        使用者上傳檔（會被定期清）
      model/{user}/{ts}/    訓練結果（會被定期清）

## API 端點（backend/app.py）

除了首頁和 /download-synthetic 是 GET，其餘都是 POST，回 JSON。

檔案與資料:

    /login                     登入，比對 listWhite.txt
    /list-files                列出已上傳檔案
    /upload-local-file         本機上傳
    /upload-samba-file         從 Samba 分享抓檔
    /copy-local-file           複製伺服器端檔案過來
    /delete-files              刪檔
    /preview-tabular           預覽表格
    /get-field-number          取欄位數
    /delete-tabular-rows       刪指定列
    /preprocess                前處理
    /check-label-uniqueness    檢查標籤欄位取值
    /download                  下載檔案

訓練:

    /run-train-xgb  /run-train-lgbm  /run-train-rf  /run-train-lr
    /run-train-tabnet  /run-train-mlp  /run-train-catboost
    /run-train-adaboost  /run-train-svm     九種傳統模型
    /run-train-tabpfn                       TabPFN
    /run-train-stacking                     堆疊學習
    /run-synthetic                          合成資料生成（參數見 synthetic/README.md）

推論與評估:

    /run-predict               預測
    /run-evaluate              評估
    /download-report           下載評估報告
    /download-stacking-models  下載堆疊模型
    /download-synthetic/<user>/<task_id>/<filename>    下載合成結果 (GET)

任務管理:

    /list-status               列出所有任務狀態
    /get-status                單一任務狀態
    /terminate-task            終止執行中任務
    /delete-folders            刪任務資料夾

## 常見問題

- 跑合成腳本出現 CRITICAL ... No module named 'dgl':
  synthcity 的 goggle 外掛缺 dgl 自己停用的訊息，系統沒用到它，無視即可。

- 裝 synthcity 時 torch 被降版或解析失敗:
  照「首次安裝」的順序裝，torch 2.2.2、networkx 2.8.8、Python 3.11 缺一不可。

- TabDDPM 訓練 CUDA out of memory:
  腳本會先自動降低 n_iter、batch 與 timesteps 重試，仍是 CUDA 錯誤時改用 CPU；
  還是失敗就手動改小 batch_size，
  或先關掉其他吃 GPU 的程序。

- 容器起不來或看不到 GPU:
  主機要有 nvidia-container-toolkit，啟動要帶 --gpus all（deploy.py 有帶）。
  docker logs 看詳細原因。

- npm install 失敗:
  確認 Node 20。node_modules 壞掉就整個刪掉重裝。

- 任務卡住:
  前端終止鈕（/terminate-task）。容器重啟造成的殭屍任務用刪除鈕清。

## I18n

支援中英文。加新語言就在 frontend/src/locales/ 加一份 label json，
再加語言選項。

## 登入

白名單 backend/listWhite.txt，一行一個使用者名稱，直接加。
此檔不進版控，首次部署時複製 listWhite.example.txt 改名後填入。

## XAI

每個模型的 XAI 寫在各自的訓練腳本裡，因為每個模型吃的資料格式不同。
