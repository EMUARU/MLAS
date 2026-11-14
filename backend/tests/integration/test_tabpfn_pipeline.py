"""
整合測試：TabPFN 完整訓練流程（train_tabpfn.py）

實際執行 train_tabpfn.py（含載入預訓練權重、訓練、評估、SHAP/LIME、輸出 metrics.json）。
需要安裝 tabpfn 並能取得權重（首次執行會從 Hugging Face 下載，需設定 HF_TOKEN），
因此預設跳過；要執行時設定環境變數 MLAS_RUN_TABPFN=1。
"""

import json
import os
import subprocess
import sys

import pytest

from conftest import requires_tabpfn

pytestmark = [
    pytest.mark.slow,
    requires_tabpfn,
    pytest.mark.skipif(os.environ.get("MLAS_RUN_TABPFN") != "1", reason="設定 MLAS_RUN_TABPFN=1 才執行"),
]


@pytest.fixture
def csv_path(tmp_path, binary_df):
    p = tmp_path / "heart.csv"
    binary_df.drop(columns=["sex"]).to_csv(p, index=False)
    return p


def run(backend_dir, *args):
    proc = subprocess.run([sys.executable, "train_tabpfn.py", *map(str, args), "--device", "cpu",
                           "--n_ensembles", "2"],
                          cwd=backend_dir, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc


@pytest.mark.parametrize("strategy, value", [("train_test_split", "0.8"), ("k_fold", "3")])
def test_train_and_evaluate(backend_dir, csv_path, tmp_path, strategy, value):
    task_dir = tmp_path / strategy
    run(backend_dir, csv_path, "label", strategy, value, "heart", task_dir)
    with open(task_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics.get("status", "success") == "success"
    if strategy == "train_test_split":
        assert (task_dir / "heart_tabpfn.pkl").exists()
        assert metrics["metrics"]["accuracy"] > 80
