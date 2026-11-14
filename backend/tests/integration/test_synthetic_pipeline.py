"""
整合測試：合成資料完整流程（run_synthetic.py → synthetic 套件）

以子程序實際執行 run_synthetic.py，用小資料、少輪數跑完
「讀檔 → 前置過濾 → 訓練生成器 → 後置過濾 →（TGANT 蒸餾）→ 品質評估 → 輸出檔案」，
驗證輸出 JSON、合成資料 CSV、品質報告與模型檔，以及合成資料 100% 符合欄位約束。

這些測試只確認流程正確，不評估生成品質（品質由論文中的消融實驗驗證）。
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from conftest import requires_sdv, requires_synthcity

pytestmark = [pytest.mark.slow, requires_sdv]

CONSTRAINTS = {
    "columns": {"age": {"type": "range", "min": 20, "max": 75},
                "sex": {"type": "options", "values": ["F", "M"]}},
    "conditionals": [{"when": {"column": "label", "operator": "==", "value": 1},
                      "then": {"column": "bmi", "type": "min", "value": 25}}],
}


def run_synthetic(backend_dir, csv, out_dir, generator="ctgan", mode="mixed", epochs="5",
                  num_samples="300", tgant=False, constraints=None, label="label"):
    args = [
        sys.executable, "run_synthetic.py",
        str(csv), label, generator, num_samples, epochs, "50", "1", "32", "32", "16", str(out_dir),
        mode, "0.0002", "0.0002", "false", "true" if tgant else "false", "2", "0.9", "3", "min",
        json.dumps(constraints) if constraints else "none",
    ]
    proc = subprocess.run(args, cwd=backend_dir, capture_output=True, text=True, timeout=900)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip().startswith("{")]
    assert proc.returncode == 0 and lines, f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1])


def assert_outputs(result, out_dir, original_columns):
    assert result["status"] == "success", result
    synth = pd.read_csv(result["outputs"]["synthetic_csv"])
    assert len(synth) == result["synthetic_rows"] > 0
    assert list(synth.columns) == original_columns
    with open(result["outputs"]["quality_report"], encoding="utf-8") as f:
        report = json.load(f)
    assert {"fidelity", "privacy", "utility"} <= set(report)
    assert "js_divergence" in report["fidelity"]
    assert os.listdir(result["outputs"]["model_dir"])
    return synth, report


@pytest.fixture
def csv_path(tmp_path, binary_df):
    p = tmp_path / "heart.csv"
    binary_df.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("generator", ["ctgan", "tvae"])
def test_mixed_mode(backend_dir, csv_path, tmp_path, binary_df, generator):
    out = tmp_path / f"out_{generator}"
    result = run_synthetic(backend_dir, csv_path, out, generator=generator)
    synth, report = assert_outputs(result, out, list(binary_df.columns))
    assert result["synthetic_rows_generated"] == 300
    assert "ml_efficacy" in report["utility"]


def test_grouped_mode_keeps_every_class(backend_dir, csv_path, tmp_path, binary_df):
    out = tmp_path / "out_grouped"
    result = run_synthetic(backend_dir, csv_path, out, mode="grouped")
    synth, _ = assert_outputs(result, out, list(binary_df.columns))
    assert set(synth["label"]) == {0, 1}
    assert {"ctgan_pos.pkl", "ctgan_neg.pkl"} <= set(os.listdir(result["outputs"]["model_dir"]))


def test_multiplier_sample_count(backend_dir, csv_path, tmp_path, binary_df):
    result = run_synthetic(backend_dir, csv_path, tmp_path / "out_mult", num_samples="2x")
    assert result["synthetic_rows_generated"] == 2 * len(binary_df)
    assert result["outputs"]["synthetic_csv"].endswith("synthetic_ctgan_2x.csv")


def test_constraints_are_fully_satisfied(backend_dir, csv_path, tmp_path, binary_df):
    out = tmp_path / "out_cons"
    result = run_synthetic(backend_dir, csv_path, out, constraints=CONSTRAINTS)
    synth, _ = assert_outputs(result, out, list(binary_df.columns))
    assert synth["age"].between(20, 75).all()
    assert synth["sex"].isin(["F", "M"]).all()
    assert (synth.loc[synth["label"] == 1, "bmi"] >= 25).all()


def test_tgant_distillation(backend_dir, csv_path, tmp_path, binary_df):
    out = tmp_path / "out_tgant"
    result = run_synthetic(backend_dir, csv_path, out, mode="grouped", tgant=True)
    synth, report = assert_outputs(result, out, list(binary_df.columns))
    unfiltered = pd.read_csv(result["outputs"]["synthetic_csv_unfiltered"])
    assert len(synth) <= len(unfiltered)
    summary = report["tgant"]["summary"]
    assert 1 <= summary["total_generations"] <= 3
    assert result["outputs"]["synthetic_csv"].endswith("synthetic_ctgan_core1.csv")


def test_without_label_column(backend_dir, csv_path, tmp_path, binary_df):
    out = tmp_path / "out_nolabel"
    result = run_synthetic(backend_dir, csv_path, out, label="none")
    _, report = assert_outputs(result, out, list(binary_df.columns))
    assert "error" in report["utility"]["ml_efficacy"]


def test_grouped_without_label_fails_cleanly(backend_dir, csv_path, tmp_path):
    args = [sys.executable, "run_synthetic.py", str(csv_path), "none", "ctgan", "50", "1", "50",
            "1", "32", "32", "16", str(tmp_path / "x"), "grouped"]
    proc = subprocess.run(args, cwd=backend_dir, capture_output=True, text=True, timeout=300)
    out = json.loads([l for l in proc.stdout.splitlines() if l.startswith("{")][-1])
    assert out["status"] == "error"


@requires_synthcity
def test_tabddpm_mixed(backend_dir, csv_path, tmp_path, binary_df):
    out = tmp_path / "out_ddpm"
    result = run_synthetic(backend_dir, csv_path, out, generator="tabddpm", epochs="50")
    assert_outputs(result, out, list(binary_df.columns))
