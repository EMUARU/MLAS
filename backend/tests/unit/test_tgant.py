"""
單元測試：synthetic.tgant（TGANT 蒸餾）

驗證演算法的行為性質：
- 標籤欄位不存在時拋出錯誤
- 蒸餾後的合成資料筆數不增加，且只保留原有欄位
- 混入「標籤錯誤的雜訊樣本」時，雜訊會被優先濾除
- 收斂條件、最大代數、第二階段開關
"""

import numpy as np
import pandas as pd
import pytest

from conftest import requires_xgboost

pytestmark = requires_xgboost


@pytest.fixture
def real_and_noisy_synth(binary_df):
    real = binary_df.drop(columns=["sex"])
    rng = np.random.default_rng(1)
    clean = real.sample(n=300, replace=True, random_state=1).reset_index(drop=True)
    clean[["age", "bmi"]] = clean[["age", "bmi"]] + rng.normal(0, 0.5, (300, 2))
    noise = clean.sample(n=60, random_state=2).copy()
    noise["label"] = 1 - noise["label"]          # 標籤翻轉的雜訊
    noise["_noise"] = 1
    clean["_noise"] = 0
    synth = pd.concat([clean, noise], ignore_index=True)
    return real, synth


def _distill(real, synth, **kw):
    from synthetic.tgant import tgant_distill_filter
    feats = [c for c in synth.columns if c != "_noise"]
    return tgant_distill_filter(real, synth[feats], "label", **kw)


def test_missing_label_raises(binary_df):
    from synthetic.tgant import tgant_distill_filter
    with pytest.raises(ValueError):
        tgant_distill_filter(binary_df, binary_df, "nope")


def test_output_shape_and_summary(real_and_noisy_synth):
    real, synth = real_and_noisy_synth
    final_synth, final_real, history, summary = _distill(real, synth, max_generations=3)
    assert list(final_synth.columns) == ["age", "bmi", "label"]
    assert len(final_synth) <= len(synth)
    assert len(final_real) <= len(real)
    assert 1 <= summary["total_generations"] <= 3
    assert summary["original_synthetic_rows"] == len(synth)
    assert summary["final_synthetic_rows"] == len(final_synth)
    assert set(history) == {"stage1", "stage2"}
    assert len(history["stage1"]) == summary["total_generations"]


def test_label_noise_is_filtered_out(real_and_noisy_synth):
    real, synth = real_and_noisy_synth
    final_synth, *_ = _distill(real, synth.drop(columns=[]), max_generations=3, enable_stage2=False)
    # 以特徵 + 標籤比對回原始列，計算殘留雜訊比例
    merged = final_synth.merge(synth.drop_duplicates(["age", "bmi", "label"]),
                               on=["age", "bmi", "label"], how="left")
    noise_ratio_before = synth["_noise"].mean()
    noise_ratio_after = merged["_noise"].fillna(0).mean()
    assert noise_ratio_after < noise_ratio_before / 2


def test_converges_on_clean_separable_data(binary_df):
    from synthetic.tgant import tgant_distill_filter
    real = binary_df.drop(columns=["sex"])
    _, _, history, summary = tgant_distill_filter(real, real.copy(), "label",
                                                  recall_threshold=0.9, max_generations=5)
    assert summary["converged"] is True
    assert history["stage1"][-1]["converged"] is True


def test_respects_max_generations(real_and_noisy_synth):
    real, synth = real_and_noisy_synth
    _, _, _, summary = _distill(real, synth, recall_threshold=1.01, max_generations=2)
    assert summary["converged"] is False
    assert summary["total_generations"] == 2


def test_stage2_can_be_disabled(real_and_noisy_synth):
    real, synth = real_and_noisy_synth
    _, _, history, summary = _distill(real, synth, max_generations=3, enable_stage2=False)
    assert history["stage2"] == []
    assert summary["stage2_enabled"] is False
    assert summary["used_generation"] == summary["stage1_used_generation"]


@pytest.mark.parametrize("metric", ["pos", "macro", "min"])
def test_multiclass_metrics(multiclass_df, metric):
    from synthetic.tgant import tgant_distill_filter
    _, _, _, summary = tgant_distill_filter(multiclass_df, multiclass_df.copy(), "label",
                                            metric=metric, max_generations=2)
    assert summary["metric"] == metric
    assert 0 <= summary["final_recall_tstr"] <= 1
