"""
單元測試：synthetic.metrics（合成資料品質評估）

驗證指標的數學性質，而不是比對特定數值：
- 相同分布的 JS / KL 散度接近 0，不同分布明顯大於 0
- JS 對稱、介於 0 與 ln 2 之間
- ML Efficacy 與 Discriminator Measure 的輸出結構與合理範圍
"""

import numpy as np
import pandas as pd
import pytest

from conftest import requires_sdv, requires_xgboost
from synthetic.metrics import (
    compute_js_divergence,
    compute_kl_divergence,
    divergence_for_column,
)


class TestDivergence:
    def test_identical_distribution_is_zero(self):
        p = [10, 20, 30]
        assert compute_js_divergence(p, p) == pytest.approx(0, abs=1e-9)
        assert compute_kl_divergence(p, p) == pytest.approx(0, abs=1e-9)

    def test_js_is_symmetric_and_bounded(self):
        p, q = [1, 0, 0], [0, 0, 1]
        js_pq = compute_js_divergence(p, q)
        assert js_pq == pytest.approx(compute_js_divergence(q, p))
        assert 0 < js_pq <= np.log(2) + 1e-6

    def test_kl_is_non_negative_and_asymmetric(self):
        p, q = [8, 1, 1], [1, 1, 8]
        assert compute_kl_divergence(p, q) > 0
        assert compute_kl_divergence([9, 1], [5, 5]) != pytest.approx(compute_kl_divergence([5, 5], [9, 1]))

    def test_kl_base_conversion(self):
        p, q = [8, 2], [2, 8]
        assert compute_kl_divergence(p, q, base=2) == pytest.approx(compute_kl_divergence(p, q) / np.log(2))

    def test_numeric_column_similar_vs_shifted(self):
        rng = np.random.default_rng(0)
        real = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
        similar = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
        shifted = pd.DataFrame({"x": rng.normal(5, 1, 1000)})
        assert divergence_for_column(real, similar, "x") < 0.05
        assert divergence_for_column(real, shifted, "x") > 0.3

    def test_categorical_column_with_unseen_category(self):
        real = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
        synth = pd.DataFrame({"c": ["a"] * 50 + ["z"] * 50})
        assert divergence_for_column(real, synth, "c", method="kl") > 0
        assert divergence_for_column(real, real, "c") == pytest.approx(0, abs=1e-6)


@requires_xgboost
class TestUtilityMetrics:
    def test_ml_efficacy_on_real_copy_matches_baseline(self, binary_df):
        from synthetic.metrics import evaluate_ml_efficacy
        df = binary_df.drop(columns=["sex"])
        result = evaluate_ml_efficacy(df, df.copy(), "label")
        assert "error" not in result
        assert result["is_binary"] is True
        assert result["classes"] == ["0", "1"]
        # 可分資料：基準與 TSTR 都應有高正確率
        assert result["baseline"]["accuracy"] > 0.9
        assert result["tstr"]["accuracy"] > 0.9

    def test_ml_efficacy_missing_label(self, binary_df):
        from synthetic.metrics import evaluate_ml_efficacy
        assert "error" in evaluate_ml_efficacy(binary_df, binary_df, "nope")

    def test_discriminator_cannot_separate_same_distribution(self):
        from synthetic.metrics import evaluate_discriminator_measure
        rng = np.random.default_rng(3)
        # 真實與合成資料是同一分布的兩份獨立樣本
        # （注意：不能用完全相同的副本，否則每筆測試資料在訓練集中都有標籤相反的雙胞胎）
        real = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
        synth = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
        result = evaluate_discriminator_measure(real, synth, n_iterations=2)
        for model in ("random_forest", "xgboost"):
            # 無法區分時，判別正確率應接近 50%
            assert 0.35 <= result[model]["accuracy"] <= 0.65

    def test_discriminator_separates_obviously_fake_data(self, binary_df):
        from synthetic.metrics import evaluate_discriminator_measure
        df = binary_df.drop(columns=["sex"])
        fake = df.copy()
        fake["age"] = fake["age"] + 100
        result = evaluate_discriminator_measure(df, fake, "label", n_iterations=2)
        assert result["random_forest"]["accuracy"] > 0.95

    def test_discriminator_grouped_reports_per_class(self, binary_df):
        from synthetic.metrics import evaluate_discriminator_measure
        df = binary_df.drop(columns=["sex"])
        result = evaluate_discriminator_measure(df, df.copy(), "label", grouped=True, n_iterations=1)
        assert set(result["per_class"]) == {"0", "1"}


@requires_sdv
def test_fidelity_report_structure(binary_df):
    from synthetic.metrics import evaluate_fidelity
    from synthetic.utils import detect_metadata
    metadata, _ = detect_metadata(binary_df)
    result = evaluate_fidelity(binary_df, binary_df.copy(), metadata)
    assert result["js_divergence"] == pytest.approx(0, abs=1e-6)
    assert set(result["details"]["js_by_column"]) == set(binary_df.columns)
    assert result["single_column_similarity"] > 0.9
