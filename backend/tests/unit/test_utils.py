"""
單元測試：synthetic.utils（資料前處理與輸出檔名）
"""

import numpy as np
import pandas as pd
import pytest

from synthetic.utils import (
    _make_core_filename,
    _make_synthetic_filename,
    convert_datetime_columns,
    prepare_ml_features,
)


def test_datetime_columns_become_unix_seconds():
    df = pd.DataFrame({"d": pd.to_datetime(["1970-01-02", "2000-01-01"]), "x": [1, 2]})
    out = convert_datetime_columns(df)
    assert out["d"].tolist() == [86400, 946684800]
    assert pd.api.types.is_datetime64_any_dtype(df["d"])  # 原資料不被修改


def test_prepare_ml_features_drops_label_and_cleans_values():
    df = pd.DataFrame({
        "num": [1.0, np.nan, np.inf, 3.0],
        "cat": ["a", "b", "a", None],
        "label": [0, 1, 0, 1],
    })
    X = prepare_ml_features(df, label_column="label")
    assert list(X.columns) == ["num", "cat"]
    # NaN 與 inf 以中位數補值（1 與 3 的中位數為 2）
    assert X["num"].tolist() == [1.0, 2.0, 2.0, 3.0]
    # 類別欄位轉為整數編碼
    assert pd.api.types.is_integer_dtype(X["cat"])


def test_prepare_ml_features_all_nan_column_filled_with_zero():
    X = prepare_ml_features(pd.DataFrame({"x": [np.nan, np.nan]}))
    assert X["x"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("num_samples, n, expected", [
    ("2x", 400, "synthetic_ctgan_2x.csv"),
    ("500", 500, "synthetic_ctgan_500.csv"),
])
def test_synthetic_filename(num_samples, n, expected):
    assert _make_synthetic_filename("upload/u/heart.csv", "CTGAN", num_samples, n) == expected


def test_filename_prefix_and_suffix_rules():
    assert _make_synthetic_filename("asthma_A40.csv", "tvae", "1x", 10) == "asthma_synthetic_tvae_1x.csv"
    assert _make_core_filename("data/asthma_A40.csv", "tvae") == "asthma_synthetic_tvae_core1_A40.csv"
    assert _make_core_filename("heart.csv", "ctgan") == "synthetic_ctgan_core1.csv"
