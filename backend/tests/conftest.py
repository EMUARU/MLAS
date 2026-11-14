"""
共用 fixture 與測試環境設定。

- 將 backend/ 與 tests/ 加入 sys.path，讓測試能直接 import 後端腳本與 synthetic 套件。
- 提供小型、可重現（固定亂數種子）的測試資料。
- 依可用套件自動跳過需要重型相依（SDV、synthcity、TabPFN）的測試。
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
for p in (BACKEND_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def has_module(name):
    return importlib.util.find_spec(name) is not None


requires_sdv = pytest.mark.skipif(not has_module("sdv"), reason="需要 sdv")
requires_xgboost = pytest.mark.skipif(not has_module("xgboost"), reason="需要 xgboost")
requires_synthcity = pytest.mark.skipif(not has_module("synthcity"), reason="需要 synthcity")
requires_tabpfn = pytest.mark.skipif(not has_module("tabpfn"), reason="需要 tabpfn")
requires_flask = pytest.mark.skipif(
    not (has_module("flask") and has_module("flask_cors") and has_module("docx")),
    reason="需要 flask、flask_cors、python-docx",
)


@pytest.fixture
def backend_dir():
    return BACKEND_DIR


@pytest.fixture
def binary_df():
    """200 筆二元分類資料，兩類可分，含數值與類別欄位。"""
    rng = np.random.default_rng(42)
    n = 200
    label = np.array([0] * 120 + [1] * 80)
    age = np.where(label == 1, rng.normal(60, 5, n), rng.normal(35, 5, n)).round().astype(int)
    bmi = np.where(label == 1, rng.normal(32, 2, n), rng.normal(22, 2, n)).round(1)
    sex = rng.choice(["F", "M"], size=n)
    df = pd.DataFrame({"age": age, "bmi": bmi, "sex": sex, "label": label})
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def multiclass_df():
    """300 筆三類別資料（類別 0/1/2），各類別特徵中心不同。"""
    rng = np.random.default_rng(7)
    rows = []
    for lab, center in [(0, 0.0), (1, 5.0), (2, 10.0)]:
        for _ in range(100):
            rows.append({
                "f1": rng.normal(center, 1.0),
                "f2": rng.normal(center * 2, 1.0),
                "label": lab,
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def constraint_df():
    """欄位約束測試用的小型資料，含 NaN。"""
    return pd.DataFrame({
        "age": [10, 25, 40, 70, 130, np.nan],
        "sex": ["F", "M", "F", "M", "X", "F"],
        "smoker": [0, 1, 0, 1, 1, 0],
        "label": [0, 1, 0, 1, 1, 0],
    })
