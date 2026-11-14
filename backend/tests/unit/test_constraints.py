"""
單元測試：synthetic.constraints（欄位約束系統）

涵蓋：
- 約束 JSON 解析（新舊格式、5 種單欄位約束、9 種條件運算子、無效輸入）
- when / then 子句判斷
- 前置過濾（訓練前）：各約束類型、NaN 保留、資料過少回退、類別流失回退
- 後置過濾（生成後）：數值裁切、類別過濾、條件規則過濾
"""

import json

import numpy as np
import pandas as pd
import pytest

from synthetic.constraints import (
    _evaluate_then_constraint,
    _evaluate_when_clause,
    parse_column_constraints,
    postfilter_synthetic_by_constraints,
    prefilter_data_by_constraints,
)


# ------------------------------------------------------------
# 解析
# ------------------------------------------------------------
class TestParse:
    @pytest.mark.parametrize("raw", [None, "", "none", "null", "{}", "NONE"])
    def test_empty_input_returns_empty(self, raw):
        assert parse_column_constraints(raw) == {}

    @pytest.mark.parametrize("raw", ["{not json", "[1, 2]", "123"])
    def test_invalid_json_returns_empty(self, raw):
        assert parse_column_constraints(raw) == {}

    def test_old_format_is_converted(self):
        raw = json.dumps({"age": {"type": "range", "min": 18, "max": 65}})
        parsed = parse_column_constraints(raw)
        assert parsed == {
            "columns": {"age": {"type": "range", "min": 18, "max": 65}},
            "conditionals": [],
        }

    @pytest.mark.parametrize("rule, expected", [
        ({"type": "range", "min": 1}, {"type": "range", "min": 1, "max": None}),
        ({"type": "min", "value": 0}, {"type": "min", "value": 0}),
        ({"type": "max", "value": 9}, {"type": "max", "value": 9}),
        ({"type": "fixed", "value": "A"}, {"type": "fixed", "value": "A"}),
        ({"type": "options", "values": ["F", "M"]}, {"type": "options", "values": ["F", "M"]}),
        ({"type": "RANGE", "max": 5}, {"type": "range", "min": None, "max": 5}),
    ])
    def test_single_constraint_types(self, rule, expected):
        parsed = parse_column_constraints(json.dumps({"columns": {"c": rule}}))
        assert parsed["columns"]["c"] == expected

    @pytest.mark.parametrize("rule", [
        {"type": "range"},                       # 缺 min/max
        {"type": "min"},                         # 缺 value
        {"type": "options", "values": "F"},      # values 不是 list
        {"type": "unknown", "value": 1},         # 不支援的類型
        {"min": 1},                              # 缺 type
        "range",                                 # 不是 dict
    ])
    def test_invalid_single_constraint_is_dropped(self, rule):
        assert parse_column_constraints(json.dumps({"columns": {"c": rule}})) == {}

    @pytest.mark.parametrize("when", [
        {"column": "a", "operator": "==", "value": 1},
        {"column": "a", "operator": "!=", "value": 1},
        {"column": "a", "operator": ">=", "value": 1},
        {"column": "a", "operator": "<=", "value": 1},
        {"column": "a", "operator": ">", "value": 1},
        {"column": "a", "operator": "<", "value": 1},
        {"column": "a", "operator": "in", "values": [1, 2]},
        {"column": "a", "operator": "not_in", "values": [1, 2]},
        {"column": "a", "operator": "between", "min": 1, "max": 2},
    ])
    def test_all_nine_operators_are_accepted(self, when):
        raw = {"conditionals": [{"when": when, "then": {"column": "b", "type": "min", "value": 0}}]}
        parsed = parse_column_constraints(json.dumps(raw))
        assert len(parsed["conditionals"]) == 1
        assert parsed["conditionals"][0]["then"] == {"column": "b", "type": "min", "value": 0}

    @pytest.mark.parametrize("cond", [
        {"when": {"column": "a", "operator": "~=", "value": 1}, "then": {"column": "b", "type": "min", "value": 0}},
        {"when": {"column": "a", "operator": "in", "value": 1}, "then": {"column": "b", "type": "min", "value": 0}},
        {"when": {"column": "a", "operator": "between", "min": 1}, "then": {"column": "b", "type": "min", "value": 0}},
        {"when": {"column": "a", "operator": "=="}, "then": {"column": "b", "type": "min", "value": 0}},
        {"when": {"column": "a", "operator": "==", "value": 1}, "then": {"type": "min", "value": 0}},
        {"when": {"column": "a", "operator": "==", "value": 1}, "then": {"column": "b", "type": "min"}},
        {"when": {"column": "a", "operator": "==", "value": 1}},
    ])
    def test_invalid_conditional_is_dropped(self, cond):
        assert parse_column_constraints(json.dumps({"conditionals": [cond]})) == {}

    def test_mixed_valid_and_invalid_rules_keep_only_valid(self):
        raw = {
            "columns": {"age": {"type": "min", "value": 0}, "bad": {"type": "nope"}},
            "conditionals": [
                {"when": {"column": "sex", "operator": "==", "value": "F"},
                 "then": {"column": "age", "type": "max", "value": 60}},
                {"when": {"column": "sex", "operator": "??", "value": "F"},
                 "then": {"column": "age", "type": "max", "value": 60}},
            ],
        }
        parsed = parse_column_constraints(json.dumps(raw))
        assert list(parsed["columns"]) == ["age"]
        assert len(parsed["conditionals"]) == 1


# ------------------------------------------------------------
# when / then 子句
# ------------------------------------------------------------
class TestClauses:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"x": [1, 2, 3, 4, 5]})

    @pytest.mark.parametrize("when, expected", [
        ({"operator": "==", "value": 3}, [False, False, True, False, False]),
        ({"operator": "!=", "value": 3}, [True, True, False, True, True]),
        ({"operator": ">=", "value": 4}, [False, False, False, True, True]),
        ({"operator": "<=", "value": 2}, [True, True, False, False, False]),
        ({"operator": ">", "value": 4}, [False, False, False, False, True]),
        ({"operator": "<", "value": 2}, [True, False, False, False, False]),
        ({"operator": "in", "values": [1, 5]}, [True, False, False, False, True]),
        ({"operator": "not_in", "values": [1, 5]}, [False, True, True, True, False]),
        ({"operator": "between", "min": 2, "max": 4}, [False, True, True, True, False]),
    ])
    def test_when_operators(self, df, when, expected):
        mask = _evaluate_when_clause(df, {"column": "x", **when})
        assert mask.tolist() == expected

    def test_when_missing_column_matches_nothing(self, df):
        mask = _evaluate_when_clause(df, {"column": "nope", "operator": "==", "value": 1})
        assert not mask.any()

    @pytest.mark.parametrize("then, expected", [
        ({"type": "range", "min": 2, "max": 4}, [False, True, True, True, False]),
        ({"type": "min", "value": 4}, [False, False, False, True, True]),
        ({"type": "max", "value": 2}, [True, True, False, False, False]),
        ({"type": "fixed", "value": 3}, [False, False, True, False, False]),
        ({"type": "options", "values": [1, 2]}, [True, True, False, False, False]),
    ])
    def test_then_constraints(self, df, then, expected):
        mask = _evaluate_then_constraint(df, {"column": "x", **then})
        assert mask.tolist() == expected

    def test_then_missing_column_passes_all(self, df):
        mask = _evaluate_then_constraint(df, {"column": "nope", "type": "min", "value": 100})
        assert mask.all()

    def test_then_nan_is_treated_as_pass(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        mask = _evaluate_then_constraint(df, {"column": "x", "type": "min", "value": 5})
        assert mask.tolist() == [False, True]


# ------------------------------------------------------------
# 前置過濾
# ------------------------------------------------------------
class TestPrefilter:
    def test_no_constraints_returns_input(self, constraint_df):
        out, report = prefilter_data_by_constraints(constraint_df, {})
        assert out is constraint_df
        assert report == {}

    def test_range_removes_out_of_range_and_keeps_nan(self, constraint_df):
        cons = parse_column_constraints(json.dumps({"age": {"type": "range", "min": 18, "max": 120}}))
        out, report = prefilter_data_by_constraints(constraint_df, cons, min_rows=1)
        # 10 與 130 被移除，NaN 保留
        assert len(out) == 4
        assert out["age"].isna().sum() == 1
        assert report["columns"]["age"]["rows_removed"] == 2
        assert report["_summary"] == {"original_rows": 6, "filtered_rows": 4, "rows_removed": 2}

    def test_options_constraint(self, constraint_df):
        cons = parse_column_constraints(json.dumps({"sex": {"type": "options", "values": ["F", "M"]}}))
        out, _ = prefilter_data_by_constraints(constraint_df, cons, min_rows=1)
        assert set(out["sex"]) == {"F", "M"}

    def test_conditional_rule_only_applies_when_triggered(self, constraint_df):
        # 若 smoker == 1，則 age 必須 <= 60
        cons = parse_column_constraints(json.dumps({"conditionals": [{
            "when": {"column": "smoker", "operator": "==", "value": 1},
            "then": {"column": "age", "type": "max", "value": 60},
        }]}))
        out, report = prefilter_data_by_constraints(constraint_df, cons, min_rows=1)
        # smoker=1 的列：age 25（保留）、70（移除）、130（移除）
        assert sorted(out["age"].dropna().tolist()) == [10, 25, 40]
        assert report["conditionals"][0]["rows_triggered"] == 3
        assert report["conditionals"][0]["rows_removed"] == 2

    def test_missing_column_is_skipped(self, constraint_df):
        cons = parse_column_constraints(json.dumps({"height": {"type": "min", "value": 0}}))
        out, report = prefilter_data_by_constraints(constraint_df, cons, min_rows=1)
        assert len(out) == len(constraint_df)
        assert report["columns"]["height"]["status"] == "skipped"

    def test_falls_back_when_too_few_rows(self, constraint_df):
        cons = parse_column_constraints(json.dumps({"age": {"type": "fixed", "value": 40}}))
        out, report = prefilter_data_by_constraints(constraint_df, cons, min_rows=10)
        assert len(out) == len(constraint_df)
        assert "_warning" in report

    def test_falls_back_when_a_label_class_disappears(self, binary_df):
        # 只保留 age < 45，幾乎所有 label=1（年齡較高）會被濾掉
        cons = parse_column_constraints(json.dumps({"age": {"type": "max", "value": 45}}))
        out, report = prefilter_data_by_constraints(binary_df, cons, min_rows=10, label_column="label")
        assert len(out) == len(binary_df)
        assert "1" in report["_label_fallback"]["missing_labels"]

    def test_backward_compatible_raw_dict(self, constraint_df):
        # 舊呼叫方式：直接傳入欄位 dict（未經 parse）
        out, _ = prefilter_data_by_constraints(
            constraint_df, {"age": {"type": "min", "value": 18}}, min_rows=1)
        assert (out["age"].dropna() >= 18).all()


# ------------------------------------------------------------
# 後置過濾
# ------------------------------------------------------------
class TestPostfilter:
    def test_numeric_range_is_clipped_not_dropped(self):
        df = pd.DataFrame({"age": [-5.0, 30.0, 200.0]})
        cons = parse_column_constraints(json.dumps({"age": {"type": "range", "min": 0, "max": 120}}))
        out, report = postfilter_synthetic_by_constraints(df, cons)
        assert out["age"].tolist() == [0.0, 30.0, 120.0]
        assert report["columns"]["age"]["status"] == "clipped"
        assert report["_summary"]["rows_removed"] == 0

    @pytest.mark.parametrize("rule, expected", [
        ({"type": "min", "value": 0}, [0.0, 30.0, 200.0]),
        ({"type": "max", "value": 100}, [-5.0, 30.0, 100.0]),
    ])
    def test_min_max_clip(self, rule, expected):
        df = pd.DataFrame({"age": [-5.0, 30.0, 200.0]})
        out, _ = postfilter_synthetic_by_constraints(df, parse_column_constraints(json.dumps({"age": rule})))
        assert out["age"].tolist() == expected

    def test_categorical_options_filter_rows(self):
        df = pd.DataFrame({"sex": ["F", "M", "X", "F"]})
        cons = parse_column_constraints(json.dumps({"sex": {"type": "options", "values": ["F", "M"]}}))
        out, report = postfilter_synthetic_by_constraints(df, cons)
        assert out["sex"].tolist() == ["F", "M", "F"]
        assert report["columns"]["sex"]["rows_after"] == 3

    def test_fixed_filter_rows(self):
        df = pd.DataFrame({"site": ["A", "B", "A"]})
        out, _ = postfilter_synthetic_by_constraints(
            df, parse_column_constraints(json.dumps({"site": {"type": "fixed", "value": "A"}})))
        assert out["site"].tolist() == ["A", "A"]

    def test_conditional_rule_removes_violations(self):
        df = pd.DataFrame({"sex": ["F", "F", "M", "M"], "age": [30, 70, 30, 70]})
        cons = parse_column_constraints(json.dumps({"conditionals": [{
            "when": {"column": "sex", "operator": "==", "value": "F"},
            "then": {"column": "age", "type": "max", "value": 60},
        }]}))
        out, report = postfilter_synthetic_by_constraints(df, cons)
        assert out.to_dict("records") == [
            {"sex": "F", "age": 30}, {"sex": "M", "age": 30}, {"sex": "M", "age": 70}]
        assert report["conditionals"][0]["rows_removed"] == 1

    def test_result_fully_satisfies_all_constraints(self, binary_df):
        cons_raw = {
            "columns": {"age": {"type": "range", "min": 30, "max": 55},
                        "sex": {"type": "options", "values": ["F"]}},
            "conditionals": [{"when": {"column": "label", "operator": "==", "value": 1},
                              "then": {"column": "bmi", "type": "min", "value": 30}}],
        }
        out, _ = postfilter_synthetic_by_constraints(binary_df, parse_column_constraints(json.dumps(cons_raw)))
        assert out["age"].between(30, 55).all()
        assert (out["sex"] == "F").all()
        assert (out.loc[out["label"] == 1, "bmi"] >= 30).all()

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"age": [-5.0, 30.0]})
        postfilter_synthetic_by_constraints(
            df, parse_column_constraints(json.dumps({"age": {"type": "min", "value": 0}})))
        assert df["age"].tolist() == [-5.0, 30.0]
