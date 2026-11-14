"""
單元測試：train_tabpfn.py（TabPFN 整合）

以替身取代 TabPFN 本體，只驗證 MLAS 整合層的邏輯：
- 門檻掃描曲線的計算
- 計算裝置選擇（auto / cpu / cuda）
- ensemble 數量防呆（None、0、負數、非數字、超過上限 32）
- 權重版本選擇（2.5 預設 / v2）
- 模型存檔
- main() 的錯誤處理（格式不支援、找不到標籤欄位、未知切分方式）
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from fakes import FakeTabPFNClassifier, install_tabpfn_stubs


@pytest.fixture
def tp(monkeypatch):
    return install_tabpfn_stubs(monkeypatch)


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


class TestThresholdCurves:
    def test_perfect_scores(self, tp):
        y = np.array([0, 0, 1, 1])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        th, acc, prec, rec, f1, spec, npv, cms = tp.generate_threshold_curves(y, proba)
        assert len(th) == 100 and th[0] == pytest.approx(0.01) and th[-1] == pytest.approx(1.0)
        i = th.index(0.5)
        assert acc[i] == 100 and rec[i] == 100 and spec[i] == 100
        assert list(cms[i]) == [2, 0, 0, 2]

    def test_recall_decreases_as_threshold_rises(self, tp):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 200)
        score = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
        _, _, _, rec, *_ = tp.generate_threshold_curves(y, score)
        assert all(a >= b for a, b in zip(rec, rec[1:]))

    def test_accepts_explicit_focus_scores(self, tp):
        y = np.array([0, 1])
        out = tp.generate_threshold_curves(y, np.zeros((2, 3)), num_class=3, y_score=np.array([0.1, 0.9]))
        assert out[1][49] == 100  # 門檻 0.5 時全部正確


class TestTrainTabPFN:
    @pytest.mark.parametrize("given, expected", [
        (None, None), (0, None), (-3, None), ("abc", None), (8, 8), ("16", 16), (50, 32),
    ])
    def test_n_ensembles_guard(self, tp, xy, tmp_path, given, expected):
        X, y = xy
        clf, _ = tp.train_tabpfn(X, y, "m", str(tmp_path), device_mode="cpu", n_ensembles=given)
        assert clf.kwargs.get("n_estimators") == expected

    @pytest.mark.parametrize("mode, expected", [("cpu", "cpu"), ("cuda", "cuda")])
    def test_explicit_device(self, tp, xy, tmp_path, mode, expected):
        X, y = xy
        clf, _ = tp.train_tabpfn(X, y, "m", str(tmp_path), device_mode=mode)
        assert clf.kwargs["device"] == expected

    def test_auto_device_follows_cuda_availability(self, tp, xy, tmp_path, monkeypatch):
        X, y = xy
        monkeypatch.setattr(tp.torch.cuda, "is_available", lambda: False)
        clf, _ = tp.train_tabpfn(X, y, "m", str(tmp_path), device_mode="auto")
        assert clf.kwargs["device"] == "cpu"

    def test_ignores_pretraining_limits(self, tp, xy, tmp_path):
        X, y = xy
        clf, _ = tp.train_tabpfn(X, y, "m", str(tmp_path), device_mode="cpu")
        assert clf.kwargs["ignore_pretraining_limits"] is True

    @pytest.mark.parametrize("version, expected", [("default", "2.5"), ("v2", "v2")])
    def test_model_version(self, tp, xy, tmp_path, version, expected):
        X, y = xy
        clf, _ = tp.train_tabpfn(X, y, "m", str(tmp_path), model_version=version, device_mode="cpu")
        assert clf.version == expected

    def test_model_is_saved(self, tp, xy, tmp_path):
        import joblib
        X, y = xy
        _, path = tp.train_tabpfn(X, y, "heart", str(tmp_path / "task"), device_mode="cpu")
        assert path.endswith("heart_tabpfn.pkl") and os.path.exists(path)
        assert isinstance(joblib.load(path), FakeTabPFNClassifier)


class TestEvaluateModel:
    def test_binary_result_structure(self, tp, xy, tmp_path):
        X, y = xy
        clf, _ = tp.train_tabpfn(X, y, "m", str(tmp_path), device_mode="cpu")
        result = tp.evaluate_model(y, clf.predict(X), clf, X)
        assert result["status"] == "success"
        assert result["num_class"] == 2
        cm = result["confusion_matrix"]
        assert sum(cm.values()) == len(y)
        assert 0 <= result["metrics"]["accuracy"] <= 100


class TestMainErrors:
    def _run(self, tp, capsys, *args):
        tp.main(*args)
        return json.loads(capsys.readouterr().out)

    def test_unsupported_file_format(self, tp, capsys, tmp_path):
        out = self._run(tp, capsys, str(tmp_path / "a.txt"), "label", "train_test_split", "0.8", "m", str(tmp_path))
        assert out["status"] == "error"

    def test_missing_label_column(self, tp, capsys, tmp_path):
        csv = tmp_path / "d.csv"
        pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(csv, index=False)
        out = self._run(tp, capsys, str(csv), "label", "train_test_split", "0.8", "m", str(tmp_path))
        assert out["status"] == "error" and "label" in out["message"]

    def test_unknown_split_strategy(self, tp, capsys, tmp_path):
        csv = tmp_path / "d.csv"
        pd.DataFrame({"x": range(10), "label": [0, 1] * 5}).to_csv(csv, index=False)
        out = self._run(tp, capsys, str(csv), "label", "bootstrap", "0.8", "m", str(tmp_path))
        assert out == {"status": "error", "message": "Unsupported split strategy"}
