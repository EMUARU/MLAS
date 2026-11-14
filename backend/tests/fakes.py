"""
測試替身（test doubles）。

TabPFN 需要下載預訓練權重、SHAP / LIME 在 train_tabpfn.py 匯入時會自動 pip install，
單元測試只想驗證 MLAS 自己寫的邏輯，所以用這些替身取代外部套件。
類別定義在可匯入的模組中，才能被 joblib 序列化。
"""

import sys
import types

import numpy as np


class FakeTabPFNClassifier:
    """記錄建構參數；fit 時記住多數類別，predict_proba 依第一個特徵給機率。"""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.version = "2.5"
        FakeTabPFNClassifier.instances.append(self)

    @classmethod
    def create_default_for_version(cls, version, **kwargs):
        obj = cls(**kwargs)
        obj.version = version
        return obj

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        score = 1 / (1 + np.exp(-(X[:, 0] - X[:, 0].mean())))
        if len(self.classes_) == 2:
            return np.column_stack([1 - score, score])
        k = len(self.classes_)
        p = np.full((len(X), k), (1 - score)[:, None] / (k - 1))
        p[:, -1] = score
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class FakeModelVersion:
    V2 = "v2"


def install_tabpfn_stubs(monkeypatch):
    """把 tabpfn / shap / lime 換成替身，並阻止 train_tabpfn.py 匯入時執行 pip install。"""
    FakeTabPFNClassifier.instances.clear()

    tabpfn = types.ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = FakeTabPFNClassifier
    constants = types.ModuleType("tabpfn.constants")
    constants.ModelVersion = FakeModelVersion
    tabpfn.constants = constants

    shap = types.ModuleType("shap")
    lime = types.ModuleType("lime")
    lime_tabular = types.ModuleType("lime.lime_tabular")
    lime_tabular.LimeTabularExplainer = object
    lime.lime_tabular = lime_tabular

    for name, mod in {
        "tabpfn": tabpfn, "tabpfn.constants": constants,
        "shap": shap, "lime": lime, "lime.lime_tabular": lime_tabular,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch = types.ModuleType("torch")
            torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            monkeypatch.setitem(sys.modules, "torch", torch)

    import subprocess
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    for name in ("train_tabpfn", "tool_train"):
        sys.modules.pop(name, None)
    import train_tabpfn
    return train_tabpfn
