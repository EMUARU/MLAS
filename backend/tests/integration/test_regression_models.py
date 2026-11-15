"""
整合測試（回歸）：原系統各模型的訓練、評估、預測與堆疊學習

直接讀取 backend/test_all.bat 的 81 個案例來執行，兩者共用同一份案例清單：
- Windows 上仍可單獨執行 test_all.bat（只看 OK / FAIL）
- pytest 版本另外檢查輸出中沒有 "status": "error"，比 test_all.bat 只看結束碼更嚴格
- 需要的套件（lightgbm、catboost、pytorch-tabnet 等）未安裝時整個檔案跳過；
  XGBoost 與堆疊學習在程式中固定使用 GPU，無 GPU 時跳過

案例之間有先後關係（評估、預測會用到前面訓練出的模型），因此依 test_all.bat 的順序執行，
所有輸出都寫在暫存資料夾，不會留在 backend 目錄。
TabPFN 相關案例需要權重，與 test_tabpfn_pipeline.py 相同，設定 MLAS_RUN_TABPFN=1 才執行。
"""

import os
import re
import shlex
import subprocess
import sys

import pytest

from conftest import BACKEND_DIR, has_module

BAT = os.path.join(BACKEND_DIR, "test_all.bat")
REQUIRED = ["xgboost", "lightgbm", "catboost", "pytorch_tabnet", "shap", "lime"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not all(has_module(m) for m in REQUIRED),
                       reason="需要 " + "、".join(REQUIRED)),
]


def load_cases():
    """從 test_all.bat 取出：資料產生指令，以及每個案例的（編號、名稱、指令）。"""
    with open(BAT, encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\r\n") for l in f]
    setup, cases, step, label = None, [], "", None
    for line in lines:
        s = line.strip()
        m_step = re.match(r"^echo\s+Step\s+(\d+):", s)
        m_case = re.match(r"^echo\s+\[(\d+)/(\d+)\]\s+(.+)$", s)
        if m_step:
            step = m_step.group(1)
        elif m_case:
            label = f"step{step}-{m_case.group(1)}-{m_case.group(3).strip().replace(' ', '_')}"
        elif s.startswith("python -c") and setup is None:
            setup = s[len("python -c"):].strip().strip('"')
        elif s.startswith("python ") and label:
            cmd = re.sub(r"\s*>nul.*$", "", s)
            cases.append((label, cmd))
            label = None
    return setup, cases


SETUP, CASES = load_cases()

# 已知問題清單：案例名稱 → 原因，會標記為 xfail（strict，修好後提醒移除）。
# 2026-11-15 曾列入 5 個（train_mlp 多類別 ×2、train_adaboost 多類別 k_fold、
# CatBoost 多類別在 evaluate.py / predict.py 的 SHAP 崩潰 ×2），已全部修正。
KNOWN_BUGS = {}


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def needs_gpu(cmd):
    """XGBoost 與堆疊學習（meta / base 皆含 XGBoost）在程式中固定使用 GPU。"""
    return any(k in cmd for k in ("train_xgb.py", "m_xgb_", "train_stacking.py"))


def to_argv(cmd):
    """把 .bat 的 python 指令轉成參數列表，腳本改用 backend 的絕對路徑。"""
    parts = shlex.split(cmd.replace('\\"', "\x00"))
    parts = [p.replace("\x00", '"') for p in parts]
    assert parts[0] == "python"
    return [sys.executable, os.path.join(BACKEND_DIR, parts[1]), *parts[2:]]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("regression")
    subprocess.run([sys.executable, "-c", SETUP], cwd=d, check=True, capture_output=True, timeout=300)
    (d / "_t").mkdir()
    return d


def _params():
    for label, cmd in CASES:
        marks = []
        if label in KNOWN_BUGS:
            marks.append(pytest.mark.xfail(reason="已知問題：" + KNOWN_BUGS[label], strict=True))
        yield pytest.param(label, cmd, id=label, marks=marks)


@pytest.mark.parametrize("label, cmd", list(_params()))
def test_case(workdir, label, cmd):
    if "tabpfn" in cmd and os.environ.get("MLAS_RUN_TABPFN") != "1":
        pytest.skip("TabPFN 案例需設定 MLAS_RUN_TABPFN=1")
    if needs_gpu(cmd) and not has_cuda():
        pytest.skip("XGBoost 與堆疊學習固定使用 GPU，無 GPU 時跳過")
    proc = subprocess.run(to_argv(cmd), cwd=workdir, capture_output=True, text=True, timeout=1800)
    detail = f"\n指令: {cmd}\nstdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-1500:]}"
    assert proc.returncode == 0, "結束碼非 0" + detail
    assert not re.search(r'"status"\s*:\s*"error"', proc.stdout), "腳本回報錯誤" + detail
