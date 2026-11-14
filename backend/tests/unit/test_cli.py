"""
單元測試：synthetic.cli（命令列進入點）的參數與錯誤處理

只測不需要實際訓練生成器的路徑；完整流程見 integration/test_synthetic_pipeline.py。
"""

import json
import sys

import pandas as pd
import pytest

from conftest import requires_sdv


def _run_main(monkeypatch, capsys, argv):
    from synthetic.cli import main
    monkeypatch.setattr(sys, "argv", ["run_synthetic.py", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    out = capsys.readouterr().out.strip().splitlines()[-1]
    return exc.value.code, json.loads(out)


def test_too_few_arguments(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, ["a.csv", "label", "ctgan"])
    assert code == 1
    assert out["status"] == "error"
    assert "參數不足" in out["message"]


def test_unsupported_file_format(monkeypatch, capsys, tmp_path):
    argv = [str(tmp_path / "data.txt"), "label", "ctgan", "10", "1", "10", "1", "8", "8", "8", str(tmp_path)]
    code, out = _run_main(monkeypatch, capsys, argv)
    assert code == 1
    assert "不支援的檔案格式" in out["message"]


@requires_sdv
def test_unsupported_generator(monkeypatch, capsys, tmp_path, binary_df):
    csv = tmp_path / "d.csv"
    binary_df.to_csv(csv, index=False)
    argv = [str(csv), "label", "gaussian", "10", "1", "10", "1", "8", "8", "8", str(tmp_path / "out"), "mixed"]
    code, out = _run_main(monkeypatch, capsys, argv)
    assert code == 1
    assert "不支援的生成器類型" in out["message"]
