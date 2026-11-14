"""
整合測試：Flask API（app.py）中擴充的路由

以 Flask test client 呼叫實際路由，並以假的子程序取代真正的訓練腳本，
驗證「前端請求 → 後端組參數 → 啟動腳本 → 解析輸出 → 更新任務狀態」這段串接：
- /run-synthetic：21 個位置參數的順序與預設值、欄位約束 JSON、雜訊輸出中找出結果 JSON、
  失敗時回傳錯誤並把 status.json 標為 error、任務被終止時的回應
- /run-train-tabpfn：TabPFN 專屬參數只在有提供時才傳給腳本
- /download-synthetic：檔案存在與不存在
"""

import json
import os

import pytest

from conftest import requires_flask

pytestmark = requires_flask


class FakePopen:
    """取代 subprocess.Popen：記錄指令並回傳預先設定的輸出。"""

    calls = []
    stdout = ""
    stderr = ""
    returncode = 0
    on_communicate = None

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        FakePopen.calls.append(cmd)

    def communicate(self):
        self.returncode = FakePopen.returncode
        if FakePopen.on_communicate:
            FakePopen.on_communicate(self)
        return FakePopen.stdout, FakePopen.stderr


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as app_module
    FakePopen.calls = []
    FakePopen.stdout, FakePopen.stderr, FakePopen.returncode = "", "", 0
    FakePopen.on_communicate = None
    monkeypatch.setattr(app_module.subprocess, "Popen", FakePopen)
    # 任務資料夾 model/... 建在暫存目錄；正式環境中工作目錄即 app 根目錄，這裡保持一致
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module.app, "root_path", str(tmp_path))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        c.module = app_module
        yield c


def _status_of(task_dir):
    with open(os.path.join(task_dir, "status.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# /run-synthetic
# ------------------------------------------------------------
class TestRunSynthetic:
    def test_default_arguments_and_order(self, client):
        FakePopen.stdout = json.dumps({"status": "success", "message": "ok"})
        res = client.post("/run-synthetic", json={"file_path": "upload/u/d.csv", "username": "u"})
        body = res.get_json()
        assert body["status"] == "success"

        cmd = FakePopen.calls[0]
        assert cmd[1] == "run_synthetic.py"
        args = cmd[2:]
        assert len(args) == 21
        assert args[:11] == ["upload/u/d.csv", "none", "ctgan", "1x", "300", "500", "1",
                             "256", "256", "128", args[10]]
        assert args[10].replace("\\", "/").startswith("model/u/synthetic/")
        assert args[11:] == ["grouped", "0.0002", "0.0002", "true", "false", "1",
                             "0.95", "10", "min", "none"]

    def test_custom_parameters_and_constraints(self, client):
        FakePopen.stdout = json.dumps({"status": "success"})
        constraints = {"columns": {"age": {"type": "range", "min": 0, "max": 120}}}
        client.post("/run-synthetic", json={
            "file_path": "d.csv", "username": "u", "label_column": "label",
            "generator_type": "tabddpm", "num_samples": "2x", "train_mode": "mixed",
            "enable_tgant": True, "use_auto_params": False, "tgant_metric": "macro",
            "column_constraints": constraints,
        })
        args = FakePopen.calls[0][2:]
        assert args[1] == "label" and args[2] == "tabddpm" and args[3] == "2x"
        assert args[11] == "mixed"
        assert args[14] == "false"     # use_auto_params
        assert args[15] == "true"      # enable_tgant
        assert args[19] == "macro"
        assert json.loads(args[20]) == constraints

    def test_result_json_found_after_library_noise(self, client):
        FakePopen.stdout = "[pyKeOps] warning\nsome log line\n" + json.dumps({"status": "success", "synthetic_rows": 5})
        body = client.post("/run-synthetic", json={"file_path": "d.csv", "username": "u"}).get_json()
        assert body["synthetic_rows"] == 5
        task_dir = os.path.join("model", "u", "synthetic", body["task_id"])
        assert _status_of(task_dir)["status"] == "success"

    def test_script_failure_marks_status_error(self, client):
        FakePopen.returncode = 1
        FakePopen.stderr = "CUDA out of memory"
        body = client.post("/run-synthetic", json={"file_path": "d.csv", "username": "u"}).get_json()
        assert body["status"] == "error"
        assert "CUDA out of memory" in body["message"]
        task_dir = FakePopen.calls[0][12]
        status = _status_of(task_dir)
        assert status["status"] == "error" and "end_time" in status

    def test_script_reported_error_is_propagated(self, client):
        FakePopen.stdout = json.dumps({"status": "error", "message": "label not found"})
        body = client.post("/run-synthetic", json={"file_path": "d.csv", "username": "u"}).get_json()
        assert body["status"] == "error"
        assert _status_of(FakePopen.calls[0][12])["status"] == "error"

    def test_invalid_json_output(self, client):
        FakePopen.stdout = "not json at all"
        body = client.post("/run-synthetic", json={"file_path": "d.csv", "username": "u"}).get_json()
        assert body["status"] == "error"
        assert "JSON" in body["message"]

    def test_terminated_task(self, client):
        def terminate(proc):
            client.module.process_pool.clear()   # 模擬 /terminate-task 已移除此任務
        FakePopen.on_communicate = terminate
        body = client.post("/run-synthetic", json={"file_path": "d.csv", "username": "u"}).get_json()
        assert body["status"] == "terminated"

    def test_task_is_removed_from_pool_after_finish(self, client):
        FakePopen.stdout = json.dumps({"status": "success"})
        body = client.post("/run-synthetic", json={"file_path": "d.csv", "username": "u"}).get_json()
        assert body["task_id"] not in client.module.process_pool


# ------------------------------------------------------------
# /run-train-tabpfn
# ------------------------------------------------------------
class TestRunTrainTabPFN:
    base = {"file_path": "d.csv", "label_column": "label", "split_strategy": "train_test_split",
            "split_value": 0.8, "model_name": "m", "username": "u"}

    def test_minimal_arguments(self, client):
        FakePopen.stdout = json.dumps({"status": "success"})
        client.post("/run-train-tabpfn", json=self.base)
        cmd = FakePopen.calls[0]
        assert cmd[1] == "train_tabpfn.py"
        assert cmd[2:7] == ["d.csv", "label", "train_test_split", "0.8", "m"]
        assert not any(a.startswith("--") for a in cmd)
        assert _status_of(cmd[7])["api"] == "tabpfn"

    def test_optional_arguments_are_forwarded(self, client):
        FakePopen.stdout = json.dumps({"status": "success"})
        client.post("/run-train-tabpfn", json={**self.base, "device": "cpu",
                                               "model_version": "v2", "n_ensembles": 4})
        cmd = FakePopen.calls[0]
        assert cmd[-6:] == ["--device", "cpu", "--model_version", "v2", "--n_ensembles", "4"]

    def test_empty_n_ensembles_is_not_forwarded(self, client):
        FakePopen.stdout = json.dumps({"status": "success"})
        client.post("/run-train-tabpfn", json={**self.base, "n_ensembles": ""})
        assert "--n_ensembles" not in FakePopen.calls[0]

    def test_failure_returns_error(self, client):
        FakePopen.returncode = 1
        FakePopen.stderr = "weights not found"
        body = client.post("/run-train-tabpfn", json=self.base).get_json()
        assert body["status"] == "error"


# ------------------------------------------------------------
# /download-synthetic
# ------------------------------------------------------------
class TestDownloadSynthetic:
    def test_existing_file(self, client):
        d = os.path.join("model", "u", "synthetic", "t1")
        os.makedirs(d)
        with open(os.path.join(d, "out.csv"), "w") as f:
            f.write("a,b\n1,2\n")
        res = client.get("/download-synthetic/u/t1/out.csv")
        assert res.status_code == 200
        assert res.data.replace(b"\r", b"") == b"a,b\n1,2\n"

    def test_missing_file_lists_available(self, client):
        d = os.path.join("model", "u", "synthetic", "t1")
        os.makedirs(d)
        open(os.path.join(d, "real.csv"), "w").close()
        res = client.get("/download-synthetic/u/t1/nope.csv")
        assert res.status_code == 404
        assert res.get_json()["available_files"] == ["real.csv"]
