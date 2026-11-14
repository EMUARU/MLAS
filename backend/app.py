# python -m venv myenv              # 建立虛擬環境
# myenv\Scripts\activate            # 啟動虛擬環境
# pip install -r requirements.txt   # 安裝依賴

# pip freeze > requirements.txt     # 更新環境版本紀錄



from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request
from flask_cors import CORS
import subprocess
import json
import os
import mimetypes
import shutil
from datetime import datetime
import base64
import zipfile
import io
from docx import Document
from cleaner import scheduled_clean
import threading
import sys
import uuid

# 建立 Flask 應用，指定靜態資源位置（給 Vue build 後的檔案用）
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, expose_headers=["Content-Disposition"])  # 啟用跨域支持

# 當訪問 "/" 時，回傳 index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# 支援前端 Vue Router：讓 404 fallback 到 index.html
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

# 儲存每個正在執行的 task_id 與對應的子進程物件
process_pool = {}

# 定時清理 model/ 和 upload/，到 cleaner.py 設定時間
# t = threading.Thread(target=scheduled_clean, daemon=True)
# t.start()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()

    try:
        with open('listWhite.txt', 'r') as f:
            whitelist = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error while loading white list: {e}"
        })

    if username in whitelist:
        return jsonify({
            "status": "success",
            "username": username
        })
    else:
        return jsonify({
            "status": "unauthorized",
            "message": f"{username} is not in the list."
        })

@app.route('/list-files', methods=['POST'])
def list_files():
    data = request.get_json()
    folder_path = data.get('folder_path')
    if not folder_path:
        return jsonify({
            "status": "error",
            "message": "Missing 'folder_path' in request."
        })

    # 取出所有 key 以 ext 開頭的參數（例如 ext1, ext2...）
    # 輸入範例
    # {
    #     "folder_path": "data",
    #     "ext1": "csv",
    #     "ext2": "txt"
    # }
    extensions = []
    for key in sorted(data.keys()):
        if key.startswith("ext") and isinstance(data[key], str) and data[key].strip():
            ext = data[key].strip()
            extensions.append(ext)
    
    try:
        fetch_result = subprocess.run(
            [sys.executable, 'listFiles.py', folder_path] + extensions,
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed or folder not found." + fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/get-field-number', methods=['POST'])
def get_field_number():
    model_path = request.json.get('model_path', None)
    if not model_path:
        return jsonify({
            "status": "error",
            "message": "Missing 'param' in request."
        })
    
    try:
        # fetch_result = subprocess.run(
        #     [sys.executable, 'getFieldNumber.py', model_path],
        #     # capture_output=True,  # 捕獲標準輸出和標準錯誤
        #     stdout=subprocess.PIPE,     # 只捕獲標準輸出
        #     stderr=subprocess.DEVNULL,  # 忽略標準錯誤
        #     text=True                   # 將輸出轉換為字符串
        # )
        
        # # debug 用
        # # print("STDOUT:", result.stdout)  # 打印标准输出
        # # print("STDERR:", result.stderr)  # 打印标准错误
        
        # if fetch_result.returncode != 0:
        #     return jsonify({
        #         "status": "error",
        #         "message": fetch_result.stderr,
        #     })

        # return jsonify(json.loads(fetch_result.stdout))

        args = [
            sys.executable, 'getFieldNumber.py',
            model_path
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)
        if p.returncode != 0:
            return jsonify({
                "status": "error",
                "message": stderr
            })
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-xgb', methods=['POST'])
def run_train_xgb():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')

    try:
        # result = subprocess.run(
        #     [sys.executable, 'train_xgb.py', file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth],
        #     # capture_output=True,        # 捕獲標準輸出和標準錯誤
        #     stdout=subprocess.PIPE,     # 只捕獲標準輸出
        #     stderr=subprocess.DEVNULL,  # 忽略標準錯誤
        #     text=True                   # 將輸出轉換為字符串
        # )

        # # print("STDOUT:", result.stdout)  # 印出標準輸出
        # # print("STDERR:", result.stderr)  # 印出標準錯誤

        # if result.returncode != 0:
        #     return jsonify({
        #         "status": "error",
        #         "message": result.stderr,
        #     })
        
        # return jsonify(json.loads(result.stdout))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "xgb",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        args = [
            sys.executable, 'train_xgb.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, learning_rate, max_depth,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-lgbm', methods=['POST'])
def run_train_lgbm():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')
    num_leaves = data.get('num_leaves')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "lgbm",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_lgbm.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, learning_rate, max_depth, num_leaves,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-rf', methods=['POST'])
def run_train_rf():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    max_depth = data.get('max_depth')
    random_state = data.get('random_state')
    n_jobs = data.get('n_jobs')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "rf",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_rf.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, max_depth, random_state, n_jobs,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-lr', methods=['POST'])
def run_train_lr():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    penalty = data.get('penalty')
    C = data.get('C')
    max_iter = data.get('max_iter')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "lr",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_lr.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            penalty, C, max_iter,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-tabnet', methods=['POST'])
def run_train_tabnet():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    batch_size = data.get('batch_size')
    max_epochs = data.get('max_epochs')
    patience = data.get('patience')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "tabnet",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_tabnet.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            batch_size, max_epochs, patience,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-mlp', methods=['POST'])
def run_train_mlp():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    hidden_layer_1 = data.get('hidden_layer_1')
    hidden_layer_2 = data.get('hidden_layer_2')
    hidden_layer_3 = data.get('hidden_layer_3')
    activation = data.get('activation')
    learning_rate_init = data.get('learning_rate_init')
    max_iter = data.get('max_iter')
    n_iter_no_change = data.get('n_iter_no_change')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "mlp",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_mlp.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            hidden_layer_1, hidden_layer_2, hidden_layer_3,
            activation, learning_rate_init, max_iter, n_iter_no_change,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-catboost', methods=['POST'])
def run_train_catboost():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    iterations = data.get('iterations')
    learning_rate = data.get('learning_rate')
    depth = data.get('depth')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "catboost",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_catboost.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            iterations, learning_rate, depth,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-adaboost', methods=['POST'])
def run_train_adaboost():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    depth = data.get('depth')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "adaboost",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_adaboost.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, learning_rate, depth,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-svm', methods=['POST'])
def run_train_svm():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    C = data.get('C')
    kernel = data.get('kernel')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "svm",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_svm.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            C, kernel,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    

@app.route('/run-train-tabpfn', methods=['POST'])
def run_train_tabpfn():
    # print("[DEBUG] run_train_tabpfn called", flush=True)

    data = request.get_json()
    file_path       = data.get('file_path')
    label_column    = data.get('label_column')
    split_strategy  = data.get('split_strategy')
    split_value     = data.get('split_value')
    model_name      = data.get('model_name')
    username        = data.get('username')

    # TabPFN 相關超參數（前端有給就帶，沒給就略過）
    device         = data.get('device')          # "auto" / "cpu" / "cuda"
    model_version  = data.get('model_version')   # "default" / "v2" ...
    n_ensembles    = data.get('n_ensembles')     # int 或字串，0 代表用預設設定

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id   = f"{timestamp}"
        task_dir  = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)

        # 先寫入 status.json = running
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "tabpfn",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        # 準備呼叫 train_tabpfn.py 的指令
        python_exe = sys.executable  # 確保使用同一個 venv
        args = [
            python_exe, "train_tabpfn.py",
            file_path,
            label_column,
            split_strategy,
            str(split_value),
            model_name,
            task_dir,
        ]

        # 有給才帶入
        if device:
            args += ["--device", str(device)]
        if model_version:
            args += ["--model_version", str(model_version)]
        if n_ensembles not in (None, ""):
            args += ["--n_ensembles", str(n_ensembles)]

        # 開子行程訓練
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        # 等訓練結束，拿 stdout / stderr
        stdout, stderr = p.communicate()

        # 若 task_id 不在 pool，代表中途被手動終止
        if task_id not in process_pool:
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })

        # 1️⃣ 先檢查 returncode / stdout 是否為空
        if p.returncode != 0 or not stdout.strip():
            error_msg = (stderr or "").strip() or "TabPFN training failed with no output."
            status["status"] = "error"
            status["msg"] = error_msg
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)

            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": error_msg
            })

        # 2️⃣ stdout 有內容，再嘗試 parse JSON
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse TabPFN output: {e}"
            status["status"] = "error"
            status["msg"] = error_msg
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)

            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": error_msg
            })

        # 3️⃣ 腳本自己回傳 status="error" 的情況
        if result.get("status") == "error":
            error_msg = result.get("message") or "TabPFN training error."
            status["status"] = "error"
            status["msg"] = error_msg
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)

            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": error_msg
            })

        # 一切正常：更新為 success
        status["status"] = "success"
        status["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        process_pool.pop(task_id, None)
        return jsonify(result)

    except Exception as e:
        # 這裡代表 route 自己掛掉（還沒進到子程式或其它意外）
        return jsonify({
            "status": "error",
            "message": str(e)
        })



@app.route('/run-predict', methods=['POST'])
def run_predict():
    data = request.get_json()
    model_path = data.get('model_path')  # 模型路徑
    mode = data.get('mode')             # 模式：file 或 input
    data_path = data.get('data_path')   # 輸入檔案路徑（file 模式）
    output_name = data.get('output_name')  # 輸出檔案名稱（file 模式）
    input_values = data.get('input_values')  # 特徵值陣列（input 模式）
    label_column = data.get('label_column')
    username = data.get('username')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"{timestamp}"
    task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/

    if not model_path or not mode:
        return jsonify({"status": "error", "message": "model_path and mode are necessary args"})

    command = [sys.executable, 'predict.py', model_path, mode, task_dir]
    if mode == 'file':
        if not data_path or not output_name:
            return jsonify({"status": "error", "message": "data_path 和 output_name 是 file 模式下的必要參數"})
        command.extend(['--data_path', data_path, '--output_name', output_name, '--label_column', label_column])
    elif mode == 'input':
        if not input_values or not isinstance(input_values, list):
            return jsonify({"status": "error", "message": "input_values 是 input 模式下的必要參數，且必須為陣列格式"})
        command.extend(['--input_values'] + list(map(str, input_values)))

    try:
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "predict",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        # 使用 Popen 開啟子進程
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        # 等待子進程結束並接收輸出
        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-evaluate', methods=['POST'])
def run_evaluate():
    data = request.get_json()
    model_path = data.get('model_path')  # 模型路徑
    data_path = data.get('data_path')   # 輸入檔案路徑
    output_name = data.get('output_name')  # 輸出檔案名稱
    label_column = data.get('label_column')
    pred_column = data.get('pred_column')
    username = data.get('username')
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "evaluate",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'evaluate.py',
            model_path, data_path, output_name, label_column, pred_column,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/delete-tabular-rows', methods=['POST'])
def delete_tabular_rows():
    data = request.get_json()
    file_path = data.get('file_path')
    rows = data.get('rows')

    try:
        fetch_result = subprocess.run(
            [sys.executable, 'deleteTabRows.py', file_path, json.dumps(rows)],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        # debug 用
        # print("STDOUT:", fetch_result.stdout)  # 打印标准输出
        # print("STDERR:", fetch_result.stderr)  # 打印标准错误
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/preview-tabular', methods=['POST'])
def preview():
    data = request.get_json()
    file_path = data.get('file_path')
    try:
        fetch_result = subprocess.run(
            [sys.executable, 'checkPreviewTab.py', file_path],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        # debug 用
        # print("STDOUT:", fetch_result.stdout)  # 打印标准输出
        # print("STDERR:", fetch_result.stderr)  # 打印标准错误
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/preprocess', methods=['POST'])
def handle_missing():
    data = request.get_json()
    file_path = data.get('file_path')
    rules = data.get("rules")
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": "File not found."
        })
    try:
        fetch_result = subprocess.run(
            [sys.executable, 'preprocess.py', file_path, json.dumps(rules)],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        # debug 用
        # print("STDOUT:", fetch_result.stdout)  # 打印标准输出
        # print("STDERR:", fetch_result.stderr)  # 打印标准错误
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": fetch_result.stderr,
            })
        return jsonify(json.loads(fetch_result.stdout))
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    download_path = data.get('download_path')

    if not download_path or not os.path.exists(download_path):
        return jsonify({
            "status": "error",
            "message": "Download path not found."
        }), 404

    # 自動判斷 mimetype（根據副檔名）
    mime_type, _ = mimetypes.guess_type(download_path)
    if not mime_type:
        mime_type = 'application/octet-stream'  # 二進位 fallback

    filename = os.path.basename(download_path)
    return send_file(
        download_path,
        mimetype=mime_type,
        as_attachment=True,
        download_name=filename
    )

@app.route('/upload-local-file', methods=['POST'])
def upload_local_file():
    if 'file' not in request.files or 'folder' not in request.form:
        return jsonify({
            "status": "error",
            "message": "Missing file or folder in request."
        })
    
    file = request.files['file']
    folder = request.form['folder']
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected."
        })
    
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, file.filename)
    file.save(save_path)
    return jsonify({
        "status": "success"
    })

@app.route('/upload-samba-file', methods=['POST'])
def upload_samba_file():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    remote_path = data.get('remote_path')
    folder = data.get('folder')

    try:
        fetch_result = subprocess.run(
            [sys.executable, 'uploadSambaFile.py', username, password, remote_path, folder],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        # debug 用
        # print("STDOUT:", result.stdout)  # 打印标准输出
        # print("STDERR:", result.stderr)  # 打印标准错误
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": fetch_result,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/copy-local-file', methods=['POST'])
def copy_local_file():
    data = request.get_json()
    source_path = data.get('source_path')
    target_folder = data.get('target_folder')

    if not source_path or not target_folder:
        return jsonify({
            "status": "error",
            "message": "Missing source_path or target_folder."
        })

    if not os.path.isfile(source_path):
        return jsonify({
            "status": "error",
            "message": f"Source file does not exist: {source_path}"
        })

    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.basename(source_path)
    target_path = os.path.join(target_folder, filename)

    try:
        shutil.copy2(source_path, target_path)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

    return jsonify({
        "status": "success",
    })

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        task_dir = data.get("task_dir")

        if not task_dir or not os.path.exists(task_dir):
            return jsonify({"status": "error", "message": "task_dir not found"}), 404

        # 在記憶體中建立 ZIP 檔
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(task_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=task_dir)
                    zipf.write(full_path, arcname)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=os.path.basename(task_dir) + ".zip"
        )

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/download-stacking-models', methods=['POST'])
def download_stacking_models():
    try:
        data = request.get_json()
        task_dir = data.get("task_dir")

        if not task_dir or not os.path.exists(task_dir):
            return jsonify({"status": "error", "message": "task_dir not found"})

        config_path = os.path.join(task_dir, "stacking_config.json")
        if not os.path.exists(config_path):
            return jsonify({"status": "error", "message": "stacking_config.json not found"})

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        files_to_include = [config_path]

        # 處理 base model 檔名
        for base_name in config.get("base_models", []):
            if base_name == "xgb":
                filename = f"base_{base_name}.json"
            elif base_name == "tabnet":
                filename = f"base_{base_name}.zip"
            else:
                filename = f"base_{base_name}.pkl"
            full_path = os.path.join(task_dir, filename)
            if os.path.exists(full_path):
                files_to_include.append(full_path)
            else:
                return jsonify({"status": "error", "message": f"{filename} not found"})

        # 處理 meta model 檔名
        meta_name = config.get("meta_model")
        if meta_name:
            if meta_name == "xgb":
                filename = f"meta_{meta_name}.json"
            elif meta_name == "tabnet":
                filename = f"meta_{meta_name}.zip"
            else:
                filename = f"meta_{meta_name}.pkl"
            full_path = os.path.join(task_dir, filename)
            if os.path.exists(full_path):
                files_to_include.append(full_path)
            else:
                return jsonify({"status": "error", "message": f"{filename} not found"})

        # 打包成 ZIP 檔
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for full_path in files_to_include:
                arcname = os.path.basename(full_path)
                zipf.write(full_path, arcname)

        zip_buffer.seek(0)
        zip_filename = os.path.basename(task_dir) + "_stacking_models.zip"

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/check-label-uniqueness', methods=['POST'])
def check_label_uniqueness():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    try:
        fetch_result = subprocess.run(
            [sys.executable, 'checkLabelUniqueness.py', file_path, label_column],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        # debug 用
        # print("STDOUT:", result.stdout)  # 打印标准输出
        # print("STDERR:", result.stderr)  # 打印标准错误
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/delete-files', methods=['POST'])
def delete_files():
    data = request.get_json()
    folder_path = data.get('folder_path')  # e.g., "upload/username"
    files = data.get('files', [])          # e.g., ["a.csv", "b.xlsx"]
    if not folder_path or not files:
        return jsonify({
            'status': 'error',
            'message': 'Missing folder_path or files list.'
        })

    deleted = []
    failed = []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted.append(filename)
            else:
                failed.append((filename, "File not found"))
        except Exception as e:
            failed.append((filename, str(e)))

    if failed:
        return jsonify({
            'status': 'error',
            'message': f"Some files could not be deleted: {failed}",
            'deleted': deleted
        })

    return jsonify({
        'status': 'success',
        'message': f"Deleted {len(deleted)} files.",
        'deleted': deleted
    })

# 抓出使用者資料夾內，每個 task_dir 內 status.json 的內容
@app.route('/list-status', methods=['POST'])
def list_status():
    data = request.get_json()
    user_dir = data.get('user_dir')  # e.g. model/alice

    if not user_dir:
        return jsonify({
            "status": "error",
            "message": "Missing 'user_dir' in request."
        })

    try:
        result = subprocess.run(
            [sys.executable, 'listStatus.py', user_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed."
            })

        return jsonify(json.loads(result.stdout))

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# 抓出該 task_dir 中 status.json 的內容
@app.route('/get-status', methods=['POST'])
def get_status():
    data = request.get_json()
    folder_path = data.get('folder_path')

    if not folder_path:
        return jsonify({
            "status": "error",
            "message": "Missing 'folder_path' in request."
        })

    status_path = os.path.join(folder_path, "status.json")
    if not os.path.exists(status_path):
        return jsonify({
            "status": "error",
            "message": f"'status.json' not found in folder: {folder_path}"
        })

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        return jsonify({
            "status": "success",
            "data": status_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to read status.json: {str(e)}"
        })

@app.route('/delete-folders', methods=['POST'])
def delete_folders():
    data = request.get_json()
    folder_path = data.get('folder_path')   # e.g., "model/alice"
    folders = data.get('folders', [])       # e.g., ["a", "b", "c"]

    if not folder_path or not folders:
        return jsonify({
            'status': 'error',
            'message': 'Missing folder_path or folders list.'
        }), 400

    deleted = []
    failed = []

    for subfolder in folders:
        full_path = os.path.join(folder_path, subfolder)
        try:
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
                deleted.append(subfolder)
            else:
                failed.append((subfolder, "Folder not found"))
        except Exception as e:
            failed.append((subfolder, str(e)))

    if failed:
        return jsonify({
            'status': 'error',
            'message': f"Some folders could not be deleted: {failed}",
            'deleted': deleted
        })

    return jsonify({
        'status': 'success',
        'message': f"Deleted {len(deleted)} folders.",
        'deleted': deleted
    })

@app.route('/terminate-task', methods=['POST'])
def terminate_task():
    data = request.get_json()
    task_id = data.get("task_id")
    username = data.get("username")  # 驗證使用者身分（如有登入系統）

    task_info = process_pool.get(task_id)

    if not task_info:
        return jsonify({
            "status": "notRunning",
            "message": f"Task ID '{task_id}' is not running."
        })

    if username and task_info.get("username") != username:
        return jsonify({
            "status": "error",
            "message": "Permission denied: task belongs to another user."
        })

    process = task_info["process"]
    task_dir = task_info["task_dir"]
    status_path = os.path.join(task_dir, "status.json")

    try:
        # 終止子進程
        process.terminate()
        process_pool.pop(task_id, None)

        # 更新 status.json
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        else:
            status = {"task_id": task_id}

        status["status"] = "terminated"
        status["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        return jsonify({
            "status": "success",
            "message": f"Task {task_id} terminated."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to terminate task: {str(e)}"
        })
    
@app.route('/run-train-stacking', methods=['POST'])
def run_train_stacking():
    data = request.get_json()
    base_models = data.get('base_models', [])
    data_name = data.get('data_name')
    label_column = data.get('label_column')
    meta_model = data.get('meta_model')
    username = data.get('username')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)

        # 修復：安全解析 JSON +++
        status_path = os.path.join(task_dir, "status.json")

        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "stacking",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            sys.executable, 'train_stacking.py',  # Flask 執行 subprocess 時使用當前 Python 環境 ***
            data_name, json.dumps(base_models), label_column, meta_model,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))  # 新增cwd參數，設定工作目錄 +++
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        

        # 當 stdout 為空時會崩潰
        # result = json.loads(stdout)  # 如果 stdout 是空的，這裡會拋出 JSONDecodeError
        # 修復：安全解析 JSON +++
        result = None
        if stdout and stdout.strip():
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError as parse_err:
                status['status'] = 'error'
                status['msg'] = f"JSON parse error: {str(parse_err)}. stderr: {stderr[:300] if stderr else 'none'}"
                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4, ensure_ascii=False)
                process_pool.pop(task_id, None)
                return jsonify({
                    "status": "error",
                    "message": status['msg']
                }), 500
        else:
            # stdout 為空
            status['status'] = 'error'
            status['msg'] = f"No output from train_stacking.py. stderr: {stderr[:500] if stderr else 'none'}"
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": status['msg']
            }), 500


        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        # stacking_config.json
        stacking_config = {
            "base_models": base_models,
            "meta_model": meta_model,
        }
        with open(os.path.join(task_dir, "stacking_config.json"), "w", encoding="utf-8") as f:
            json.dump(stacking_config, f, indent=4, ensure_ascii=False)
        return jsonify(json.loads(stdout))
    
    
    except Exception as e:
        # 更新 status.json 為 error
        # 修復：Exception 轉字串
        error_msg = str(e) or "Unknown error." # e 不是字串，Exception 物件不能直接 JSON 序列化
        if task_dir and os.path.exists(task_dir):
            status['status'] = 'error'
            status['msg'] = error_msg
            try:
                with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=4, ensure_ascii=False)
            except:
                pass
            
        # 任務完成後移除該 process 記錄
        if task_id:
            process_pool.pop(task_id, None)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500
    


@app.route('/run-synthetic', methods=['POST'])
def run_synthetic():
    """
    執行合成資料生成（支援 CTGAN 和 TVAE）
    
    支援功能：
    1. 生成器類型：CTGAN（Conditional GAN）、TVAE（Variational Autoencoder）
    2. 分群訓練（Grouped Training）- 支援多類別
    3. TGANT 篩選（TGANT Filtering）
    4. 自動參數調整（根據類別樣本數）
    5. 完整品質評估（Fidelity/Privacy/Utility）
    
    TVAE 論文參數：epochs=5000, compress_dims=(64,64), decompress_dims=(64,64), embedding_dim=64
    """
    data = request.json
    
    # 基本參數
    file_path = data.get('file_path')
    generator_type = data.get('generator_type', 'ctgan')  # 'ctgan', 'tvae' 或 'tabddpm'
    num_samples = data.get('num_samples', '1x')
    username = data.get('username')
    
    # 標籤欄位（單標籤）
    label_column = data.get('label_column', 'none')
    if not label_column:
        label_column = 'none'
    
    # 生成器參數（CTGAN 和 TVAE 共用）
    # SDV 官方預設值：
    # - epochs: 300 (自動參數會根據樣本數倍增: n<100 CTGAN×10/TVAE×15, n<500 CTGAN×5/TVAE×10, ...)
    # - batch_size: 500 (必須是 10 的倍數)
    # - generator_dim: 256 (TVAE 用作 compress_dims，自動調整: n<500 用 128, n<100 用 64)
    # - discriminator_dim: 256 (TVAE 用作 decompress_dims)
    # - embedding_dim: 128
    epochs = str(data.get('epochs', '300'))  # SDV 預設 300，自動參數會根據樣本數倍增
    batch_size = str(data.get('batch_size', '500'))
    discriminator_steps = str(data.get('discriminator_steps', '1'))  # TVAE 不使用
    generator_dim = str(data.get('generator_dim', '256'))      # TVAE: compress_dims
    discriminator_dim = str(data.get('discriminator_dim', '256'))  # TVAE: decompress_dims
    embedding_dim = str(data.get('embedding_dim', '128'))
    
    # 進階參數
    train_mode = data.get('train_mode', 'grouped')  # 'mixed' 或 'grouped'
    generator_lr = str(data.get('generator_lr', '0.0002'))
    discriminator_lr = str(data.get('discriminator_lr', '0.0002'))
    enable_tgant = 'true' if data.get('enable_tgant', False) else 'false'
    generate_multiplier = str(data.get('generate_multiplier', '1'))
    use_auto_params = 'true' if data.get('use_auto_params', True) else 'false'
    
    # TGANT 進階參數（新增）
    tgant_threshold = str(data.get('tgant_threshold', '0.95'))
    tgant_max_gen = str(data.get('tgant_max_gen', '10'))
    tgant_metric = data.get('tgant_metric', 'min')  # 'min' 或 'macro'
    
    # 欄位約束（Column Constraints）
    # 格式: {"col_name": {"type": "range|min|max|fixed|options", ...}, ...}
    column_constraints_raw = data.get('column_constraints', None)
    if column_constraints_raw and isinstance(column_constraints_raw, dict):
        column_constraints_json = json.dumps(column_constraints_raw, ensure_ascii=False)
    else:
        column_constraints_json = 'none'
    
    # 建立任務目錄
    task_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid.uuid4())[:8]
    task_dir = os.path.join('model', username, 'synthetic', task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 寫入 status.json = running（對齊 run-train-tabpfn 等既有訓練路由）
    status = {
        "task_id": task_id,
        "username": username,
        "status": "running",
        "params": data,
        "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "api": "synthetic",
    }
    status_path = os.path.join(task_dir, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=4, ensure_ascii=False)

    def _update_status(state, msg=None):
        # 更新 status.json（success / error）
        status["status"] = state
        if msg:
            status["msg"] = str(msg)[:1000]
        status["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
    
    # 執行訓練腳本
    cmd = [
        sys.executable, 'run_synthetic.py',
        file_path,              # 1. 資料檔案路徑
        label_column,           # 2. 標籤欄位
        generator_type,         # 3. 生成器類型 (ctgan/tvae/tabddpm)
        num_samples,            # 4. 生成數量 (1x/2x/數字)
        epochs,                 # 5. 訓練輪數 (CTGAN:300, TVAE:5000, TabDDPM:n_iter)
        batch_size,             # 6. 批次大小
        discriminator_steps,    # 7. 判別器步數 (TVAE/TabDDPM 不使用)
        generator_dim,          # 8. CTGAN:gen_dim / TVAE:compress_dims / TabDDPM:num_timesteps
        discriminator_dim,      # 9. CTGAN:disc_dim / TVAE:decompress_dims (TabDDPM 不使用)
        embedding_dim,          # 10. 嵌入維度 (TabDDPM 不使用)
        task_dir,               # 11. 輸出目錄
        train_mode,             # 12. 訓練模式 (mixed/grouped)
        generator_lr,           # 13. 生成器學習率 (TVAE 不使用)
        discriminator_lr,       # 14. 判別器學習率 (TVAE 不使用)
        use_auto_params,        # 15. 是否使用自動參數
        enable_tgant,           # 16. 是否啟用 TGANT 篩選
        generate_multiplier,    # 17. 生成倍數
        tgant_threshold,        # 18. TGANT 收斂門檻 (0.5~0.99)
        tgant_max_gen,          # 19. TGANT 最大迭代代數 (3~50)
        tgant_metric,            # 20. TGANT 評估指標 (min/macro)
        column_constraints_json  # 21. 欄位約束 JSON (可選)
    ]
    
    # 以 Popen 啟動並註冊至 process_pool，供 /terminate-task 終止
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process_pool[task_id] = {
        "process": p,
        "username": username,
        "task_dir": task_dir
    }
    stdout_text, stderr_text = p.communicate()

    # 若 task_id 已不在 pool，代表已被 /terminate-task 終止
    # （terminate-task 已將 status.json 更新為 terminated，這裡不再覆寫）
    if task_id not in process_pool:
        return jsonify({
            "status": "terminated",
            "message": "Task was terminated by user.",
            "task_id": task_id
        })
    process_pool.pop(task_id, None)

    # 包裝為 CompletedProcess，使後續既有邏輯完全不變
    result = subprocess.CompletedProcess(cmd, p.returncode, stdout=stdout_text, stderr=stderr_text)
    
    # 錯誤處理
    if result.returncode == 0 and result.stdout.strip():
        try:
            # 過濾掉非 JSON 行（pyKeOps, synthcity 等的輸出）
            stdout_lines = result.stdout.strip().split('\n')
            json_line = None
            
            # 從後往前找，最後一行應該是 JSON
            for line in reversed(stdout_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if json_line:
                output = json.loads(json_line)
                output['task_id'] = task_id
                _update_status('error' if output.get('status') == 'error' else 'success', output.get('message'))
                return jsonify(output)
            else:
                # 嘗試直接解析整個 stdout
                output = json.loads(result.stdout)
                output['task_id'] = task_id
                _update_status('error' if output.get('status') == 'error' else 'success', output.get('message'))
                return jsonify(output)
                
        except json.JSONDecodeError as e:
            _update_status('error', f'JSON 解析錯誤: {str(e)}')
            return jsonify({
                'status': 'error',
                'message': f'JSON 解析錯誤: {str(e)}',
                'stdout': result.stdout[:2000] if result.stdout else 'empty',
                'stderr': result.stderr[:2000] if result.stderr else 'empty'
            })
    else:
        # 詳細錯誤訊息
        error_msg = result.stderr if result.stderr else 'Script execution failed with no output'
        _update_status('error', error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'stdout': result.stdout[:1000] if result.stdout else 'empty',
            'stderr': result.stderr[:1000] if result.stderr else 'empty',
            'returncode': result.returncode,
            'cmd': ' '.join(cmd[:5]) + '...'  # 顯示前幾個參數
        })


@app.route('/download-synthetic/<username>/<task_id>/<filename>')
def download_synthetic(username, task_id, filename):
    """下載合成資料檔案"""
    file_path = os.path.join('model', username, 'synthetic', task_id, filename)
    
    # 調試：列出目錄中的檔案
    task_dir = os.path.join('model', username, 'synthetic', task_id)
    if os.path.exists(task_dir):
        files_in_dir = os.listdir(task_dir)
    else:
        files_in_dir = []
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({
            'status': 'error',
            'message': f'檔案不存在: {file_path}',
            'available_files': files_in_dir  # 顯示實際有哪些檔案
        }), 404

if __name__ == '__main__':
    app.run(debug=True)