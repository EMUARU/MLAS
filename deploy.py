import subprocess
import os

# ============================================================
# 部署設定：換機器只需改這一區三個值
#   BASE_DIR   專案在主機上的路徑
#   IMAGE_NAME Docker image 名稱
#   PORT_OUT   對外開放的 port
# ============================================================
BASE_DIR = "/path/to/mlas"
IMAGE_NAME = "mlas"
PORT_OUT = 8081

PORT_IN = 5000
NO_CACHE = False   # 設 True 以 --no-cache 從零重建（驗證環境可重現時使用）

CONTAINER_NAME = f"{IMAGE_NAME}_container"
HOST_UPLOAD_DIR = os.path.join(BASE_DIR, "backend/upload")
HOST_MODEL_DIR = os.path.join(BASE_DIR, "backend/model")
HOST_WHITE_LIST = os.path.join(BASE_DIR, "backend/listWhite.txt")

# 專案需 CUDA GPU 方能執行；主機須先安裝 nvidia-container-toolkit
GPU_FLAG = "--gpus all "

# Hugging Face token（下載 TabPFN 權重用），由環境變數提供，勿寫進程式碼
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def run_cmd(cmd: str):
    print(f"執行中: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"指令失敗: {cmd}")
    else:
        print(f"完成: {cmd}")


def main():
    # 停用並移除舊容器與 image
    run_cmd(f"docker stop {CONTAINER_NAME}")
    run_cmd(f"docker rm {CONTAINER_NAME}")
    run_cmd(f"docker rmi {IMAGE_NAME}")

    # 建立 image
    build_flags = "--no-cache --progress=plain " if NO_CACHE else ""
    run_cmd(f"docker build {build_flags}-t {IMAGE_NAME}:latest .")

    # 啟動新容器
    run_cmd(
        f"docker run {GPU_FLAG}-d "
        f"-p {PORT_OUT}:{PORT_IN} "
        f"--name {CONTAINER_NAME} "
        f"-e HF_TOKEN={HF_TOKEN} "
        f"-v {HOST_UPLOAD_DIR}:/app/upload "
        f"-v {HOST_MODEL_DIR}:/app/model "
        f"-v {HOST_WHITE_LIST}:/app/listWhite.txt "
        f"{IMAGE_NAME}:latest"
    )


if __name__ == "__main__":
    main()
