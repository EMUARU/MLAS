# -- Build Vue 前端 --
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/ .
RUN npm install && npm run build

# -- 建立 Flask + gunicorn (使用 NVIDIA CUDA base image) --
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# 設定非互動式安裝，避免時區等提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 安裝 Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    build-essential \
    cmake \
    libboost-dev \
    libssl-dev \
    libffi-dev \
    git \
    curl \
    fonts-noto-cjk \
    libpcre3 \
    grep \
    && rm -rf /var/lib/apt/lists/*

# 設定 python3.11 為預設
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# 升級 pip 並安裝 wheel
RUN python -m pip install --upgrade pip setuptools wheel

# 先安裝會與系統套件衝突的 blinker
RUN pip install --ignore-installed blinker

# 安裝 Python 套件（核心相依，含 sdv/sdmetrics；torch 已釘 2.2.2 以相容 synthcity）
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# 安裝 synthcity（TabDDPM 用）。synthcity 0.2.12 依賴宣告較嚴（torch<2.3、networkx<3.0），
# 上一步 requirements.txt 已將 torch 釘為 2.2.2、networkx 釘為 2.8.8，故此處不會反向降級；
# 單獨一行安裝以隔離其大量相依。
RUN pip install --no-cache-dir synthcity==0.2.12

# 複製 Flask 檔案與 Vue 靜態檔
COPY backend/ .
COPY --from=frontend-builder /app/dist ./static

# 若有版本檢查腳本
RUN python tool_checkVersion.py

EXPOSE 5000

# -w 1 --threads 4：1 個處理程序 + 4 個執行緒，避免多個獨立處理程序不共享記憶體導致 process_pool 失效
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
