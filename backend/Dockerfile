# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Docling 版面/表格模型依赖 PyTorch。默认装 CPU 版，避免镜像带入 ~3.5GB CUDA 运行时。
# 切换 GPU 时：注释掉下面 CPU 块，取消 GPU 块注释，并在 requirements.txt 中按需恢复 easyocr。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版（需宿主机 NVIDIA 驱动；默认 PyPI 会拉 CUDA 版 torch + nvidia-* 包，镜像体积显著增大）
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install torch torchvision

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
