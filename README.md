# ResearchDock

论文归档与问答系统（MVP）。当前仓库已推进到 **Milestone 3：摘要与结构化 / 首页工作台**。

## 技术栈


| 组件    | 说明                                                        |
| ----- | --------------------------------------------------------- |
| 前端    | Next.js + Tailwind CSS + shadcn/ui                        |
| 后端    | FastAPI                                                   |
| 数据库   | PostgreSQL + pgvector（Docker 镜像 `pgvector/pgvector:pg16`） |
| 编排占位  | n8n                                                       |
| 一体化部署 | Docker Compose（适合服务器一键拉起）                                 |
| 日常调试  | 本机命令行分别启动前后端与 n8n，日志直接打在终端                                |


详细设计见 [docs/tech-spec.md](docs/tech-spec.md)。

---

## 本地开发（命令行启动，推荐调试）

适合在本机改代码、看实时日志。需已安装：**Python 3.12+**、**Node.js 18+**、以及 **PostgreSQL（含 pgvector）** 或通过 Compose **仅启动数据库容器**。若要在本机运行 PDF 文档解析 worker，还需安装 **Redis**；若要生成图片/图表描述，请在 `.env` 中配置可用的 GLM-4.6V API key。若要验证 Milestone 3 的摘要生成与首页对话，还需配置 OpenAI 兼容接口信息。

### 1. 环境与数据库

1. 在仓库根目录复制环境变量。后端会依次读取 **仓库根目录** 的 `.env` 与 `**backend/.env`**（后者可覆盖前者）：
  ```bash
   cp .env.example .env
  ```
2. **数据库**任选其一：
  - **仅起数据库容器**（本机不单独装 PostgreSQL 时）：
     `docker-compose.yml` 已将数据库端口映射到宿主机（默认 `POSTGRES_PORT=5432`，可在 `.env` 中修改）。首次启动会自动执行 `db/init/` 下的建表、种子数据与 Alembic 基线版本写入。
  - **本机已安装 PostgreSQL + pgvector**：自行创建库 `paper_archive` 与用户，并手动执行：
    创建一个**空库**即可；后端启动时会自动执行 Alembic `upgrade head`，完成建表、默认组织与 `admin` 种子写入。若你接的是一个**已有表但尚未纳入 Alembic 管理**的旧库，首次启动会拒绝自动迁移，此时请在 `backend` 目录执行：
    ```bash
    alembic stamp head
    ```
    若确认旧库已经与当前 schema 对齐，也可临时设置 `DB_AUTO_STAMP_EXISTING_SCHEMA=true` 后启动一次，由后端自动补写版本表。
3. **数据库连接分项**：根目录 `.env` 中与库相关的项为 `POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`、`POSTGRES_PORT`、`POSTGRES_HOST`；后端启动时会自动拼成 `DATABASE_URL`。在宿主机上跑 **uvicorn** 时请保持 **`POSTGRES_HOST=127.0.0.1`**（或本机 IP），**不要使用主机名 `db`**（`db` 只在 Compose 内部可解析）。若修改了 **`POSTGRES_PORT`**（例如 `5433`），无需再改别处。若仍需整串覆盖，可设置可选环境变量 **`DATABASE_URL`**。

### 2. 启动后端（FastAPI）

在 `**backend**` 目录下执行（日志在当前终端；`--reload` 便于断点与热重载）：

**Bash / zsh：**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 若根目录无 .env，可再导出：
export APP_SECRET_KEY=dev-secret
export PUBLIC_ORIGIN=http://localhost:3000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**PowerShell（Windows，Conda）：**

```powershell
cd backend
conda create -n researchdock python=3.12 -y
conda activate researchdock
pip install -r requirements.txt
$env:APP_SECRET_KEY = "dev-secret"
$env:PUBLIC_ORIGIN = "http://localhost:3000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

若提示无法 `conda activate`，可先执行 `conda init powershell` 后**重新打开**终端；环境 `researchdock` 只需创建一次，之后直接进入 `backend` 执行 `conda activate researchdock` 即可。

后端默认会在启动时自动同步数据库到最新 Alembic revision。若你需要在测试、排障或只读环境里关闭它，可设置 `DB_AUTO_MIGRATE_ON_STARTUP=false`。

健康检查：[http://localhost:8000/health](http://localhost:8000/health)

### 3. 启动 Celery Worker（PDF 文档解析异步任务）

本地命令行开发时，`celery worker` 需要**单独启动一个终端**。若使用 `docker compose up -d --build` 启全部服务，则 `celery-worker` 会随 Compose 一起启动，无需再手动执行。

默认依赖：

- `REDIS_URL=redis://localhost:6379/1`
- `DOCUMENT_EXTRACTOR=docling`
- `DOCLING_DO_OCR=true`
- `DOCLING_DO_TABLE_STRUCTURE=true`
- `PICTURE_VLM_MODEL=glm-4.6v`
- `PICTURE_VLM_API_KEY=your-api-key`
- `OPENAI_API_KEY=your-api-key`
- `OPENAI_MODEL=your-model`

**Bash / zsh：**

```bash
cd backend
source .venv/bin/activate
export REDIS_URL=redis://localhost:6379/1
export CELERY_BROKER_URL=redis://localhost:6379/1
export CELERY_RESULT_BACKEND=redis://localhost:6379/1
export DOCUMENT_EXTRACTOR=docling
export DOCLING_DO_OCR=true
export DOCLING_DO_TABLE_STRUCTURE=true
export PICTURE_VLM_MODEL=glm-4.6v
export PICTURE_VLM_API_KEY=your-api-key
celery -A app.celery_app.celery_app worker --loglevel=info
```

**PowerShell（Windows，Conda）：**

```powershell
cd backend
conda activate researchdock
$env:REDIS_URL = "redis://localhost:6379/1"
$env:CELERY_BROKER_URL = "redis://localhost:6379/1"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
$env:DOCUMENT_EXTRACTOR = "docling"
$env:DOCLING_DO_OCR = "true"
$env:DOCLING_DO_TABLE_STRUCTURE = "true"
$env:PICTURE_VLM_MODEL = "glm-4.6v"
$env:PICTURE_VLM_API_KEY = "your-api-key"
celery -A app.celery_app.celery_app worker --loglevel=info
```

说明：

- 上传 PDF 后，后端 API 只负责入队；真正的 Docling 文档解析、表格结构化与图片/图表描述在 worker 中执行。
- 当前实现使用 Docling 作为唯一 PDF 抽取管线，并将页、块、表格、图片结构写入数据库。
- 图片/图表描述默认通过 GLM-4.6V 视觉模型完成；未配置 `PICTURE_VLM_API_KEY` 时不会阻塞整篇文档解析。
- 本机运行 worker 时，请确认 `redis-server` 已启动。
- 默认 `DOCLING_OCR_ENGINE=easyocr` 时，`requirements.txt` 使用 `docling[easyocr]` 安装 EasyOCR；若此前只装过 `docling==2.94.0`，需补装：`pip install "docling[easyocr]==2.94.0"` 或 `pip install easyocr`。
- 若 PDF 自带 text layer 质量很差、会把中文论文解析成整篇乱码，可设置 `DOCLING_FORCE_FULL_PAGE_OCR=true`，强制用整页 OCR 覆盖嵌入文字层。
- 若需把图片裁剪结果继续送入 GLM-4.6V 做图表描述，请设置 `DOCLING_GENERATE_PICTURE_IMAGES=true`；`DOCLING_IMAGES_SCALE` 控制导出图片分辨率，默认 `2.0`。
- **模型缓存卷**：Compose 下 `celery-worker` 挂载命名卷 `model_cache` → 容器内 `/data/models`。通过 `.env` 配置 `MODEL_CACHE_PATH`、`DOCLING_ARTIFACTS_PATH`、`EASYOCR_MODULE_PATH`、`HF_HOME`（后三项可留空，自动落在根目录子路径）。首次解析仍会下载模型，但会写入该卷，**重建容器后无需重复下载**。
- 本机开发可在 `.env` 设置 `MODEL_CACHE_PATH=./data/models`，目录已加入 `.gitignore`。
- 若不想装 EasyOCR，可将 `DOCLING_OCR_ENGINE` 改为 `rapidocr`（需 `pip install rapidocr onnxruntime`）或 `tesseract`（需系统安装 Tesseract）。
- 若已配置 `OPENAI_*` 环境变量，worker 会在文本提取完成后继续生成中文摘要与结构化信息。

### 4. 启动前端（Next.js）

另开终端（日志在当前终端）：

**Bash / zsh：**

```bash
cd frontend
npm install
export NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

**PowerShell：en**

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
npm run dev
```

开发地址通常为 [http://localhost:3000](http://localhost:3000)。登录后首页默认进入论文工作台，可在 `论文` / `对话` tab 间切换。

### 5. 启动 n8n（命令行）

若不使用 Compose 里的 n8n 容器，可在本机用 **npx** 启动（需 Node.js；日志在当前终端）：

**Bash / zsh：**

```bash
npx n8n
```

**PowerShell：**

```powershell
npx n8n
```

浏览器默认打开 [http://localhost:5678](http://localhost:5678)。

### 6. 运行 sample-data eval

用于验证 sample 数据集上的 retrieval / e2e 表现。建议在 `backend` 目录运行，并先确认 `.env` 中数据库与模型接口配置可用。

```bash
cd backend
python -m scripts.sample_data_eval --mode both --subset smoke
```

只跑单个问题时，可用 `--question-id` 指定题号；此时 `--timeout-seconds` 会生效，默认超时为 600 秒：

```bash
python -m scripts.sample_data_eval --mode e2e --subset smoke --question-id en_001
python -m scripts.sample_data_eval --mode e2e --subset smoke --question-id zh_061 --timeout-seconds 300
python -m scripts.sample_data_eval --mode retrieval --subset full --question-id cross_093 --timeout-seconds 180
```

说明：

- `--question-id` 只会运行当前 `subset` 内的单个问题；若题号不在当前子集，会直接报错。
- `--timeout-seconds` 仅在单题模式下生效；传 `0` 或负数表示不启用单题超时。
- 若想把结果落盘，可继续配合 `--output path/to/report.json` 使用。

---

## 快速启动（Docker Compose，适合服务器发布）

1. 复制环境变量模板并按需修改：
  ```bash
   cp .env.example .env
  ```
2. 启动全部服务：
  ```bash
   docker compose up -d --build
  ```
   需要直接在终端看全部容器日志时，去掉 `-d`：
3. 验收访问（Compose 内 `nginx` 仅监听宿主机本机回环地址，由服务器外部 `nginx` 统一接管 HTTPS 与域名入口）：

  | 入口     | URL                                                          | 说明                    |
  | ------ | ------------------------------------------------------------ | --------------------- |
  | 前端     | [https://your-domain](https://your-domain)                   | 由服务器外部 `nginx` 转发到 Compose 内 `nginx` |
  | 后端健康检查 | [https://your-domain/health](https://your-domain/health)     | 由外部 `nginx` 转发，返回 `{"status":"ok"}` |
  | n8n    | [https://your-domain/n8n/](https://your-domain/n8n/)         | 由外部 `nginx` 转发，首次打开按向导初始化 |


---

## 使用 ACR 构建、推送与部署

适合将 `frontend` 与 `backend` 镜像推送到 Azure Container Registry（ACR），再在服务器上拉取并部署。部署相关文件统一放在 `deploy/` 目录中：

- `deploy/build-and-push-acr.sh`
- `deploy/build-and-push-acr.ps1`
- `deploy/deploy-from-acr.sh`
- `deploy/nginx/default.conf`
- `deploy/nginx/external-site.conf.example`

### 1. 配置 ACR 环境变量

先复制环境变量模板：

```bash
cp .env.example .env
```

在 `.env` 中至少补齐以下配置：

```env
ACR_REGISTRY=your-registry.azurecr.io
IMAGE_NAMESPACE=researchdock
IMAGE_TAG=latest
```

若走远程部署，请额外在服务器本机 `nginx` 中配置域名入口。仓库已提供可直接复制的模板：

- `deploy/nginx/external-site.conf.example`

将其中的域名、证书路径和本机回源端口改成你的实际值即可。

### 2. 构建并推送到 ACR

请先自行完成 ACR 登录，例如：

```bash
docker login your-registry.azurecr.io
```

**Bash / zsh：**

```bash
bash deploy/build-and-push-acr.sh
```

**PowerShell（Windows）：**

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\build-and-push-acr.ps1
```

如需指定其他环境文件或镜像 tag，也可以在执行前覆盖环境变量：

```bash
ENV_FILE=.env.prod IMAGE_TAG=v1.0.0 bash deploy/build-and-push-acr.sh
```

```powershell
$env:ENV_FILE = ".env.prod"
$env:IMAGE_TAG = "v1.0.0"
powershell -ExecutionPolicy Bypass -File .\deploy\build-and-push-acr.ps1
```

### 3. 服务器上从 ACR 拉取并部署

部署脚本会复用当前仓库中的 `docker-compose.yml`，但将 `backend`、`celery-worker`、`frontend` 切换为 ACR 镜像；Compose 内部仍通过 `nginx` 汇总前端、后端与 `/n8n/`，而 HTTPS 与域名入口交给服务器本机 `nginx`。

同样请先在服务器上自行完成：

```bash
docker login your-registry.azurecr.io
```

服务器上执行：

```bash
cp .env.example .env
```

填写好服务器自己的 `.env` 后运行：

```bash
bash deploy/deploy-from-acr.sh all
```

如果只想更新代码相关服务（`backend` / `frontend` / `celery-worker`，并在完成后重启内部 `nginx`）：

```bash
bash deploy/deploy-from-acr.sh app
```

如需指定其他环境文件、compose 文件或镜像 tag：

```bash
ENV_FILE=.env.prod COMPOSE_FILE=docker-compose.yml IMAGE_TAG=v1.0.0 bash deploy/deploy-from-acr.sh all
```

```bash
ENV_FILE=.env.prod COMPOSE_FILE=docker-compose.yml IMAGE_TAG=v1.0.0 bash deploy/deploy-from-acr.sh app
```

部署完成后可用以下命令检查：

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
```

说明：

- 构建脚本会分别构建并推送 `backend`、`frontend` 两个镜像。
- 镜像命名规则为 `ACR_REGISTRY/IMAGE_NAMESPACE/backend:IMAGE_TAG` 和 `ACR_REGISTRY/IMAGE_NAMESPACE/frontend:IMAGE_TAG`。
- 部署脚本支持两种模式：`all` 为全量重部署整套 Compose，`app` 为仅重部署应用相关服务。
- 部署脚本会自动执行 `docker compose pull`、`docker compose up -d --force-recreate`，并在完成后额外重启一次 Compose 内 `nginx`。
- `celery-worker` 会复用 `backend` 的同一镜像。
- Compose 内 `nginx` 统一代理前端页面、`/api/*`、`/api/ws/*`、`/health` 与 `/n8n/`，默认仅监听 `127.0.0.1:${FRONTEND_PORT}`。
- 服务器本机 `nginx` 负责监听 `80/443`、配置 SSL 证书，并反向代理到 `http://127.0.0.1:${FRONTEND_PORT}`。
- 如果服务器上的仓库路径不是默认位置，可通过 `ENV_FILE` 与 `COMPOSE_FILE` 指向对应文件。

---

## 初始账号（仅数据库种子，不可注册）

- **用户名**：`admin`
- **密码**：`123456`

请在生产环境修改密码或删除种子脚本后自行维护用户数据。

初始表结构与默认 `admin` 种子由 Alembic 基线 revision 和 `db/init/01_schema.sql` 共同维护：前者用于应用启动自动升级，后者用于 Docker 首次初始化时的快速 bootstrap。若需重新初始化，请删除对应的 Docker 数据卷后重新 `docker compose up`。

---

## Milestone 3 验收清单

- `docker compose ps` 中 `frontend`、`backend`、`db`、`n8n`、`celery-worker` 均可正常运行
- `GET /health` 返回正常
- 浏览器打开前端，使用 `admin` / `123456` 登录成功并进入论文工作台
- 上传 PDF 后，首页左侧出现论文记录，右侧可查看摘要与解析文本预览
- 首页可切换到 `对话` tab，并通过 `OPENAI_*` 配置得到模型回复
- 右上角任务入口可查看进行中和失败任务

---

## 仓库结构（摘录）

```
├── backend/           # FastAPI：认证、论文上传/列表/详情、任务、对话
├── frontend/          # Next.js：登录页、论文工作台、通用对话、任务查看
├── db/init/           # PostgreSQL 初始化 SQL（表结构 + admin 种子）
├── deploy/            # 构建/部署脚本与 nginx 配置
├── docker-compose.yml
└── .env.example
```

---

## 环境变量说明

见根目录 [.env.example](.env.example)。要点：

- `POSTGRES_*`：`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / **`POSTGRES_PORT`** / **`POSTGRES_HOST`**（本机后端用 `127.0.0.1`，Compose 内后台服务由编排注入 **`POSTGRES_HOST=db`**），后端会自动拼接连接串。**可选** `DATABASE_URL`：设置且非空时覆盖拼接结果。
- `POSTGRES_PORT`：数据库容器映射到主机的端口，便于本机 `psql` 与本地 uvicorn 使用 `127.0.0.1`（或 `POSTGRES_HOST`）连接。
- `PUBLIC_ORIGIN`：浏览器访问系统的公开地址；后端 CORS 与 Compose 下的 API 同源访问都基于它。远程部署建议使用 `https://your-domain`。
- `FRONTEND_PORT`：Compose 内 `nginx` 映射到宿主机本机回环地址的 HTTP 端口；推荐设为 `8080`，再由服务器本机 `nginx` 反向代理到 `http://127.0.0.1:FRONTEND_PORT`。
- `PUBLIC_ORIGIN/n8n/`：Compose 部署下 n8n 的公开访问地址，由 `nginx` 反向代理到容器内部的 `5678` 端口。
- `deploy/nginx/external-site.conf.example`：服务器本机 `nginx` 的站点模板，负责 SSL 终止并转发到本机 `FRONTEND_PORT`。
- `APP_SECRET_KEY`：JWT 签名密钥，生产环境务必更换。
- `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`：首页通用对话与论文摘要生成使用的 OpenAI 兼容接口配置。
- `OPENAI_TIMEOUT_SECONDS` / `OPENAI_VERIFY_SSL`：控制聊天与摘要请求的超时和证书校验。
- `DOCUMENT_EXTRACTOR`：当前固定为 `docling`，用于 PDF 文档解析。
- `DOCLING_DO_OCR` / `DOCLING_DO_TABLE_STRUCTURE`：控制 Docling OCR 与表格结构识别步骤。
- `DOCLING_OCR_ENGINE` / `DOCLING_OCR_LANGUAGES`：控制 Docling 标准管线中的 OCR 引擎与语言；默认 `easyocr` 依赖 `docling[easyocr]`（见 `requirements.txt`），非 `docling` 主包自带。
- `DOCLING_FORCE_FULL_PAGE_OCR`：是否强制整页 OCR 并覆盖 PDF 原有文字层；对“内嵌 text layer 有毒、默认解析成乱码”的中文 PDF 更有帮助，但会更依赖 OCR 质量。
- `DOCLING_GENERATE_PICTURE_IMAGES` / `DOCLING_IMAGES_SCALE`：控制 Docling 是否导出图片裁剪结果，以及导出图片分辨率；启用后会把图片 bytes 继续传入当前 GLM 图片描述接口。
- `MODEL_CACHE_PATH` / `DOCLING_ARTIFACTS_PATH` / `EASYOCR_MODULE_PATH` / `HF_HOME`：Docling、EasyOCR、Hugging Face 模型缓存目录；生产建议挂载持久卷（见 `docker-compose.yml` 中 `model_cache`）。
- `PICTURE_VLM_*`：图片/图表描述模型配置，默认使用 GLM-4.6V；API key 不会写入任务元数据。
- 生产环境若走 HTTPS，请将 `COOKIE_SECURE` 设为 `true`。

