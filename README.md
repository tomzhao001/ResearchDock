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

适合在本机改代码、看实时日志。需已安装：**Python 3.12+**、**Node.js 18+**、以及 **PostgreSQL（含 pgvector）** 或通过 Compose **仅启动数据库容器**。若要在本机运行 PDF OCR worker，还需安装 **Redis**，并在 `.env` 中配置可用的 GLM-OCR API key。若要验证 Milestone 3 的摘要生成与首页对话，还需配置 OpenAI 兼容接口信息。

### 1. 环境与数据库

1. 在仓库根目录复制环境变量。后端会依次读取 **仓库根目录** 的 `.env` 与 `**backend/.env`**（后者可覆盖前者）：
  ```bash
   cp .env.example .env
  ```
2. **数据库**任选其一：
  - **仅起数据库容器**（本机不单独装 PostgreSQL 时）：
     `docker-compose.yml` 已将数据库端口映射到宿主机（默认 `POSTGRES_PORT=5432`，可在 `.env` 中修改）。首次启动会自动执行 `db/init/` 下的建表与种子数据。
  - **本机已安装 PostgreSQL + pgvector**：自行创建库 `paper_archive` 与用户，并手动执行：
    ```bash
    psql -h 127.0.0.1 -p 5432 -U paper_user -d paper_archive -f db/init/01_schema.sql
    ```
    （端口与账号请与本地实例一致。）
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

健康检查：[http://localhost:8000/health](http://localhost:8000/health)

### 3. 启动 Celery Worker（PDF OCR 异步任务）

本地命令行开发时，`celery worker` 需要**单独启动一个终端**。若使用 `docker compose up -d --build` 启全部服务，则 `celery-worker` 会随 Compose 一起启动，无需再手动执行。

默认依赖：

- `REDIS_URL=redis://localhost:6379/1`
- `OCR_PROVIDER=glm_ocr`
- `LLM_OCR_API_KEY=your-api-key`
- `LLM_OCR_MODEL=glm-ocr`
- `OPENAI_API_KEY=your-api-key`
- `OPENAI_MODEL=your-model`

**Bash / zsh：**

```bash
cd backend
source .venv/bin/activate
export REDIS_URL=redis://localhost:6379/1
export CELERY_BROKER_URL=redis://localhost:6379/1
export CELERY_RESULT_BACKEND=redis://localhost:6379/1
export OCR_PROVIDER=glm_ocr
export LLM_OCR_API_KEY=your-api-key
export LLM_OCR_MODEL=glm-ocr
celery -A app.celery_app.celery_app worker --loglevel=info
```

**PowerShell（Windows，Conda）：**

```powershell
cd backend
conda activate researchdock
$env:REDIS_URL = "redis://localhost:6379/1"
$env:CELERY_BROKER_URL = "redis://localhost:6379/1"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
$env:OCR_PROVIDER = "glm_ocr"
$env:LLM_OCR_API_KEY = "your-api-key"
$env:LLM_OCR_MODEL = "glm-ocr"
celery -A app.celery_app.celery_app worker --loglevel=info
```

说明：

- 上传 PDF 后，后端 API 只负责入队；真正的文本提取与 OCR 在 worker 中执行。
- 当前实现优先读取 PDF 文本层；只有页级文本质量不足时才会触发 OCR fallback。
- 当前 OCR fallback 通过智谱官方 `GLM-OCR` 接口完成，不再依赖本地 `tesseract`。
- 本机运行 worker 时，请确认 `redis-server` 已启动，且 `.env` 中已配置 `LLM_OCR_API_KEY`。
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

初始表结构与默认 `admin` 种子都在 `db/init/01_schema.sql` 中；若需重新初始化，请删除对应的 Docker 数据卷后重新 `docker compose up`。

---

## Milestone 3 验收清单

- `docker compose ps` 中 `frontend`、`backend`、`db`、`n8n`、`celery-worker` 均可正常运行
- `GET /health` 返回正常
- 浏览器打开前端，使用 `admin` / `123456` 登录成功并进入论文工作台
- 上传 PDF 后，首页左侧出现论文记录，右侧可查看摘要与文本 / OCR 预览
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
- `OCR_PROVIDER`：当前默认 `glm_ocr`，通过 adapter 统一接入 OCR 服务。
- `LLM_OCR_API_KEY`：智谱 GLM-OCR 的 API key；worker 触发 OCR fallback 时必填。
- `LLM_OCR_BASE_URL` / `LLM_OCR_MODEL`：默认分别为智谱官方 `layout_parsing` 接口与 `glm-ocr` 模型。
- 生产环境若走 HTTPS，请将 `COOKIE_SECURE` 设为 `true`。

