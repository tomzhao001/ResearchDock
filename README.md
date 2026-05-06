# ResearchDock

论文归档与问答系统（MVP）。当前仓库已完成 **Milestone 1：工程骨架**。

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

适合在本机改代码、看实时日志。需已安装：**Python 3.12+**、**Node.js 18+**、以及 **PostgreSQL（含 pgvector）** 或通过 Compose **仅启动数据库容器**。若要在本机运行 PDF OCR worker，还需安装 **Redis**，并在 `.env` 中配置可用的 GLM-OCR API key。

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
    psql -h 127.0.0.1 -p 5432 -U paper_user -d paper_archive -f db/init/02_seed.sql
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
export FRONTEND_ORIGIN=http://localhost:3000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**PowerShell（Windows，Conda）：**

```powershell
cd backend
conda create -n researchdock python=3.12 -y
conda activate researchdock
pip install -r requirements.txt
$env:APP_SECRET_KEY = "dev-secret"
$env:FRONTEND_ORIGIN = "http://localhost:3000"
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

### 4. 启动前端（Next.js）

另开终端（日志在当前终端）：

**Bash / zsh：**

```bash
cd frontend
npm install
export NEXT_PUBLIC_API_URL=http://localhost:8000
export NEXT_PUBLIC_N8N_URL=http://localhost:5678
npm run dev
```

**PowerShell：en**

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
$env:NEXT_PUBLIC_N8N_URL = "http://localhost:5678"
npm run dev
```

开发地址通常为 [http://localhost:3000](http://localhost:3000)。

### 5. 启动 n8n（命令行）

若不使用 Compose 里的 n8n 容器，可在本机用 **npx** 启动（需 Node.js；日志在当前终端）：

**Bash / zsh：**

```bash
export N8N_PORT=5678
npx n8n
```

**PowerShell：**

```powershell
$env:N8N_PORT = "5678"
npx n8n
```

浏览器打开 [http://localhost:5678](http://localhost:5678)。

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
3. 验收访问：

  | 入口     | URL                                                          | 说明                    |
  | ------ | ------------------------------------------------------------ | --------------------- |
  | 前端     | [http://localhost:3000](http://localhost:3000)               | 登录后进入首页               |
  | 后端健康检查 | [http://localhost:8000/health](http://localhost:8000/health) | 应返回 `{"status":"ok"}` |
  | n8n    | [http://localhost:5678](http://localhost:5678)               | 首次打开按向导初始化            |


---

## 初始账号（仅数据库种子，不可注册）

- **用户名**：`admin`
- **密码**：`123456`

请在生产环境修改密码或删除种子脚本后自行维护用户数据。

种子在首次创建数据库卷时由 `db/init/02_seed.sql` 写入；若需重新初始化，请删除对应的 Docker 数据卷后重新 `docker compose up`。

---

## Milestone 1 验收清单

- `docker compose ps` 中 `frontend`、`backend`、`db`、`n8n` 均为 running
- `GET /health` 返回正常
- 浏览器打开前端，使用 `admin` / `123456` 登录成功并进入首页
- 可打开 n8n Web 界面

---

## 仓库结构（摘录）

```
├── backend/           # FastAPI：健康检查、JWT（HttpOnly Cookie）、/api/auth/*
├── frontend/          # Next.js：登录页、受保护首页
├── db/init/           # PostgreSQL 初始化 SQL（表结构 + admin 种子）
├── docker-compose.yml
└── .env.example
```

---

## 环境变量说明

见根目录 [.env.example](.env.example)。要点：

- `POSTGRES_*`：`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / **`POSTGRES_PORT`** / **`POSTGRES_HOST`**（本机后端用 `127.0.0.1`，Compose 内后台服务由编排注入 **`POSTGRES_HOST=db`**），后端会自动拼接连接串。**可选** `DATABASE_URL`：设置且非空时覆盖拼接结果。
- `POSTGRES_PORT`：数据库容器映射到主机的端口，便于本机 `psql` 与本地 uvicorn 使用 `127.0.0.1`（或 `POSTGRES_HOST`）连接。
- `NEXT_PUBLIC_API_URL`：浏览器访问后端的地址（默认 `http://localhost:8000`）。
- `NEXT_PUBLIC_N8N_URL`：前端「打开 n8n」链接（默认 `http://localhost:5678`）。
- `FRONTEND_ORIGIN`：后端 CORS 允许的前端源（默认 `http://localhost:3000`）。
- `APP_SECRET_KEY`：JWT 签名密钥，生产环境务必更换。
- `OCR_PROVIDER`：当前默认 `glm_ocr`，通过 adapter 统一接入 OCR 服务。
- `LLM_OCR_API_KEY`：智谱 GLM-OCR 的 API key；worker 触发 OCR fallback 时必填。
- `LLM_OCR_BASE_URL` / `LLM_OCR_MODEL`：默认分别为智谱官方 `layout_parsing` 接口与 `glm-ocr` 模型。
- 生产环境若走 HTTPS，请将 `COOKIE_SECURE` 设为 `true`。

