# 论文归档与问答系统 MVP 技术设计文档

## 1. 文档信息

- 文档类型：Tech Spec
- 目标版本：MVP v0.1
- 部署形态：自部署 Web 应用
- 运行方式：Docker Compose
- 面向对象：个人用户 / 小团队内部使用

## 2. 背景与目标

本项目旨在构建一个可自部署的论文归档与问答系统。用户可以在 Web UI 中配置自己的 OpenAI 兼容 API 信息、公开论文链接抓取规则和基础处理参数，系统定期抓取公开论文链接，或接收用户手动上传的 PDF 文档，自动完成解析、摘要生成、结构化提取、归档与问答。

MVP 阶段的核心目标如下：

- 支持通过 Web UI 配置 OpenAI 兼容 API
- 支持公开论文链接自动抓取
- 支持用户手动上传 PDF
- 支持自动提取摘要、要点和结构化信息
- 支持基于归档论文内容的检索与问答
- 支持通过 Docker 在服务器中部署和运行
- 支持通过 n8n 对抓取与同步流程进行轻量可视化配置

## 3. 非目标

以下内容不纳入 MVP：

- 登录态网站抓取
- 验证码、强反爬网站适配
- 自动全网发现新论文
- 多租户与复杂权限系统
- 复杂 agent 自主规划
- 自动生成综述、周报或专题分析
- 企业级计费、审计、配额系统

## 4. 用户场景

### 4.1 场景一：公开论文链接自动归档

用户在 Web UI 中录入一组公开论文链接，并配置抓取频率。系统定时抓取网页，优先发现并下载 PDF，提取文本内容，调用 OpenAI 兼容模型生成摘要和结构化信息，最后建立检索索引并展示在归档库中。

### 4.2 场景二：手动上传 PDF

用户手动上传 PDF 文件，系统解析 PDF 正文并执行与自动抓取相同的摘要、结构化、索引和归档流程。

### 4.3 场景三：面向归档论文自由问答

用户在问答页面输入问题，系统从归档论文中检索相关文本片段，调用模型生成回答，并附带引用来源，帮助用户核验答案。

## 5. 需求概述

## 5.1 功能需求

### 5.1.1 配置管理

系统需要支持以下配置项：

- OpenAI 兼容 API Base URL
- API Key
- Chat Model 名称
- Embedding Model 名称
- 抓取 URL 列表
- 抓取频率
- 单次抓取数量限制
- 摘要语言
- chunk 大小与 overlap

### 5.1.2 文档输入

系统仅支持两种输入方式：

- 公开论文链接
- 用户手动上传 PDF

### 5.1.3 归档处理

每篇论文需要完成以下处理：

- 识别文档元信息
- 提取正文文本
- 去重
- 生成摘要
- 生成结构化结果
- 文本切块
- embedding 建立
- 向量检索入库

### 5.1.4 检索与问答

系统需要支持：

- 论文列表浏览
- 论文详情查看
- 基于论文归档内容的自由提问
- 返回带引用来源的回答

## 5.2 非功能需求

- 支持单机 Docker Compose 部署
- 初次部署后可通过 UI 完成基本配置
- 后端具备基础任务重试与错误记录能力
- 问答回答必须尽量基于归档内容，不允许无依据扩写
- 结构设计为后续扩展多用户或更多来源预留空间

## 6. 总体架构

MVP 推荐采用如下架构：

- 前端：`Next.js`
- 后端 API：`FastAPI`
- 工作流编排与定时调度：`n8n`
- 数据库：`PostgreSQL`
- 向量存储：`pgvector`
- 文档解析：`PyMuPDF` / `pypdf` / `BeautifulSoup`
- 文件存储：本地挂载目录或对象存储兼容目录
- 部署方式：`Docker Compose`

系统组件职责如下：

### 6.1 Web UI

负责：

- 系统配置管理
- 链接源管理
- PDF 上传
- 论文归档列表展示
- 论文详情页展示
- 问答页面
- 任务状态展示

### 6.2 API 服务

负责：

- 接收前端上传文件
- 接收 n8n 的抓取请求
- 下载公开网页和 PDF
- 解析正文
- 调用 OpenAI 兼容 API
- 生成摘要与结构化信息
- 执行切块与 embedding
- 提供问答接口
- 管理数据库读写

### 6.3 n8n

负责：

- 定时触发同步任务
- 遍历配置的公开链接
- 调用后端抓取接口
- 可视化展示抓取流程
- 为后续扩展保留轻量可配置入口

### 6.4 PostgreSQL + pgvector

负责：

- 存储论文元数据
- 存储摘要和结构化字段
- 存储切块文本
- 存储 embedding
- 存储任务记录

## 7. 架构决策

## 7.1 为什么使用 n8n

n8n 适合处理定时任务、HTTP 调用、任务流可视化和轻量配置。MVP 中使用 n8n 的目标不是承载完整业务逻辑，而是作为调度和可配置工作流入口。

### 设计原则

- 业务核心逻辑在代码服务中实现
- n8n 只负责编排、触发和调用
- 尽量避免将检索、切块、问答策略硬编码进 n8n 节点中

## 7.2 为什么使用 FastAPI

FastAPI 适合快速构建结构清晰的后端服务，便于提供同步接口、后台任务接口和文件上传接口，并且与 Python 文档解析和 AI 生态兼容较好。

## 7.3 为什么使用 PostgreSQL + pgvector

MVP 优先追求部署简单。`pgvector` 可以直接在 PostgreSQL 中完成向量存储和检索，避免引入额外的向量数据库组件。后续若数据规模提升，可迁移到 `Qdrant`。

## 8. 关键流程设计

## 8.1 公开链接抓取流程

1. n8n 根据计划任务触发同步
2. n8n 读取配置的链接列表
3. n8n 逐条调用后端抓取接口
4. 后端请求网页 HTML
5. 后端提取标题、摘要、作者、发布日期和 PDF 链接
6. 若存在 PDF，优先下载 PDF
7. 若不存在 PDF，则解析网页正文
8. 后端计算内容 hash 并执行去重
9. 后端调用模型生成摘要和结构化结果
10. 后端切块并生成 embedding
11. 后端写入数据库
12. 前端展示任务结果和归档结果

## 8.2 PDF 上传流程

1. 用户在 Web UI 上传 PDF
2. 前端调用上传接口
3. 后端保存文件
4. 后端解析 PDF 文本
5. 后端抽取元信息
6. 后端调用模型生成摘要和结构化结果
7. 后端切块、生成 embedding 并写库
8. 前端展示归档结果

## 8.3 问答流程

1. 用户在问答页输入问题
2. 前端调用问答接口
3. 后端基于问题生成 query embedding
4. 在 `paper_chunks` 中检索相关文本片段
5. 后端拼装 prompt，要求模型仅基于引用内容回答
6. 模型返回回答
7. 后端返回答案及引用片段、论文标题和来源链接
8. 前端展示答案和引用来源

## 9. 数据模型

以下为建议的数据表结构，字段可在实现时按框架习惯微调。

## 9.1 `app_settings`

用于保存系统级配置。

字段建议：

- `id`
- `openai_base_url`
- `openai_api_key_encrypted`
- `chat_model`
- `embedding_model`
- `default_summary_language`
- `default_chunk_size`
- `default_chunk_overlap`
- `created_at`
- `updated_at`

说明：

- API Key 应加密存储
- 第一版可以默认单实例配置，后续再扩展用户维度

## 9.2 `sources`

用于保存公开抓取源配置。

字段建议：

- `id`
- `name`
- `url`
- `source_type`
- `enabled`
- `schedule_cron`
- `max_items_per_run`
- `last_run_at`
- `created_at`
- `updated_at`

## 9.3 `papers`

用于保存论文主实体。

字段建议：

- `id`
- `title`
- `authors`
- `abstract_raw`
- `source_url`
- `pdf_url`
- `doi`
- `published_at`
- `content_hash`
- `ingest_type`
- `status`
- `created_at`
- `updated_at`

说明：

- `ingest_type` 取值如 `link`、`upload`
- `status` 取值如 `pending`、`processed`、`failed`

## 9.4 `paper_assets`

用于保存原始文件与中间产物。

字段建议：

- `id`
- `paper_id`
- `asset_type`
- `storage_path`
- `mime_type`
- `raw_text`
- `metadata_json`
- `created_at`

## 9.5 `paper_summaries`

用于保存模型生成结果。

字段建议：

- `id`
- `paper_id`
- `summary_language`
- `abstract_zh`
- `summary_points`
- `research_problem`
- `method`
- `findings`
- `limitations`
- `model_name`
- `prompt_version`
- `created_at`

## 9.6 `paper_chunks`

用于保存检索切块。

字段建议：

- `id`
- `paper_id`
- `chunk_index`
- `content`
- `embedding`
- `token_count`
- `page_from`
- `page_to`
- `metadata_json`
- `created_at`

## 9.7 `jobs`

用于保存任务执行记录。

字段建议：

- `id`
- `job_type`
- `source_id`
- `paper_id`
- `status`
- `error_message`
- `retry_count`
- `started_at`
- `finished_at`
- `created_at`

## 10. API 设计

以下接口为 MVP 建议接口，实际实现时可按 REST 风格继续细分。

## 10.1 配置接口

### `GET /api/settings`

获取当前系统配置。

### `PUT /api/settings`

更新系统配置。

请求体示例：

```json
{
  "openaiBaseUrl": "https://example.com/v1",
  "openaiApiKey": "sk-xxx",
  "chatModel": "gpt-4o-mini",
  "embeddingModel": "text-embedding-3-small",
  "defaultSummaryLanguage": "zh-CN",
  "defaultChunkSize": 1000,
  "defaultChunkOverlap": 150
}
```

## 10.2 链接源接口

### `GET /api/sources`

获取抓取源列表。

### `POST /api/sources`

新增抓取源。

### `PUT /api/sources/{id}`

更新抓取源。

### `DELETE /api/sources/{id}`

删除抓取源。

### `POST /api/sources/{id}/sync`

手动触发单个抓取源同步。

## 10.3 上传接口

### `POST /api/papers/upload`

上传 PDF 并归档。

请求类型：

- `multipart/form-data`

字段：

- `file`

## 10.4 抓取接口

### `POST /api/ingest/link`

供 n8n 或前端调用，用于抓取单个公开链接。

请求体示例：

```json
{
  "sourceId": "optional-source-id",
  "url": "https://example.com/paper-page"
}
```

## 10.5 论文接口

### `GET /api/papers`

分页获取论文列表。

支持参数：

- `keyword`
- `status`
- `sourceId`
- `page`
- `pageSize`

### `GET /api/papers/{id}`

获取论文详情。

### `GET /api/papers/{id}/chunks`

获取论文切块信息，主要用于调试或内部管理页。

## 10.6 问答接口

### `POST /api/qa/query`

基于归档论文提问。

请求体示例：

```json
{
  "question": "这篇论文的核心方法是什么？",
  "paperIds": [],
  "topK": 6
}
```

返回体示例：

```json
{
  "answer": "该论文主要提出了......",
  "citations": [
    {
      "paperId": "paper-1",
      "paperTitle": "Example Paper",
      "sourceUrl": "https://example.com/paper",
      "snippet": "The proposed method uses..."
    }
  ]
}
```

## 10.7 任务接口

### `GET /api/jobs`

获取最近任务记录。

### `GET /api/jobs/{id}`

获取任务详情。

## 11. 文档抓取与解析策略

## 11.1 输入类型识别

后端接收 URL 后按以下顺序识别：

1. 是否为直接 PDF 链接
2. 是否为普通 HTML 页面
3. 是否可从页面中发现 PDF 链接

优先级规则：

- PDF 优先于网页正文
- 若 PDF 获取失败，则回退到网页正文解析

## 11.2 网页解析策略

MVP 中只做轻量通用解析：

- 通过 `httpx` 拉取 HTML
- 使用 `BeautifulSoup` 提取标题、meta 信息和正文
- 扫描可能的 PDF 链接
- 对常见论文站点保留轻量适配能力

不做：

- 浏览器自动化抓取
- 登录态抓取
- 复杂站点行为模拟

## 11.3 PDF 解析策略

推荐优先使用 `PyMuPDF`，保留 `pypdf` 作为补充。

流程：

1. 读取 PDF
2. 按页提取文本
3. 清洗页眉页脚和重复噪音
4. 合并正文
5. 保留页码范围信息，供引用展示

## 11.4 去重策略

去重应至少基于以下信号组合：

- `source_url`
- `pdf_url`
- `doi`
- `title`
- `content_hash`

原则：

- 明显重复文档不重复入库
- 同论文不同入口应尽量合并到同一 `paper`

## 12. 摘要与结构化提取设计

## 12.1 输出结构

每篇论文默认生成以下字段：

- 中文摘要
- 3 到 5 条要点总结
- 研究问题
- 方法
- 主要发现
- 局限性

## 12.2 模型调用原则

- 使用用户配置的 OpenAI 兼容接口
- Prompt 固定为模板化结构
- 输出要求 JSON 化，便于结构化存储
- 若结构化解析失败，保留原始输出并记录错误

## 12.3 Prompt 原则

- 明确限制只基于给定文本生成
- 明确字段定义
- 明确不确定时应返回空值或说明信息不足

## 13. 检索与问答设计

## 13.1 切块策略

MVP 默认策略：

- 按字符或 token 切块
- `chunk_size` 默认 800 到 1200
- `chunk_overlap` 默认 100 到 200

切块时保留：

- `paper_id`
- 页码范围
- chunk 顺序
- chunk 元数据

## 13.2 检索策略

MVP 采用标准向量检索：

- query embedding
- top-k 检索
- 简单重排序可后置

默认参数：

- `top_k = 5 ~ 8`

## 13.3 回答约束

Prompt 中需明确：

- 只允许基于检索结果回答
- 若证据不足，必须明确说明
- 必须返回引用内容对应的来源信息

## 14. 前端页面设计

MVP 页面建议如下：

## 14.1 设置页

功能：

- 配置 OpenAI 兼容 API
- 配置默认摘要与检索参数

## 14.2 抓取源管理页

功能：

- 新增、编辑、删除公开链接源
- 启用 / 禁用
- 手动触发同步
- 查看最近同步时间

## 14.3 上传页

功能：

- 上传 PDF
- 查看上传处理状态

## 14.4 论文列表页

功能：

- 查看归档论文
- 搜索与筛选
- 按时间排序

## 14.5 论文详情页

功能：

- 查看论文标题、作者、来源链接
- 查看中文摘要
- 查看结构化字段
- 查看原始来源

## 14.6 问答页

功能：

- 自由提问
- 查看答案
- 查看引用片段和来源论文

## 14.7 任务页

功能：

- 查看抓取任务
- 查看错误信息
- 手动重试失败任务

## 15. n8n 集成设计

## 15.1 定位

n8n 作为可视化工作流层，承担：

- 定时触发
- 抓取链接列表遍历
- 调用后端同步接口
- 简单失败重试

## 15.2 建议工作流

工作流名称：`paper-source-sync`

建议节点：

1. Schedule Trigger
2. HTTP Request 获取启用中的抓取源
3. Split In Batches 遍历抓取源
4. HTTP Request 调用 `/api/ingest/link`
5. IF 判断成功/失败
6. 记录日志或发送通知

## 15.3 设计边界

不建议放在 n8n 中实现：

- PDF 解析
- 切块
- embedding
- 检索策略
- RAG prompt 拼装

这些逻辑统一放在后端服务中，避免工作流变得不可维护。

## 16. 安全设计

## 16.1 API Key 存储

- API Key 需加密存储
- 前端展示时只回显掩码
- 日志中不得打印完整密钥

## 16.2 文件安全

- 仅允许上传 PDF
- 校验 MIME type 和扩展名
- 限制单文件大小
- 文件存储目录与服务代码隔离

## 16.3 问答边界

- 问答接口默认只读取归档论文内容
- 不开放任意外部工具调用

## 16.4 服务隔离

Docker Compose 中建议：

- API、DB、n8n 分服务运行
- 挂载目录分离
- 仅暴露必要端口

## 17. 日志与可观测性

MVP 至少需要：

- 抓取任务日志
- PDF 解析错误日志
- 模型调用错误日志
- 问答请求日志
- 任务成功 / 失败统计

建议：

- 后端使用结构化日志
- 为每次归档任务生成 trace id
- 前端任务页展示最近执行结果

## 18. 部署设计

## 18.1 运行组件

Docker Compose 至少包含以下服务：

- `frontend`
- `backend`
- `db`
- `n8n`

可选：

- `nginx`

## 18.2 环境变量

建议至少包含：

```env
POSTGRES_DB=paper_archive
POSTGRES_USER=paper_user
POSTGRES_PASSWORD=change_me
BACKEND_PORT=8000
FRONTEND_PORT=3000
N8N_PORT=5678
FILE_STORAGE_PATH=/data/files
APP_SECRET_KEY=change_me
```

说明：

- OpenAI 兼容 API 配置优先通过 Web UI 保存到数据库
- 环境变量只存放系统级启动配置

## 18.3 首次部署流程

1. 准备服务器与 Docker 环境
2. 配置 `.env`
3. 执行 `docker compose up -d`
4. 打开前端页面
5. 在设置页配置 OpenAI 兼容 API 信息
6. 在抓取源页添加公开论文链接
7. 手动触发首次同步或等待定时任务

## 19. 里程碑与实施计划

## Milestone 1：工程骨架

目标：

- 搭建前端、后端、数据库、n8n、Docker Compose

验收：

- 服务全部可启动
- 前端可访问
- 后端健康检查可用

## Milestone 2：输入与归档

目标：

- 实现公开链接抓取
- 实现 PDF 上传
- 实现去重与原文入库

验收：

- 至少能成功处理公开链接和上传 PDF 各一条

## Milestone 3：摘要与结构化

目标：

- 跑通模型调用
- 保存摘要与结构化字段

验收：

- 论文详情页能看到完整摘要结果

## Milestone 4：问答

目标：

- 实现切块、embedding、检索和问答

验收：

- 问答页能基于归档内容返回带引用的回答

## Milestone 5：产品化完善

目标：

- 增强任务页
- 完善 n8n 集成
- 完善错误处理和部署文档

验收：

- 可在服务器部署并稳定演示

## 20. 风险与应对

### 20.1 风险：网页结构不稳定

应对：

- 优先 PDF
- HTML 只做通用轻解析
- 上传 PDF 作为兜底路径

### 20.2 风险：模型输出结构不稳定

应对：

- 使用 JSON 输出约束
- 增加解析失败重试
- 保留原始响应用于排查

### 20.3 风险：回答幻觉

应对：

- 强制基于检索片段回答
- 必须返回引用
- 证据不足时明确拒答

### 20.4 风险：MVP 范围失控

应对：

- 严格只做公开链接和上传 PDF 两种输入
- 不引入复杂 agent
- 不引入登录抓取

## 21. 后续扩展方向

MVP 完成后，可按以下方向扩展：

- 多用户配置隔离
- 更丰富的结构化抽取字段
- 基于主题的专题综述生成
- 支持标签、收藏和自定义分类
- 支持登录态抓取
- 向量库迁移到 `Qdrant`
- 接入更复杂的重排序与混合检索

## 22. 待确认问题

以下问题建议在正式开工前确认：

- MVP 是否需要真实用户系统，还是单管理员配置即可
- 上传文件是否需要对象存储，还是本地挂载足够
- 第一版问答是否允许限定在指定论文范围内
- 是否需要最基础的通知机制，如同步失败提示
- 是否需要支持中英双语摘要输出

---

如果本技术方案确认通过，下一步建议继续产出：

1. `README.md`：项目启动和部署说明
2. `api-spec.md`：接口详细定义
3. `schema.sql` 或 ORM 模型草案
4. `docker-compose.yml` 初版
