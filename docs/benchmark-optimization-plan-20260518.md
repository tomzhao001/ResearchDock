# Full Benchmark 观察与优化 Roadmap（2026-05-18）

## 目的

本文档仅用于后续编码规划，基于 `backend/reports/full-benchmark-20260518-102806.json` 的结果，快速总结当前主要问题、优先优化方向，以及建议的实施顺序。

## 本次 Benchmark 的核心观察

### 1. 当前瓶颈首先在检索入口，不在模型本身

- retrieval `hit@10 = 0.5`，说明系统不是完全捞不到证据，但基础召回仍不稳。
- `term_lookup` 表现明显好于 `single_fact` / `multi_turn`，说明 exact-term 路径已经有一定效果，自然语言问法和上下文追问仍是弱项。
- `multi_turn hit@10 = 0.0`，说明真实追问能力基本还没打通。

### 2. 中文 / 跨语言检索是当前最大短板

- `en hit@10 = 0.7273`
- `zh hit@10 = 0.0`
- `mixed hit@10 = 0.25`

这说明系统对英文检索已经具备一定基础能力，但中文问题到英文论文证据的跨语言链路仍明显失效。

### 3. 失败主要集中在 recall，而不是纯排序

- `by_failure_stage.recall = 35`
- `by_failure_stage.chunking = 13`
- `by_failure_stage.fusion = 1`

这说明当前主要问题仍是“候选没有被捞出来”，其次才是 chunk 边界与证据组织问题，fusion 本身不是主要矛盾。

### 4. 端到端回答链路仍明显落后于检索

- `groundedness = 0.0392`
- `citation_precision = 0.0392`
- `support_coverage = 0.3627`
- `abstention_accuracy = 0.0392`

这说明即使部分题已经检索到相关内容，系统也还没有稳定地把证据转成“可归因、可拒答”的答案。

## 主要根因判断

### A. Query rewrite / cross-lingual rewrite 基本未稳定工作

从 benchmark trace 看，大量中文题出现：

- `rewrite_status = llm_failed_fallback`
- `retrieval_query_en = null`

这意味着：

- 中文问题没有被稳定改写成适合英文论文检索的英文表达
- fallback 又只对少数带英文术语的题有效
- 结果是中文自然语言问题既拿不到 sparse，也拿不到有效 dense 候选

这是当前最优先的问题。

### B. Reranker 没有真正稳定参与主链路

大量题目出现：

- `rerank_status = fallback_to_fused`
- `rerank_error = rerank_failed`

说明 reranker 目前更像“名义存在”，但实际上频繁失败或被跳过，无法形成稳定的 second-stage ranking 收益。

### C. Chunking / evidence packaging 仍然影响 summary、多段问题和表格题

当前结构化预处理已经比固定窗口更进一步，但 benchmark 里仍有明显的 `chunking` 失败和 `chunking_risk` 信号，说明：

- 结构信息虽然保留了，但还没有被充分利用
- 小块召回后返回给回答阶段的上下文仍不够稳定
- summary / multi-span / table 问题仍会被 chunk 边界切碎

### D. Attribution-first 回答链路还没闭环

`support_coverage` 高于 `groundedness`，说明部分题目已经能摸到相关证据，但最终答案仍没有被严格约束在证据上。当前问题不仅是“能不能检到”，也是“检到后会不会乱答”。

## 优先优化位置

### P0：先修 query rewrite 与中文检索入口

建议优先处理：

1. 查清 `llm_failed_fallback` 的真实失败原因
2. 给中文问题增加 deterministic rewrite fallback，而不是只依赖 LLM
3. 为常见问题模板补 query normalization
4. 为多轮问题补 decontextualization，把“这里/那个/上一问”改写成显式实体
5. 避免中文 query 在 fallback 状态下只走 dense、不走 sparse

这是当前 ROI 最高的一步。

### P1：修 reranker 可用性与可观测性

建议优先处理：

1. 给 rerank 失败补充详细错误日志
2. 在 benchmark 或启动时增加 reranker health check
3. 先确保 rerank 可以稳定 `applied`，再做排序效果调优

目标不是先做更强 reranker，而是先让现有 reranker 真正工作。

### P2：继续补 Phase 2 的 chunking 与多粒度返回

建议优先处理：

1. parent-child retrieval
2. sentence-window / section-window 返回
3. 表格 caption-body-row 绑定增强
4. summary / multi-span 类问题优先返回 section 级上下文，而不是孤立小块

目标是降低 `chunking` 失败，并改善长段总结、跨段拼接和表格定位。

### P3：补强 attribution-first 回答链路

建议优先处理：

1. 收紧 evidence sufficiency threshold
2. 证据不足时显式拒答
3. 回答只基于 selected evidence 生成
4. 强化 verifier 与 evidence selection 的联动

目标是把“检到一点相关内容就开始回答”改成“证据足够才回答，不足就拒答”。

## 建议的落地顺序

### 第一阶段：先把基础检索链路打稳

1. 修 query rewrite
2. 修中文 fallback
3. 修多轮 decontextualization
4. 修 reranker 可用性

验收重点：

- `zh hit@10` 从 0 拉起来
- `multi_turn hit@10` 不再为 0
- rerank 不再大量 `fallback_to_fused`

### 第二阶段：提升结构化召回与证据稳定性

1. parent-child retrieval
2. section-window / sentence-window
3. table retrieval 强化

验收重点：

- `chunking` 失败数下降
- `summary` / `multi_span` 稳定提升
- 表格题 rank 更靠前

### 第三阶段：把回答链路补成可用

1. evidence selection 收紧
2. verifier 与拒答闭环
3. grounded generation 收敛

验收重点：

- `groundedness`
- `citation_precision`
- `abstention_accuracy`

这三项如果不明显改善，说明系统仍处在“会检索，但不够可信”的阶段。

## 对是否进入更重优化阶段的判断

当前还不建议把主线资源切到重型 `Phase 5`（例如 graph retrieval、完整 RAPTOR、复杂多维知识结构）。

更合理的策略是：

- 主线继续补 `Phase 2 / 3 / 4`
- 等基础检索、中文跨语言链路、rerank、归因回答都更稳定后
- 再做轻量 `Phase 5A` 验证，例如：
  - leaf chunks + section summaries
  - paper summaries
  - 小范围多粒度索引试验

## 一句话结论

当前系统的主要问题不是“模型不够强”，而是：

1. 中文到英文证据的 query rewrite 链路没有稳定跑通
2. reranker 没有真正稳定参与排序
3. chunking 和证据组织还不足以支撑 summary / 多段 / 表格题
4. 回答链路还没有做到严格 grounded 与稳定拒答

后续编码应先把这些基础层打稳，再考虑更重的知识结构升级。
