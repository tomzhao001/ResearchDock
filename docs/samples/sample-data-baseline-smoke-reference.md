# Sample-data Smoke 评测参照（Phase 0 / 1 / 2）

本文档从各阶段完整报告中抽取关键数字，便于后续在**同一评测配置**下持续对比。完整逐题明细仍在 JSON 中，本页更适合作为“先看什么、怎么解读”的速查表。

- **Phase 0 完整报告：** `backend/reports/sample-data-report.json`
- **Phase 1 完整报告：** `backend/reports/sample-data-report-phase1-hybrid.json`
- **Phase 2 完整报告：** `backend/reports/sample-data-report-phase2.json`

---

## 评测配置

| 字段 | 值 |
|------|-----|
| `mode` | `both`（检索 + 端到端） |
| `subset` | `smoke`（12 题端到端；含 gold 的检索题为 11，另 1 题跳过 gold） |
| `judge_mode` | `heuristic`（启发式判分，非 LLM 裁判） |

---

## 先怎么看这份结果

1. **先确认口径一致。** 只有在 `mode`、`subset`、`judge_mode` 一致时，跨阶段数字才可直接对比。
2. **先看汇总，再看分桶。** 优先看 `retrieval.summary` 与 `end_to_end.summary`；分桶 (`by_category` / `by_language`) 用来定位问题，不用来替代总分。
3. **最后回到逐题 JSON。** 如果某个汇总指标变化明显，要去看 `retrieval.questions` / `end_to_end.questions`，确认是整体改善，还是被少数题拉动。
4. **不要把所有字段都当成“越高越好”的总目标。** 有些字段本身是计数或辅助信号；有些指标虽然数值上更高更好，但只能在相同题集、相同判分口径下解释。

## 指标解读

| 字段 | 怎么看 |
|------|--------|
| `count` | 样本数，不是效果分数；主要用于确认对比口径是否一致。 |
| `skipped_without_gold` | 越低越好，说明可参与检索评测的题更多。 |
| retrieval `hit@k` | 越高越好，表示 gold evidence 是否出现在前 `k` 条结果里。适合看“有没有捞到”。 |
| retrieval `ndcg@k` | 越高越好，比 `hit@k` 更关注 gold 排名是否靠前。适合看“排得好不好”。 |
| retrieval `mrr` | 越高越好，强调首个正确结果的位置；对排序质量更敏感。 |
| e2e `groundedness` | 越高越好，表示回答是否真正被证据支撑。这个值不只受检索影响，也受 citation 选择和生成影响。 |
| e2e `citation_precision` | 越高越好，但不能孤立看。系统如果引用更多 chunk，可能在检索变强时反而把这个值拉低。 |
| e2e `abstention_accuracy` | 越高越好，但要结合题型看。它既受拒答题影响，也会受“本该回答却误拒答”影响。 |

## 三阶段快速对比

| 指标 | Phase 0 | Phase 1 | Phase 2 |
|------|---------|---------|---------|
| retrieval `hit@5` | 0.0909 | 0.2727 | 0.2727 |
| retrieval `hit@10` | 0.0909 | 0.2727 | 0.2727 |
| retrieval `ndcg@5` | 0.0352 | 0.138 | 0.1602 |
| retrieval `ndcg@10` | 0.0519 | 0.138 | 0.1602 |
| retrieval `mrr` | 0.0455 | 0.0939 | 0.1212 |
| e2e `groundedness` | 0.1667 | 0.1667 | 0.1667 |
| e2e `citation_precision` | 0.1 | 0.05 | 0.05 |
| e2e `abstention_accuracy` | 0.3333 | 0.3333 | 0.4167 |

### 如何解读这张对比表

- `Phase 2` 主要改善体现在**检索排序质量**（`ndcg` / `mrr`）与**拒答准确率**，不是端到端所有指标同步上升。
- `groundedness` 三阶段持平，说明当前瓶颈已经不只是“能不能检到”，还包括**选证据**和**生成答案**。
- `citation_precision` 从 Phase 0 到 Phase 1/2 下降，不代表检索退化；更像是系统开始稳定引用知识库后，附带了更多非 gold citation。

---

## Phase 0（原始基线）

### 当次库内样例论文 ID

| `paper_key` | `paper_id` |
|-------------|------------|
| `berger2021_trns_tdcs_adhd` | 14 |
| `huo2022_eeg_biofeedback_adhd_tics` | 15 |

### 检索汇总（`retrieval.summary`）

| 指标 | 值 |
|------|-----|
| 参与统计题数 `count` | 11 |
| 无 gold 跳过 `skipped_without_gold` | 1 |
| `hit@1` | 0.0 |
| `hit@5` | 0.0909 |
| `hit@10` | 0.0909 |
| `ndcg@1` | 0.0 |
| `ndcg@5` | 0.0352 |
| `ndcg@10` | 0.0519 |
| `mrr` | 0.0455 |

### 检索分桶（`retrieval.breakdown`）

### 按题型 `by_category`

| category | `hit@10` | `mrr` | `count` |
|----------|----------|-------|---------|
| `single_fact` | 0.0 | 0.0 | 3 |
| `term_lookup` | 0.0 | 0.0 | 2 |
| `table_result` | 0.5 | 0.25 | 2 |
| `multi_span` | 0.0 | 0.0 | 1 |
| `summary` | 0.0 | 0.0 | 1 |
| `multi_turn` | 0.0 | 0.0 | 2 |

### 按语种 `by_language`

| language | `hit@10` | `mrr` | `count` |
|----------|----------|-------|---------|
| `en` | 0.125 | 0.0625 | 8 |
| `zh` | 0.0 | 0.0 | 3 |

### 端到端汇总（`end_to_end.summary`）

| 指标 | 值 |
|------|-----|
| 题数 `count` | 12 |
| `groundedness` | 0.1667 |
| `citation_precision` | 0.1 |
| `abstention_accuracy` | 0.3333 |

### 端到端分桶（`end_to_end.breakdown`）

### 按题型 `by_category`

| category | `groundedness` | `abstention_accuracy` | `count` |
|----------|----------------|----------------------|---------|
| `multi_turn` | 0.0 | 0.0 | 2 |
| `single_fact` | 0.0 | 0.0 | 3 |
| `term_lookup` | 0.0 | 0.5 | 2 |
| `table_result` | 0.5 | 0.5 | 2 |
| `multi_span` | 0.0 | 1.0 | 1 |
| `summary` | 0.0 | 0.0 | 1 |
| `abstention` | 1.0 | 1.0 | 1 |

### 按语种 `by_language`

| language | `groundedness` | `abstention_accuracy` | `count` |
|----------|----------------|----------------------|---------|
| `en` | 0.125 | 0.375 | 8 |
| `zh` | 0.0 | 0.0 | 3 |
| `mixed` | 1.0 | 1.0 | 1 |

---

## 调优后如何对比

1. 使用相同 `subset`（如 `smoke`）与 `judge_mode`（如 `heuristic`），避免指标口径变化。
2. 先看 `retrieval.summary`，再看 `end_to_end.summary`，最后再下钻到 `breakdown` 和逐题 JSON。
3. 如果只是 `hit@10` 上升，但 `groundedness` 不动，通常说明“检到了，但没被正确引用或生成出来”。
4. 如果 `citation_precision` 下降，不要立刻判定为回退；先看是否是 citation 变多、路由更积极，或拒答策略变化导致。
5. 原始报告中部分题目 `retrieved` 为空列表，属于**召回失败信号**；而 Phase 1 / 2 更值得关注的是“虽然有召回，但 gold 排名是否够前、答案是否真正用了对的证据”。

---

## Phase 1 参照（Hybrid Retrieval）

以下数字来自 Phase 1 完成后的 smoke 回归，便于后续 Phase 2+ 继续比对。

- **完整报告路径：** `backend/reports/sample-data-report-phase1-hybrid.json`
- **实现范围：** `PostgreSQL sparse + pgvector dense + RRF fusion + GLM rerank`

### 当次库内样例论文 ID

| `paper_key` | `paper_id` |
|-------------|------------|
| `berger2021_trns_tdcs_adhd` | 20 |
| `huo2022_eeg_biofeedback_adhd_tics` | 21 |

### 检索汇总（`retrieval.summary`）

| 指标 | 值 |
|------|-----|
| 参与统计题数 `count` | 11 |
| 无 gold 跳过 `skipped_without_gold` | 1 |
| `hit@1` | 0.0 |
| `hit@5` | 0.2727 |
| `hit@10` | 0.2727 |
| `ndcg@1` | 0.0 |
| `ndcg@5` | 0.138 |
| `ndcg@10` | 0.138 |
| `mrr` | 0.0939 |

### 检索分桶（`retrieval.breakdown`）

#### 按题型 `by_category`

| category | `hit@10` | `mrr` | `count` |
|----------|----------|-------|---------|
| `single_fact` | 0.3333 | 0.1111 | 3 |
| `term_lookup` | 0.5 | 0.1 | 2 |
| `table_result` | 0.5 | 0.25 | 2 |
| `multi_span` | 0.0 | 0.0 | 1 |
| `summary` | 0.0 | 0.0 | 1 |
| `multi_turn` | 0.0 | 0.0 | 2 |

#### 按语种 `by_language`

| language | `hit@10` | `mrr` | `count` |
|----------|----------|-------|---------|
| `en` | 0.0 | 0.0 | 8 |
| `zh` | 1.0 | 0.3444 | 3 |

### 端到端汇总（`end_to_end.summary`）

| 指标 | 值 |
|------|-----|
| 题数 `count` | 12 |
| `groundedness` | 0.1667 |
| `citation_precision` | 0.05 |
| `abstention_accuracy` | 0.3333 |

### 与原始基线的关键对比

| 指标 | 原始基线 | Phase 1 | 变化 |
|------|----------|---------|------|
| retrieval `hit@5` | 0.0909 | 0.2727 | +0.1818 |
| retrieval `hit@10` | 0.0909 | 0.2727 | +0.1818 |
| retrieval `ndcg@5` | 0.0352 | 0.138 | +0.1028 |
| retrieval `ndcg@10` | 0.0519 | 0.138 | +0.0861 |
| retrieval `mrr` | 0.0455 | 0.0939 | +0.0484 |
| e2e `groundedness` | 0.1667 | 0.1667 | 0.0 |
| e2e `citation_precision` | 0.1 | 0.05 | -0.05 |
| e2e `abstention_accuracy` | 0.3333 | 0.3333 | 0.0 |

### Phase 1 额外信号

- 原始基线里 `retrieved=[]` 的题数为 `5`；Phase 1 已降到 `0`。
- `term_lookup` 的 `hit@10` 从 `0.0` 提升到 `0.5`，`single_fact` 的 `hit@10` 从 `0.0` 提升到 `0.3333`。
- 当前提升主要体现在 `zh` 桶；`en` 桶在 Phase 1 报告中仍明显偏弱，后续 Phase 建议重点观察 rerank / fusion 是否对英文 exact-term 排序不稳定。

---

## Phase 2 参照（Structure-aware Chunking）

以下数字来自 Phase 2 完成后的 smoke 回归，便于继续和 Phase 0 / 1 做并排对照。

- **完整报告路径：** `backend/reports/sample-data-report-phase2.json`
- **实现范围：** `document preanalysis + structure-aware chunking + enriched chunk metadata`

### 当次库内样例论文 ID

| `paper_key` | `paper_id` |
|-------------|------------|
| `berger2021_trns_tdcs_adhd` | 22 |
| `huo2022_eeg_biofeedback_adhd_tics` | 23 |

### 检索汇总（`retrieval.summary`）

| 指标 | 值 |
|------|-----|
| 参与统计题数 `count` | 11 |
| 无 gold 跳过 `skipped_without_gold` | 1 |
| `hit@1` | 0.0 |
| `hit@5` | 0.2727 |
| `hit@10` | 0.2727 |
| `ndcg@1` | 0.0 |
| `ndcg@5` | 0.1602 |
| `ndcg@10` | 0.1602 |
| `mrr` | 0.1212 |

### 检索分桶（`retrieval.breakdown`）

#### 按题型 `by_category`

| category | `hit@10` | `mrr` | `count` |
|----------|----------|-------|---------|
| `single_fact` | 0.3333 | 0.1111 | 3 |
| `term_lookup` | 0.5 | 0.25 | 2 |
| `table_result` | 0.5 | 0.25 | 2 |
| `multi_span` | 0.0 | 0.0 | 1 |
| `summary` | 0.0 | 0.0 | 1 |
| `multi_turn` | 0.0 | 0.0 | 2 |

#### 按语种 `by_language`

| language | `hit@10` | `mrr` | `count` |
|----------|----------|-------|---------|
| `en` | 0.0 | 0.0 | 8 |
| `zh` | 1.0 | 0.4444 | 3 |

### 端到端汇总（`end_to_end.summary`）

| 指标 | 值 |
|------|-----|
| 题数 `count` | 12 |
| `groundedness` | 0.1667 |
| `citation_precision` | 0.05 |
| `abstention_accuracy` | 0.4167 |

### 端到端分桶（`end_to_end.breakdown`）

#### 按题型 `by_category`

| category | `groundedness` | `abstention_accuracy` | `count` |
|----------|----------------|----------------------|---------|
| `multi_turn` | 0.0 | 0.5 | 2 |
| `single_fact` | 0.3333 | 1.0 | 3 |
| `term_lookup` | 0.0 | 0.0 | 2 |
| `table_result` | 0.0 | 0.0 | 2 |
| `multi_span` | 0.0 | 0.0 | 1 |
| `summary` | 0.0 | 0.0 | 1 |
| `abstention` | 1.0 | 1.0 | 1 |

#### 按语种 `by_language`

| language | `groundedness` | `abstention_accuracy` | `count` |
|----------|----------------|----------------------|---------|
| `en` | 0.0 | 0.375 | 8 |
| `zh` | 0.3333 | 0.3333 | 3 |
| `mixed` | 1.0 | 1.0 | 1 |

### 与前两阶段的关键对比

| 指标 | Phase 0 | Phase 1 | Phase 2 |
|------|---------|---------|---------|
| retrieval `hit@5` | 0.0909 | 0.2727 | 0.2727 |
| retrieval `ndcg@5` | 0.0352 | 0.138 | 0.1602 |
| retrieval `ndcg@10` | 0.0519 | 0.138 | 0.1602 |
| retrieval `mrr` | 0.0455 | 0.0939 | 0.1212 |
| e2e `groundedness` | 0.1667 | 0.1667 | 0.1667 |
| e2e `citation_precision` | 0.1 | 0.05 | 0.05 |
| e2e `abstention_accuracy` | 0.3333 | 0.3333 | 0.4167 |

### Phase 2 额外信号

- 相比 Phase 1，`hit@10` 没再提升，但 `ndcg@5` / `ndcg@10` / `mrr` 继续上升，说明 Phase 2 主要改善的是**正确证据排序位置**。
- `zh` 检索桶继续增强，但 `en` 检索桶仍然偏弱，说明结构化切块没有解决英文 exact-term / 表格项召回排序问题。
- `abstention_accuracy` 从 `0.3333` 提升到 `0.4167`，但 `groundedness` 仍持平，说明下一步优化重点应放在**citation 选择、表格证据利用、回答生成**，而不只是 chunking 本身。
