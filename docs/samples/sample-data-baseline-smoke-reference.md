# Sample-data 评测基线（原始参照）

本文档从当次完整报告抽取关键数字，便于后续调优后**同一配置**下对比。完整逐题明细仍在 JSON 中。

- **完整报告路径：** `backend/reports/sample-data-report.json`

---

## 评测配置

| 字段 | 值 |
|------|-----|
| `mode` | `both`（检索 + 端到端） |
| `subset` | `smoke`（12 题端到端；含 gold 的检索题为 11，另 1 题跳过 gold） |
| `judge_mode` | `heuristic`（启发式判分，非 LLM 裁判） |

---

## 当次库内样例论文 ID

| `paper_key` | `paper_id` |
|-------------|------------|
| `berger2021_trns_tdcs_adhd` | 14 |
| `huo2022_eeg_biofeedback_adhd_tics` | 15 |

---

## 检索汇总（`retrieval.summary`）

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

---

## 检索分桶（`retrieval.breakdown`）

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

---

## 端到端汇总（`end_to_end.summary`）

| 指标 | 值 |
|------|-----|
| 题数 `count` | 12 |
| `groundedness` | 0.1667 |
| `citation_precision` | 0.1 |
| `abstention_accuracy` | 0.3333 |

---

## 端到端分桶（`end_to_end.breakdown`）

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
2. 将新报告仍输出到 `backend/reports/`，与本 JSON / 本表对照；关注 `retrieval.summary` 与 `end_to_end.summary` 是否同步改善。
3. 原始报告中部分题目 `retrieved` 为空列表，属于**需单独排查的信号**（与排序好坏不同）；逐题核对见 JSON 内 `retrieval.questions`。
