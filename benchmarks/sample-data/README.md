# Sample-Data Benchmark

`benchmarks/sample-data/` 提供两篇样本文献的最小评测集与验收闭环数据。

## Files
- `papers.json`: 样本文献定义与稳定 `paper_key`
- `questions.jsonl`: 107 条题目与 gold 标注
- `sessions.jsonl`: 5 组多轮追问
- `smoke_question_ids.json`: 14 条 smoke 子集

## Question Fields
- `q_id`: 稳定问题 ID
- `question`: 用户问题
- `category`: `single_fact` / `term_lookup` / `table_result` / `multi_span` / `summary` / `multi_turn` / `abstention`
- `difficulty`: `easy` / `medium` / `hard`
- `language`: 题目归属的数据语种，`en`/`zh`/`mixed`
- `requires_multi_span`: 是否需要跨多个证据段整合
- `needs_table_or_figure`: 是否依赖表格或图示附近内容
- `allow_fallback_general`: 是否允许最终答案走 `fallback_general`
- `expected_abstention`: 是否应当拒答
- `gold_evidence`: gold 证据片段列表，每条包含 `paper_key` 和 `snippet`
- `expected_keywords`: e2e 评测时用于启发式打分的关键词
- `keyword_hit_threshold`: 命中多少个关键词才算答案覆盖到位
- `session_id` / `turn_index`: 多轮问题所属会话与轮次

## Gold Resolution
首版 benchmark 不直接把数据库中的 `paper_id`、`chunk_id` 写死，而是通过：

1. `paper_key`
2. `gold_evidence[].snippet`

在当前索引结果里动态解析 gold chunk。这样 sample-data benchmark 可以在不依赖固定主键的前提下反复重跑。

## Annotation Notes

- `expected_keywords` 应尽量由 `gold_evidence` 直接支持；如果关键词是合理推断、同义表达或表格/图示锚点，`gold_evidence` 应保留足够上下文。
- `gold_evidence_satisfy="any"` 表示命中任一 gold evidence 即可满足该题的证据要求；默认需要覆盖全部 gold evidence。
- 拒答题允许 `gold_evidence=[]`，但题干应限定为当前两篇样本文献确实没有明确答案的问题。
- 对于 `requires_multi_span=true` 的题，`gold_evidence` 通常应包含多个证据片段；若其中一条只是表格、图示或章节锚点，应确保另一条包含可直接作答的信息。

## Commands
在 `backend/` 目录运行：

```bash
python -m scripts.sample_data_eval --mode retrieval --subset smoke
python -m scripts.sample_data_eval --mode both --subset full --output reports/sample-data-report.json
```

### 报告输出路径

- 所有 benchmark 报告写入 `backend/reports/`（已在 `.gitignore` 中忽略）。
- **不要**将 `--output` 指向 `backend/` 根目录；若只传文件名（如 `--output smoke.json`），脚本会自动写入 `reports/smoke.json`。
- 建议命名：`reports/smoke-<描述>.json`、`reports/full-benchmark-<YYYYMMDD>.json`、`reports/full-retrieval-<YYYYMMDD>.json`。
