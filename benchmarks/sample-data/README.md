# Sample-Data Benchmark

`benchmarks/sample-data/` 提供两篇样本文献的最小评测集与验收闭环数据。

## Files
- `papers.json`: 样本文献定义与稳定 `paper_key`
- `questions.jsonl`: 96 条题目与 gold 标注
- `sessions.jsonl`: 4 组多轮追问
- `smoke_question_ids.json`: 12 条 smoke 子集

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

## Commands
在 `backend/` 目录运行：

```bash
python -m scripts.sample_data_eval --mode retrieval --subset smoke
python -m scripts.sample_data_eval --mode both --subset full --output reports/sample-data-report.json
```
