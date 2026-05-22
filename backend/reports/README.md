# Benchmark Reports

本目录存放 `sample_data_eval` 生成的 JSON 报告，**不提交 git**（见仓库根目录 `.gitignore`）。

## 生成方式

在 `backend/` 目录运行，例如：

```bash
python -m scripts.sample_data_eval --mode both --subset smoke --output reports/smoke.json
python -m scripts.sample_data_eval --mode both --subset full --output reports/full-benchmark-20260518.json
python -m scripts.sample_data_eval --mode retrieval --subset full --output reports/full-retrieval-20260519.json
```

只传文件名时（不含目录），脚本会自动写入本目录。

## 命名建议

| 场景 | 示例 |
|------|------|
| smoke 快速验证 | `smoke-after-h2.json` |
| full 全量 both | `full-benchmark-YYYYMMDD.json` |
| full 仅 retrieval | `full-retrieval-YYYYMMDD.json` |
| 阶段检查点 | `phase4-smoke-benchmark.json` |
