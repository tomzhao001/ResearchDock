# OCR 优化 Roadmap（2026-05-19）

## 目的

本文档用于记录当前 Docling OCR 管线在中文论文场景下的主要问题、已验证结论，以及后续推荐的优化方向，供后续实现时直接参考。

本文档关注两个核心问题：

1. 如何识别 PDF 自带 text layer 是否“不可信”，并在合适的时候切换到全页 OCR。
2. 当全页 OCR 已经开启后，如何进一步降低纯 OCR 的识别误差。

## 当前背景

当前系统使用 Docling 作为唯一 PDF 文档解析管线，并将解析结果写入：

- `PaperAsset.raw_text`
- `paper_document_blocks`
- `paper_document_tables`
- `paper_document_pictures`

当前运行配置中，Docling OCR 主要通过 `EasyOCR` 执行。

近期排查发现，部分中文论文 PDF 在默认模式下会出现以下问题：

- `raw_text`、`blocks`、`tables` 都有值，但文本整体是乱码。
- 乱码形态不是普通 OCR 错字，而是大量 ASCII 碎片、符号串、单字符断裂。
- 同一份 PDF 用 PyMuPDF 直接抽 text layer 时，可能是正常中文，也可能表现为“带毒 text layer”；Docling 默认行为并不保证一定会彻底替换该层文本。

## 已验证结论

针对一份实际出问题的中文论文 PDF，已经做过如下对比实验：

- 同一份 PDF
- 同一套 Docling + EasyOCR 配置
- 唯一变量：`force_full_page_ocr = False / True`

实验结论如下：

### 1. `force_full_page_ocr = False`

Docling 输出的 `markdown`、`blocks`、`tables` 以 ASCII 垃圾串为主，例如：

- `! !" :; !`
- `\ Y# o j pqr ...`
- 表格头部被解析为符号和碎片字母

这类输出不是“少量错字”，而是整篇正文不可用。

### 2. `force_full_page_ocr = True`

同一份 PDF 在强制全页 OCR 后：

- 正文恢复为可读中文
- 表格标题、表头、正文段落恢复为可理解文本
- 仍存在少量 OCR 误识别，但已经不是整篇乱码

这说明：

- 默认模式下，Docling 可能仍沿用了不可信的 embedded text layer。
- 对于这类“text layer 有毒”的中文 PDF，`force_full_page_ocr=true` 是有效的补救手段。

## 当前已经具备的基础能力

当前仓库中已经补入一个全局环境变量：

- `DOCLING_FORCE_FULL_PAGE_OCR`

用途：

- `false`：保持 Docling 默认策略
- `true`：强制整页 OCR，覆盖 PDF 原有文字层

该能力适合在已知 PDF 来源质量较差、中文 text layer 经常出问题时直接启用。

但长期来看，不建议对所有 PDF 都一刀切地强制全页 OCR。

## 当前实现状态（更新）

目前代码已经开始按分层方式落地，主模块位置如下：

- `backend/app/services/paper_pipeline/workflow.py`
  - `pdf_ingest` 图中新增 `assess_ocr_strategy`、`postprocess_ocr_text` 与 `glm_ocr_escalation` 节点
- `backend/app/services/ocr/quality.py`
  - 基于抽样页 text layer 的轻量质量判定与 routing decision
- `backend/app/services/ocr/postprocess.py`
  - 统一串联文本归一化、SymSpell、medical wordlist、规则修复
- `backend/app/services/ocr/escalation.py`
  - 在本地后处理之后，对仍然可疑的 block 触发 OCR escalation
- `backend/app/services/ocr/glm_escalation.py`
  - GLM-OCR API 适配层
- `backend/app/services/ocr/spell_correction.py`
  - SymSpell 与医学术语词典纠错封装
- `backend/app/services/ocr/rules.py`
  - 单位、缩写、`±` 等低风险规则修复

当前仍保留 `Docling` 作为唯一 PDF 结构化主入口；`RapidOCR` 继续作为 Docling 内默认 OCR backend，本地后处理负责低成本修复，而 `GLM-OCR` 仅作为后置 escalation provider 处理少量疑难块。

## 为什么不建议所有 PDF 都强制全页 OCR

对于 born-digital PDF（尤其是正常导出的数字论文），原生 text layer 往往在以下方面优于 OCR：

- DOI、年份、剂量、数值
- 英文缩写
- 表格数字精度
- 专有名词拼写
- 可重复性和处理成本

如果对所有 PDF 都开启全页 OCR，会带来以下副作用：

- 正常 PDF 的数字和缩写可能反而变差
- 表格中的数值、单位、标点更容易被 OCR 误读
- 处理速度下降
- 模型下载、缓存、资源消耗更高

因此，更合理的方向是：

1. 先识别 text layer 是否可信
2. 对明显异常的 PDF 或页面再切 OCR
3. 对纯 OCR 结果再做轻量修复

## 总体优化思路

推荐将未来优化拆成四层，而不是一次性做重构。

### P0：保留全局兜底开关

目的：

- 先给线上一个可立即使用的止血手段

建议：

- 保留 `DOCLING_FORCE_FULL_PAGE_OCR`
- 默认值保持 `false`
- 当某一批 PDF 来源已知 text layer 普遍不可信时，可以运营或部署层面手动切到 `true`

说明：

- 这一层已经完成，不是最终形态，但非常有价值

### P1：增加“轻量判定 + 自动切换”能力

目的：

- 避免所有 PDF 都强制纯 OCR
- 自动识别“有毒 text layer”

这是 ROI 最高、最值得优先落地的一层。

### P2：增加纯 OCR 后处理

目的：

- 解决 `force_full_page_ocr=true` 后仍存在的数字、英文缩写、术语误识别

### P3：逐步演进到页级混合策略

目的：

- 正常页保留 text layer
- 异常页单独 OCR
- 降低整篇纯 OCR 的精度损失

### P4：仅对极难区域引入 VLM 精修

目的：

- 对少数复杂页面、复杂表格、图文混排区域补一层更强但更贵的能力

## 推荐的自动判定策略

### 核心原则

判断目标不是“这页看起来奇怪”，而是：

**这份 PDF 的 embedded text layer 是否足够可信，可以直接进入结构化解析。**

### 建议的输入来源

建议先基于轻量 text layer 抽取做预判，而不是一上来就整篇双跑。

优先做法：

1. 先用 PDF 原生文本抽取能力抽前 `3` 到 `5` 页正文
2. 对每页计算若干质量指标
3. 先做页级判定
4. 再做文档级决策

### 推荐指标

对每页文本计算以下指标：

- `char_count`：非空白字符数
- `cjk_ratio`：中文字符占比
- `ascii_printable_ratio`：ASCII 可打印字符占比
- `replacement_char_count`：替代字符数量，例如 `�`
- `single_char_token_ratio`：单字符 token 比例
- `suspicious_run_count`：疑似乱码片段数量
- `line_break_density`：换行密度
- `alnum_fragment_ratio`：英数碎片比例

建议额外做两个辅助信号：

- `contains_keywords_page_markers`：是否命中“摘要 / 关键词 / 结果 / 讨论 / 参考文献”等典型页标记
- `contains_structured_numeric_evidence`：是否出现 DOI、年份、数值列、剂量、P 值等结构化信息

### 页级分类

推荐把页面先分成三类：

- `good_text_layer`
- `no_text_layer`
- `toxic_text_layer`

#### `no_text_layer`

可用初始规则：

- `char_count < 30`

说明：

- 基本没有可用文字层
- 通常是扫描页或图片页

#### `toxic_text_layer`

可用初始规则：

满足以下任意两条即可判为可疑：

- `char_count > 80` 且 `cjk_ratio < 0.20`
- `ascii_printable_ratio > 0.55`
- `replacement_char_count` 明显偏高
- `single_char_token_ratio` 明显偏高
- 标题、摘要页缺乏连续中文短语
- 命中大量可疑片段，例如连续符号串、单字符碎片串

额外建议：

- 对中文论文，如果标题页、摘要页、关键词页出现大段 ASCII 碎片，应提高该页权重

#### `good_text_layer`

除以上两类之外，可暂时认为可用。

### 文档级决策

建议按下面逻辑做文档级路由。

#### 保持默认模式

满足以下条件时，继续默认模式：

- 前 `3` 页中没有 `toxic_text_layer`
- 且标题/摘要页正常

#### 页级 OCR（中期目标）

满足以下条件时，未来可走页级 OCR：

- 只有少量离散异常页
- 可疑页占比低于 `30%`
- 主要问题集中在扫描页、附录页、图页或表格页

#### 整篇切换到全页 OCR

满足以下任意条件时，建议整篇切到 `force_full_page_ocr=true`：

- 前 `3` 页中有 `2` 页及以上是 `toxic_text_layer`
- 标题页或摘要页明显乱码
- 抽样页中可疑页占比达到或超过 `40%`
- 文本字符数很多，但几乎不构成正常中文自然语言

## 推荐增加一个“试探页 OCR”步骤

如果希望进一步降低误判，可以在 P1 中增加一个 canary 机制。

### 思路

对判为 `toxic_text_layer` 的页面，不立刻整篇切换，而是：

1. 先选一页关键页（优先标题页 / 摘要页 / 第一页正文）
2. 对这一页临时做一次 OCR
3. 比较 text layer 与 OCR 输出质量
4. 如果 OCR 输出显著更像正常中文，再升级到整篇 OCR

### 可以比较的信号

- OCR 结果的中文字符数显著提升
- OCR 结果包含连续中文短语
- OCR 结果恢复了标题、摘要、表头等结构
- OCR 结果中 ASCII 垃圾串显著下降

### 价值

这一步可以避免规则误判导致的“好 PDF 被错误切成纯 OCR”。

## 如何降低纯 OCR 的识别误差

`force_full_page_ocr=true` 解决的是“整篇乱码”的问题，但它不能解决所有文本质量问题。

在当前中文医学论文场景下，推荐优先做以下三类后处理，而不是直接把整篇任务交给更贵的视觉大模型。

### 1. 基础文本归一化

适合做成通用后处理步骤。

建议处理：

- 全角半角统一
- 异常空格清理
- 连续单字符拼接修复
- 常见标点归一化
- OCR 造成的断词修复

这部分可以沿用现有 normalization 思路继续扩展，但要避免过度“清洗”导致误伤。

### 2. 结构化字段定向修复

建议优先针对高价值字段做规则修复：

- DOI
- PMID
- 年份
- 百分比
- 剂量与单位
- P 值
- 样本量
- 量表名
- 药物名

常见混淆包括：

- `l / I / 1`
- `0 / O`
- `5 / S`
- `8 / B`
- `rn / m`

建议：

- 对命中结构化模式的位置单独做纠错，不要整篇盲目替换

### 3. 表格区域专项修复

表格往往是纯 OCR 最容易受损的部分，同时也是最影响检索质量的部分。

建议对表格额外做：

- 表头词表匹配
- 数值单元格合法性检查
- 单位列纠错
- 表格 caption 与内容关联修复
- 行列对齐后的数值合理性校验

说明：

- 如果未来要做表格专项提升，优先级应高于正文风格润色

## 为什么不建议直接“全篇换成 VLM OCR”

对于当前目标，VLM 不是第一优先级。

原因：

- 成本更高
- 吞吐更低
- 对数字、表格数值、缩写并不一定比传统 OCR 更稳
- 难以解释和修复

VLM 更适合做以下补充场景：

- 图文混排特别复杂的页面
- OCR 多次失败的页面
- 复杂图表、流程图、图片说明
- 需要语义描述而不仅是字符识别的区域

因此，更推荐：

- 先用 text layer / OCR / 规则修复解决 80% 问题
- 再把 VLM 作为“难例精修”能力

## 推荐实施顺序

### 第一阶段：最小可用自动路由

目标：

- 不再手动判断哪些 PDF 应切纯 OCR

建议实现：

1. 新增一个轻量质量判定模块
2. 对前 `3` 到 `5` 页做 text layer 抽样
3. 产生页级标签与文档级决策
4. 将决策结果写入 `asset.metadata_json["extraction"]`
5. 自动决定是否启用 `force_full_page_ocr`

交付标准：

- 至少能识别“整篇乱码型 PDF”
- 不明显误伤正常 born-digital PDF

### 第二阶段：加 canary OCR

目标：

- 降低误判率

建议实现：

1. 对可疑文档先跑一页试探 OCR
2. 比较 text layer 与 OCR 的质量差异
3. 再决定是否整篇切换

交付标准：

- 对边界案例更稳

### 第三阶段：加纯 OCR 后处理

目标：

- 降低全页 OCR 后的数字、术语、表格误差

建议实现：

1. 通用文本归一化
2. 结构化字段修复
3. 表格专项修复

交付标准：

- DOI、年份、剂量、样本量、P 值误差率下降

### 第四阶段：页级混合管线

目标：

- 不整篇强制 OCR
- 仅对异常页 OCR

建议实现：

1. page-level route
2. normal page 使用原生 text layer
3. bad page 使用 OCR
4. 合并为统一结构输出

交付标准：

- 正常页保住 text fidelity
- 异常页恢复可读性

## 推荐的代码落点

以下仅为建议，不要求一次性实现。

### 1. 新增文档质量判定模块

建议新增类似模块：

- `backend/app/services/document_quality.py`

职责：

- 轻量抽样
- 页面指标计算
- 页级判定
- 文档级路由决策

### 2. 在 Docling 提取前插入路由

建议在文档解析主流程中：

- 先做 text layer 质量判定
- 再决定是否设置 `force_full_page_ocr`

如果后续需要保留人工覆盖，则优先级建议为：

1. 显式环境变量强制值
2. 自动路由决策
3. 默认配置

### 3. 在元数据中保留判定证据

建议写入：

- `text_layer_quality`
- `sampled_pages`
- `page_classifications`
- `routing_decision`
- `routing_reason`
- `force_full_page_ocr_applied`

价值：

- 方便排查
- 方便前端和 Benchmark 观察
- 方便后续回溯误判

## Benchmark 与验证建议

后续如果开始实现，建议至少做以下验证。

### 样本集划分

至少准备三类样本：

1. 正常 born-digital PDF
2. text layer 有毒的中文论文 PDF
3. 扫描件 / 图片型 PDF

### 指标建议

至少观测：

- `auto_route_accuracy`
- `false_positive_rate`
- `false_negative_rate`
- `ocr_forced_ratio`
- `block/table 可读率`
- DOI / 数值 / 英文缩写准确率

### 人工抽检重点

重点人工抽检：

- 标题
- 摘要
- 关键词
- 第一张表
- DOI
- 样本量
- 量表名

## 风险与注意事项

### 1. 参考文献页容易误判

参考文献页天然：

- 中文少
- 英文多
- 标点多
- DOI 多

因此不能仅凭“中文比例低”判整篇有毒。

### 2. 英文摘要页不能按中文正文标准判断

英文摘要页也可能导致误报，必须结合页位置和上下文判断。

### 3. 纯 OCR 后不要做过强的全文替换

尤其是：

- DOI
- 数值
- 单位
- 缩写

应优先做结构化位置修复，而不是全文 regex 暴力替换。

### 4. 不能过早引入过重模型

如果在 P1 还没做好前就直接上全篇 VLM，容易让系统复杂度和成本快速上升，但不一定解决主要问题。

## 当前推荐结论

结合当前已经验证的现象，推荐的未来优化路线为：

1. 保留全局 `DOCLING_FORCE_FULL_PAGE_OCR` 兜底开关
2. 优先实现“轻量判定 + 自动切换”
3. 再补“纯 OCR 后处理”
4. 最后演进到“页级混合管线”
5. VLM 仅作为难例精修，而不是默认整篇方案

如果只允许做一件事，优先级最高的是：

**先实现 text layer 质量判定与自动路由。**

这一步既能解决“整篇乱码型 PDF”，也不会像全局强制 OCR 那样伤害所有正常文档。
