# ResearchDock OCR 流程与增强方案现状说明

## 文档目的

本文档总结 ResearchDock 当前仓库里已经落地的 OCR 处理流程，重点说明：

- Docling 作为主解析框架时，项目额外做了哪些增强；
- OCR 前置如何判断 PDF 的 text layer 是否可信；
- OCR 后置如何做文本清洗、术语纠错、规则修复；
- 如何通过大模型对可疑区域做托底；
- 最终结构化结果如何落库、进入检索与后续摘要流程。

本文档描述的是“当前实现状态”，不是纯 roadmap。对于尚未落地的中长期方向，会在最后单独标注。

## 一句话结论

当前项目的 OCR 主链路不是“直接把 PDF 丢给 Docling 就结束”，而是一个分层流程：

1. 先用 `PyMuPDF` 对 PDF 前几页 text layer 做轻量体检；
2. 决定是否让 Docling 强制走全页 OCR；
3. 由 `Docling + RapidOCR` 产出结构化文档；
4. 对正文、表格、caption 等文本做本地后处理；
5. 只对命中可疑条件的 block 做区域级 `GLM-OCR` 升级识别；
6. 将页、块、表格、图片等结构化结果落库，并重建索引；
7. 在 OCR 结果稳定后，再触发论文摘要与问答题集提取。

也就是说，Docling 仍然是主入口，但项目已经围绕它补上了“前置判断 + 后置修复 + 大模型托底 + 元数据留痕”这几层能力。

## 整体执行流程

### 1. 任务入口

用户上传 PDF 后，后端会创建 `pdf_ingest` 任务，由 Celery worker 异步处理。当前 OCR 主流程位于 `backend/app/services/paper_pipeline/workflow.py` 的 `PdfIngestGraphRunner`。

主流程节点顺序如下：

1. `load_context`
2. `mark_processing`
3. `assess_ocr_strategy`
4. `extract_document`
5. `postprocess_ocr_text`
6. `glm_ocr_escalation`
7. `describe_pictures`
8. `persist_document_structure`
9. `rebuild_index`
10. `complete_job`

其中真正与 OCR 质量直接相关的核心增强是：

- `assess_ocr_strategy`：前置路由判定
- `extract_document`：带策略的 Docling 提取
- `postprocess_ocr_text`：本地低成本修复
- `glm_ocr_escalation`：可疑区域的大模型托底

### 2. 当前 OCR 主栈

当前默认栈可以概括为：

`Docling + RapidOCR -> 文本归一化 -> SymSpell/medical wordlist -> 自定义规则修复 -> GLM-OCR escalation`

这条链路的设计思路是：

- 结构化与版面解析继续交给 Docling；
- 常规 OCR 识别继续走本地 OCR；
- 低成本、可解释的问题尽量在本地修；
- 只有命中可疑条件的 block 才升级到远程模型。

## 第一层：前置路由与质量判定

### 为什么要加这一层

项目里已经明确遇到过一种典型问题：PDF 自带 text layer，但是这个 text layer 是“有毒”的。  
这种 PDF 在默认模式下可能会让 Docling 读到大量乱码、ASCII 碎片、断裂字符，而不是正常中文。

因此项目没有简单地对所有 PDF 一刀切做全页 OCR，而是先做一次轻量判断，再决定是否强制整篇 OCR。

### 实现位置

前置判定逻辑在：

- `backend/app/services/ocr/quality.py`

工作流里对应节点是：

- `assess_ocr_strategy`

### 判定输入

系统会先用 `PyMuPDF` 打开 PDF，并抽样前几页的原生 text layer。默认抽样页数由配置项控制：

- `OCR_QUALITY_ROUTING_ENABLED=true`
- `OCR_QUALITY_SAMPLE_PAGES=3`

也就是说，当前实现不是先跑两遍完整 OCR 再比较，而是先做一个低成本预判。

### 页面级质量指标

对每个抽样页，当前实现会计算以下信号：

- `char_count`：非空白字符数
- `cjk_ratio`：中文字符占比
- `ascii_printable_ratio`：ASCII 可打印字符占比
- `replacement_char_count`：替代字符 `�` 数量
- `single_char_token_ratio`：单字符 token 比例
- `suspicious_run_count`：连续短碎片串数量
- `line_break_density`：换行密度
- `alnum_fragment_ratio`：短英数碎片比例
- `contains_keywords_page_markers`：是否出现 `摘要`、`关键词`、`abstract`、`results` 等页标记
- `contains_structured_numeric_evidence`：是否出现 DOI、年份、剂量、P 值等结构化数字证据

需要注意的是，并不是所有采集到的指标都会直接进入当前版本的路由判定。  
例如 `line_break_density` 和 `contains_structured_numeric_evidence` 目前更偏向留痕和观测信号，当前主判定规则真正直接使用的还是 `char_count`、`cjk_ratio`、`ascii_printable_ratio`、替代字符、短碎片比例等核心特征。

### 页面分类

每个采样页会被归类为：

- `good_text_layer`
- `no_text_layer`
- `toxic_text_layer`

当前规则大意如下：

#### `no_text_layer`

如果页面有效字符过少，例如：

- `char_count < 30`

则认为该页几乎没有可用 text layer。

#### `toxic_text_layer`

如果页面字符很多，但同时出现多项可疑特征，例如：

- 中文占比很低；
- ASCII 碎片比例很高；
- 有替代字符；
- 单字符 token 比例过高；
- 出现大量短碎片串；
- 英数字碎片比例高；

则判为 `toxic_text_layer`。

此外，前两页如果缺少常见标题/摘要页标记，同时中文占比又异常低，也会被提高警惕。

### 文档级决策

页面判断完成后，会汇总成一个 `OcrRoutingDecision`，核心字段包括：

- 是否强制 `force_full_page_ocr`
- Docling 应使用的 OCR engine
- 是否开启 postprocess
- 是否开启 escalation
- 判定来源 `source`
- 判定原因 `routing_reason`
- 抽样页数
- toxic 页数量
- 每页 assessment 明细

当前文档级触发强制全页 OCR 的条件主要有三类：

- 抽样页里至少有 2 页是 `toxic_text_layer`
- `toxic` 页比例达到或超过 40%
- 前两页（标题/摘要高权重页）出现 `toxic_text_layer`

### 手动兜底与自动路由的关系

项目仍保留了一个人工总开关：

- `DOCLING_FORCE_FULL_PAGE_OCR`

当前行为更准确地说是：

1. 如果质量路由开启，系统仍会先尝试做抽样评估；
2. 但只要 `DOCLING_FORCE_FULL_PAGE_OCR=true`，最终 routing decision 会被手动值覆盖，整篇强制全页 OCR；
3. 如果质量路由关闭、`PyMuPDF` 不可用或 PDF 打不开，则直接回退到默认策略。

这意味着项目既支持线上人工止血，也支持自动识别“有毒 text layer”。

## 第二层：Docling 提取阶段的项目增强

### 1. Docling 仍然是唯一结构化主入口

当前 `DOCUMENT_EXTRACTOR=docling`，项目没有把 PDF 结构化链路拆成多个主引擎竞争，而是让 Docling 继续负责：

- 页面提取
- block 提取
- 表格结构提取
- 图片对象提取
- markdown 导出

### 2. OCR backend 做了可控封装

项目没有把 Docling 当作完全黑盒来用，而是对其 OCR backend 进行了封装和可配置化。

当前支持：

- `rapidocr`（默认）
- `easyocr`
- `tesserocr`
- `tesseract`

其中默认推荐是 `rapidocr`。

### 3. RapidOCR 做了额外工程化处理

在 `backend/app/services/docling_extraction.py` 里，项目额外补了这些能力：

- 根据语言配置选择当前更偏中文或英文的识别模型参数；
- 自动构建 RapidOCR v5 所需模型清单；
- 在本地模型缓存目录中检查缺失 artifact；
- 缺失时自动下载并写入缓存；
- 将缓存路径注入 Docling pipeline；
- 支持 `force_full_page_ocr` 随路由策略动态打开。

这部分不是 Docling 默认业务流程本身，而是项目为了让 OCR backend 更稳定可部署做的工程增强。

### 4. 不只拿 markdown，还会抽出结构化上下文

Docling 提取后，项目不会只保留一份平铺文本，而是继续抽取并补齐结构化上下文：

- `pages`
- `blocks`
- `tables`
- `pictures`
- `docling_json`

并为 block/table/picture 补充：

- `page_number`
- `section_path`
- `heading_level`
- `reading_order`
- `bbox`
- `provenance`

其中 `section_path`、`reading_order`、`bbox` 这些字段非常关键，因为它们直接支撑了后续：

- 结构化落库；
- 检索重建；
- 区域级 OCR escalation；
- 图片描述与上下文关联。

### 5. 表格和图片不是孤立提取，而是做了上下文对齐

项目会将 Docling 的 table/picture 对象与遍历得到的上下文行进行匹配，依据主要是：

- 页码
- bbox 距离
- block 类型偏好

这样做的目的，是把表格、图片重新挂回更接近正文结构的位置，保住：

- `section_path`
- `heading_level`
- `reading_order`
- `provenance`

这一步对于后续问答与检索很重要，因为系统不是只需要“识别出字”，还需要知道这些字属于哪一节、哪张表、哪张图。

## 第三层：本地后处理与规则修复

### 为什么要先做本地后处理

即便已经切到全页 OCR，常见问题仍然存在：

- 全角半角混乱
- 单位与数字之间被错误插空格
- `土` / `士` 被识别成 `±`
- `I / l / O / o / 1 / 0` 混淆
- 医学术语或缩写被识别错

这些问题如果全部交给大模型处理，成本高、可解释性弱，也容易误改原文。  
所以项目先用本地、可控、低成本的后处理解决大部分低风险问题。

### 实现位置

- `backend/app/services/ocr/postprocess.py`
- `backend/app/services/ocr/text_normalization.py`
- `backend/app/services/ocr/spell_correction.py`
- `backend/app/services/ocr/medical_terms.py`
- `backend/app/services/ocr/rules.py`

### 处理范围

后处理不是只修正文 block，而是会覆盖：

- `document.markdown_text`
- 每个 `block.text`
- 表格 `caption`
- 表格 `markdown`
- 表格 `data` 中的字符串单元格

代码里虽然也会遍历 `document.pages`，但当前 `DoclingDocumentExtractor._extract_pages()` 主要填充的是页码、尺寸和 page metadata，并没有真正把页面文本灌入 `page.text`。  
因此，当前真正有文本内容并稳定参与后处理的，主要还是 `markdown_text`、`block.text` 以及表格相关文本。

这说明项目在设计时已经把“表格 OCR 质量”纳入主流程，而不是只关心正文。

### 1. 文本归一化

当前归一化主要做的是全角 ASCII 折叠，包括：

- 全角英文转半角
- 全角数字转半角
- 全角标点转半角
- 全角空格转普通空格

这一步的目标不是“美化文本”，而是先把 OCR 常见的字符形态差异压平，方便后续规则修复和词典纠错。

### 2. SymSpell + 医学词表纠错

项目引入了 `symspellpy`，并配合医学词表做术语纠错。

词表来源有两层：

1. 默认使用仓库自带的 `backend/resources/medical-wordlist/research-core.txt`
2. 如果配置了 `OCR_MEDICAL_WORDLIST_PATH`，则可以加载外部目录或文件下的全部 `.txt` 词表

纠错逻辑不是简单的全量替换，而是分层进行：

- 如果 token 本身已是已知医学术语，则直接保留；
- 优先走 SymSpell 近似匹配；
- 再尝试基于常见 OCR 混淆签名匹配；
- 最后再用 `difflib` 做 fallback。

这里的“混淆签名”专门针对 OCR 常见误识别做了处理，例如：

- `I / L -> 1`
- `O -> 0`
- `II -> H`

这使它特别适合修正医学缩写和药名类 token。

### 3. 自定义低风险规则修复

项目额外加了一批规则修复，优先修那些结构比较清晰、误伤风险较低的问题。

当前已实现的典型规则包括：

- 将 `3.0土4.0`、`3.0士4.0` 修复为 `3.0±4.0`
- 将 `10 g` 归一为 `10g`
- 将 `I0g`、`l0g`、`O5%` 这类数字+单位 token 修复为更合理的数字形式

这些规则的共同特点是：

- 作用范围比较窄；
- 可以通过正则精确命中；
- 可解释性强；
- 风险远低于全文自由改写。

### 4. 后处理会留下统计信息

后处理完成后，会在文档 metadata 中记录：

- 是否启用 postprocess
- 归一化次数
- SymSpell 修正次数
- 纠错总次数（当前实现中同时写入 `symspell_correction_count` 和 `medical_term_correction_count`）
- 规则修复次数

需要注意的是，当前统计还没有精确拆分“到底是 SymSpell 命中、医学词表命中、混淆签名命中还是 difflib 命中”。  
也就是说，这里更适合理解为“纠错阶段发生了多少次 token 级修正”，而不是来源已经被精细归因。

这些统计会进一步写入 `asset.metadata_json["extraction"]["ocr_postprocess"]`，方便后续排查效果。

## 第四层：大模型托底，但只托底命中可疑条件的块

### 核心原则

当前项目并没有把整个 OCR 任务直接改成“全篇 VLM OCR”，而是只把大模型当成 escalation provider。

也就是说：

- 常规块仍然沿用 Docling + 本地后处理结果；
- 只有命中可疑条件的 block 才会升级给 `GLM-OCR`；
- 而且升级单位不是整页、整篇，而是“PDF 页面中的一个 bbox 区域”。

这是当前实现里最重要的成本控制点之一。

### 实现位置

- `backend/app/services/ocr/escalation.py`
- `backend/app/services/ocr/glm_escalation.py`
- `backend/app/services/ocr/pdf_region.py`

### 触发条件

一个 block 要进入 escalation，至少要满足：

- 有 `page_number`
- 有 `bbox`
- 文本通过 `looks_suspicious_ocr_text()` 判断为可疑

可疑文本判断主要关注：

- 是否存在大量短碎片串
- 短 token 比例是否过高

这说明项目目前的托底是“精确打点”式，而不是模糊地整页重跑。
不过当前实现并没有额外设置“最多只升级 N 个 block”的上限，凡是满足条件的可疑 block 都会尝试 escalation。

### GLM-OCR 的调用方式

当一个 block 需要 escalation 时，系统会：

1. 用 `PyMuPDF` 从 PDF 中按页码和 bbox 裁出区域；
2. 以 2x 分辨率渲染成 PNG；
3. 转成 data URL；
4. 连同原 OCR 结果一起发给 GLM OCR 模型。

Prompt 设计非常克制，核心要求是：

- 只输出识别后的纯文本；
- 保留大小写、数字、单位、标点、换行和特殊符号；
- 不要根据医学常识或上下文擅自改写；
- 不要把“看起来错的原文”修成你认为对的内容。

这类 prompt 的目标不是让模型“理解并润色”，而是让它尽量做严格转写。

### 大模型托底后的后置验收

项目并不会无条件接受大模型返回结果，而是还有一层 acceptance check。

候选结果会被拒绝的情况包括：

- provider 明确表示不该替换；
- 候选文本为空；
- 候选与原文完全相同；
- 候选过长，像是幻觉性扩写；
- 候选过短，信息丢失明显；
- 在通用守门逻辑里，如果候选看起来可疑、而原文反而不可疑，则会拒绝替换。

只有通过这些检查，block 才会真正替换原始文本。

这一步非常关键，因为它说明项目里的大模型不是“最终裁判”，而是“候选提供者”，最后仍由本地规则做守门。

### 替换后的统一重渲染

如果至少有一个 block 被成功替换，系统会重新生成 `document.markdown_text`，确保：

- block 层的纠正结果
- markdown 展平结果
- 后续落库与检索文本

保持一致。

### escalation 的可观测信息

真正进入 escalation 尝试的 block，会记录自己的 escalation metadata，例如：

- provider
- model_name
- original_text
- candidate_text
- accepted
- reason
- confidence
- usage
- error

整篇文档还会记录汇总统计：

- attempted_block_count
- accepted_block_count
- skipped_block_count
- error_block_count
- disabled_reason

这些信息同样会写入 `asset.metadata_json["extraction"]["ocr_escalation"]`。

## 第五层：结构化落库与后续链路

### OCR 结果不是只保存在一份 raw_text

当前项目在 OCR 完成后，会把结构化结果写入多张表，而不是只存一段大文本：

- `paper_document_pages`
- `paper_document_blocks`
- `paper_document_tables`
- `paper_document_pictures`

写入内容包括：

- 文本本体
- 顺序信息
- 节级路径
- 页码
- bbox
- provenance
- 图片描述
- 各类 metadata

同时，旧结构会先清空，避免重解析时新旧结构混杂。

### 统一渲染后进入索引

结构化结果落库后，会重建 paper index。  
统一渲染逻辑会把：

- heading
- paragraph
- table caption / table rows
- picture caption / picture description

按 `reading_order` 和页码重新组织成可检索文本。

因此，OCR 优化的收益不会只停留在“页面看起来更对”，而是会直接影响：

- RAG chunk 质量
- section summary / paper summary 质量
- 后续组织题集抽取质量

### OCR 后还会继续触发 LLM 下游任务

当 `pdf_ingest` 完成后，如果聊天模型配置可用、并且已经有可用于摘要的结构化文本，系统会先触发 `paper_summary`。  
而 `paper_question_set` 还需要额外满足例如：

- 已经生成 `structured_summary`
- 当前组织配置了 question set

等条件后，才会继续入队。

也就是说，OCR 在这个项目里不是终点，而是后续结构化理解、检索和问答链路的地基。

## 当前项目相对“纯 Docling”的额外优化清单

如果只看“项目额外做了什么”，可以概括成下面几类：

### 1. 前置质量路由

- 用 `PyMuPDF` 先抽样前几页 text layer；
- 判断是否存在 `toxic_text_layer`；
- 自动决定是否启用 `force_full_page_ocr`；
- 保留人工总开关 `DOCLING_FORCE_FULL_PAGE_OCR`。

### 2. OCR backend 工程化

- 默认切到 `RapidOCR`；
- 按语言选择 OCR 模型；
- 自动下载并缓存 RapidOCR v5 模型 artifact；
- 通过配置控制 OCR engine 和行为。

### 3. 结构保真增强

- 不只要 markdown，还提取 pages/blocks/tables/pictures；
- 为结构化对象补齐 `section_path`、`reading_order`、`bbox`、`provenance`；
- 表格与图片会回挂到上下文结构中。

### 4. 本地后处理

- 全角半角归一化；
- SymSpell 拼写纠错；
- 医学词表补强；
- 针对单位、`±`、数字混淆等问题做规则修复；
- 覆盖正文、表格、caption、表格单元格等多个层面。

### 5. 大模型区域级托底

- 只处理可疑 block；
- 只裁局部 bbox，不整页、不整篇重跑；
- 用严格转写 prompt 降低“帮你脑补”的风险；
- 用本地 acceptance check 决定是否采纳模型结果。

### 6. 可观测性与留痕

- 将 routing decision、postprocess stats、escalation stats 统一写入 extraction metadata；
- 保留每页 assessment；
- 保留每个 block 的 escalation 决策信息；
- 方便后续回溯和 benchmark。

## 配置项总览

当前 OCR 相关关键配置包括：

### Docling 与 OCR backend

- `DOCUMENT_EXTRACTOR`
- `DOCLING_DO_OCR`
- `DOCLING_DO_TABLE_STRUCTURE`
- `DOCLING_OCR_ENGINE`
- `DOCLING_OCR_LANGUAGES`
- `DOCLING_FORCE_FULL_PAGE_OCR`
- `DOCLING_GENERATE_PICTURE_IMAGES`
- `DOCLING_IMAGES_SCALE`
- `DOCLING_DOCUMENT_TIMEOUT_SECONDS`

### 前置路由

- `OCR_QUALITY_ROUTING_ENABLED`
- `OCR_QUALITY_SAMPLE_PAGES`
- `OCR_DOCLING_FALLBACK_ENGINE`

### 本地后处理

- `OCR_POSTPROCESS_ENABLED`
- `OCR_SYMSPELL_ENABLED`
- `OCR_RULES_ENABLED`
- `OCR_MEDICAL_WORDLIST_PATH`

### 大模型托底

- `OCR_ESCALATION_ENABLED`
- `OCR_ESCALATION_PROVIDER`
- `GLM_OCR_BASE_URL`
- `GLM_OCR_API_KEY`
- `GLM_OCR_MODEL`
- `GLM_OCR_TIMEOUT_SECONDS`
- `GLM_OCR_MAX_RETRIES`
- `GLM_OCR_VERIFY_SSL`

## 当前实现的边界与尚未落地部分

下面这些方向在仓库文档中已有规划，但当前代码还没有完整落地成主流程：

### 1. canary OCR / 试探页 OCR

当前实现是：

- 先看 text layer 质量，再决定整篇是否强制全页 OCR。

还没有做到：

- 对单个关键页先临时做一次试探 OCR，再比较 text layer 与 OCR 质量后决定整篇策略。

### 2. 页级混合 OCR

当前实现是文档级决策：

- 要么按默认策略；
- 要么整篇 `force_full_page_ocr=true`。

还没有做到：

- 正常页保留 text layer；
- 只有坏页单独做 OCR；
- 最后再合并统一结构输出。

### 3. 更细的表格专项修复

当前已对表格文本做统一后处理，但还没有独立的表格专用校验层，例如：

- 数值合法性校验
- 列级单位一致性检查
- 表头词表专项纠正

## 最终总结

当前 ResearchDock 的 OCR 流程，本质上已经从“单纯调用 Docling”演进成了一个四层防线：

1. **前置路由**：先判断 text layer 是否可信，自动决定是否强制全页 OCR；
2. **结构化提取**：继续由 Docling 负责版面、块、表格、图片等主结构输出；
3. **本地后处理**：用归一化、词典纠错、规则修复解决大部分低风险 OCR 错误；
4. **大模型托底**：只对命中可疑条件的 block 做区域级 GLM-OCR 升级识别，并用本地规则验收结果。

因此，项目当前的 OCR 优化重点并不在“替换掉 Docling”，而在于：

- 给 Docling 增加更稳的前置路由；
- 给 OCR 输出增加低成本、可解释的后处理；
- 把大模型限制在最需要它的小范围疑难样本上；
- 把所有决策和修复过程写进 metadata，方便后续排查、benchmark 和继续演进。
