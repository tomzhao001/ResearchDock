# AI 知识库检索优化 Roadmap

## 背景

ResearchDock 当前已经具备最小可用的论文知识库问答能力，但从实际测试反馈看，检索效果明显不足，尤其是在“简单问题也匹配不到”的场景下已经暴露出基础检索层的问题。这个文档的目标不是罗列概念，而是结合当前系统实现，给出一条能逐步落地、逐步验证的优化路线。

本文一方面参考了当前主流 RAG / AI 知识库优化方案，另一方面直接映射到你现在仓库中的实现，重点回答三个问题：

- 现在为什么效果差
- 主流系统通常怎么解决
- 在这个项目里应该按什么顺序做，才能以最小复杂度拿到最大收益

## 当前系统现状诊断

### 已实现能力

- 文档入库后会做固定窗口分块、embedding、入库到 `paper_chunks`
- 对话时会把问题和最近两条用户消息拼成检索 query
- 检索阶段会尝试使用 embedding 相似度；如果没有 embedding 或失败，则退化成简单词项重叠
- 回答阶段有知识库优先和 fallback_general 两种模式

核心实现位于：

- `backend/app/services/rag.py`
- `backend/app/services/llm.py`
- `backend/app/services/pdf_extraction.py`

### 当前效果差的主要原因

#### 1. 分块策略过于粗糙

当前分块是“全文空白折叠后按固定字符窗口切块”，默认 `1000` 字符、`150` overlap。这种方式实现简单，但对论文、技术文档、双栏 PDF、表格、章节结构都不友好。它无法保留：

- 章节标题
- 段落边界
- 图表/表格上下文
- 页码与版面关系

结果是 chunk 在语义上经常“不完整”或“掺杂太多噪音”，embedding 很容易被稀释。

#### 2. 检索不是 BM25，也不是标准 hybrid retrieval

当前 lexical 部分只是 `_lexical_score()` 的词项交集比例，不是 BM25。它没有：

- 词频/逆文档频率加权
- 文档长度归一
- 稀疏召回能力

这会直接导致缩写、专有名词、公式名、配置项、方法名这类“必须精确命中”的问题召回很差。业界大量经验都认为技术文档、内部文档、法律/医疗文档等场景，单纯向量检索经常漏掉 exact-match 证据，[Redis 的混合检索文章](https://redis.io/blog/hybrid-search-benefits-rag-systems) 和社区讨论都在强调这一点，[r/Rag 的生产经验贴](https://www.reddit.com/r/Rag/comments/1rf7xf6/whats_your_experience_with_hybrid_retrieval) 甚至给出从约 60% 到 85% 的体验提升，但这类数字应视为社区个例而不是通用 benchmark。

#### 3. 没有 reranker，粗召回结果直接进入生成

当前 top-k 结果来自一次初排，没有 cross-encoder reranker 或 LLM reranker。现实里 dense/sparse 的 first-pass retrieval 通常负责“高召回”，reranker 负责“高精度”。不少实践都把 `BM25 + vector + reranker` 视为 RAG 的默认组合，[Superlinked 的总结](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) 与 [Towards AI 的实践文](https://towardsai.net/p/machine-learning/hybrid-search-rag-that-actually-works-bm25-vectors-reranking-in-python) 都是这个思路。

#### 4. 对话 query 没有做真正的“检索查询重写”

当前只是把最近两条用户消息拼到当前问题后面，这对 pronoun、省略、追问、长指令型问题帮助很有限。检索系统真正需要的是：

- decontextualization（把“它/这个方法/上面那篇论文”改写成显式实体）
- intent stripping（把“帮我总结、列优缺点、对比一下”里的生成指令和检索主题拆开）
- multi-query / query expansion（生成多个检索表达）

这一方向近两年很活跃，[Anthropic 的 contextual retrieval 指南](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)、[Anthropic Engineering 的 contextual retrieval 介绍](https://www.engineering.fyi/article/introducing-contextual-retrieval)、[HyDE 文档](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde) 和 SIGIR 的 multi-query rewriting 工作都说明：查询表达本身就是 RAG 成败的重要变量。

#### 5. 引用与归因还不够细

现在返回的是 chunk 级 snippet，`page_from/page_to` 还是空的，也没有 claim-level attribution。这样虽然已经比“无引用”好，但还不够支持：

- 用户校验
- 后续自动评估 citation precision
- 严格拒答阈值

归因质量本身已经是单独研究方向。[Attribute First, then Generate](https://aclanthology.org/2024.acl-long.182/) 和 [AIS attribution framework](https://direct.mit.edu/coli/article/49/4/777/116438/Measuring-Attribution-in-Natural-Language) 都强调：是否有“可核验来源”，不只是 UI 问题，而是系统可靠性问题。

#### 6. 没有评测集，优化会靠感觉

现在系统还没有 retrieval eval / answer eval 基线。没有类似：

- 问题集
- gold passages
- hit@k / MRR / nDCG
- groundedness / attribution quality

这意味着你现在觉得“效果差”，但还无法知道差在：

- 召回
- 排序
- chunking
- OCR 噪音
- 还是生成阶段误读

## 外部主流方案调研总结

### 1. Hybrid retrieval 已经是主流基线

现在成熟系统很少只靠单一路径向量检索。常见做法是：

- sparse：BM25 / keyword / inverted index
- dense：embedding retrieval
- fusion：RRF 或加权融合
- rerank：cross-encoder 或 LLM reranker

这种组合的原因很简单：BM25 擅长 exact terminology，dense 擅长语义相近但措辞不同的问题。[Redis](https://redis.io/blog/hybrid-search-benefits-rag-systems)、[Superlinked](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)、[Towards AI](https://towardsai.net/p/machine-learning/hybrid-search-rag-that-actually-works-bm25-vectors-reranking-in-python) 都把这当成现实系统的常规组合。

### 2. Chunking 正从“固定长度”转向“结构感知”

主流优化不再只讨论 chunk size，而是讨论 chunk boundary 是否保留作者原始思路、文档结构和版面关系。[Weaviate 的 chunking 指南](https://weaviate.io/blog/chunking-strategies-for-rag) 直接把 chunking 归入 context engineering；[ScienceDirect 的语义 chunking 论文](https://www.sciencedirect.com/science/article/pii/S0950705125019343) 和 [Databricks 的 chunking 实践总结](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089) 也都在强调“没有通用最优 chunking，必须按文档类型设计”。

对论文/技术 PDF 来说，尤其重要的是：

- section-aware chunking
- sentence-window retrieval
- parent-child retrieval
- layout-aware parsing

### 3. 层级式 / 树状 / 多粒度检索适合长文档

RAPTOR 把 chunk 递归聚类、摘要，构建树状索引，让系统既能取到叶子级细节，也能取到高层语义总结，[RAPTOR 论文](https://arxiv.org/abs/2401.18059) 和 [官方实现](https://github.com/parthsarthi03/raptor) 是这一路线的代表。另一个更轻量、工程上更早能落地的方向是 parent-child / small-to-big retrieval：用小 chunk 做召回，再返回大 chunk 或 surrounding window 供生成，[small-to-big retrieval 文章](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4) 就是这个思路。

对你现在这个系统而言，完整 RAPTOR 不是第一优先级，但“多粒度索引”非常值得提前纳入设计。

### 4. Query rewriting / contextual retrieval 正在成为高 ROI 优化

很多系统匹配不到，并不是因为库里没答案，而是用户提问方式和索引表达方式不一致。典型问题包括：

- pronoun / 指代
- 问题太口语化
- 一个 query 里混入多个任务指令
- 术语和文档原文不一致

[Anthropic contextual retrieval](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide) 展示了 contextual embeddings 带来的 Pass@10 提升；工程文章中也给出 retrieval failure 明显下降的信号，[Anthropic Engineering](https://www.engineering.fyi/article/introducing-contextual-retrieval) 报告了 retrieval failure 下降和与 reranking 组合后的进一步收益。HyDE 和 multi-query rewriting 则是更通用的 query-side recall 提升手段，[Haystack 的 HyDE 文档](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde) 和 SIGIR 的 multi-query rewriting 研究都值得参考。

### 5. Obsidian / second-brain 路线强调“关系”和“本地知识组织”

Obsidian 类知识库的经验对团队知识库很有参考价值，因为它们天然关注：

- note granularity
- backlinks / wikilinks
- graph traversal
- tags / hyperedges
- local-first 可验证性

例如 [obsidian-note-taking-assistant](https://github.com/sspaeti/obsidian-note-taking-assistant) 已经把 semantic search、backlinks、graph traversal、shared tags、graph-boosted search 放在一起。这类系统的启发是：知识库不仅是 chunk 的集合，还可以是“文本 + 链接关系 + 标签关系 + summary”的多维结构。

### 6. 结构化抽取正在成为高质量知识库的上游能力

如果原始文档包含表格、实体、实验结果、方法名、指标值、引用关系，仅靠纯文本 chunk 往往不够。`LangExtract` 的价值就在这里：它强调“带 source grounding 的结构化抽取”，每个抽取结果都能回到原文字符偏移，[Google 官方博客](https://developers.googleblog.com/introducing-langextract-a-gemini-powered-information-extraction-library/) 和 [LangExtract README](https://github.com/google/langextract/blob/main/README.md) 都把 `precise source grounding` 和 `structured outputs` 作为核心卖点。

这对论文知识库尤其重要，因为论文问答经常不是“找一句话”，而是：

- 找一个方法定义
- 找实验设置
- 找指标比较
- 找局限性
- 找某个数据表中的具体数值

## 目标架构原则

未来的知识库检索层建议遵循下面五个原则：

1. 召回必须是多路的，不要押宝单一路径
2. 索引必须保留文档结构，而不是只有纯文本 blob
3. 排序必须显式优化，不让粗召回结果直接进生成
4. 引用必须可核验，最好能精确到页码、段落、字符范围
5. 所有优化都必须有离线评测闭环

## Roadmap 总览

### Phase 0：先建立评测与可观测性

这是最优先的工作，因为没有评测就没有优化。

#### 目标

- 把“效果差”变成可量化问题
- 把调参从主观体验变成可回归测试

#### 建议改动

- 建一个小型 benchmark 集，至少 80 到 150 个问题
  - factual exact match
  - 缩写/方法名
  - 多轮追问
  - 总结类问题
  - 表格/实验结果类问题
- 为每个问题标注：
  - gold paper
  - gold chunk / passage
  - 是否需要跨段整合
  - 是否允许 fallback general
- 增加指标：
  - retrieval hit@k
  - MRR / nDCG
  - answer groundedness
  - citation precision
  - abstention accuracy
- 落日志：
  - query rewrite 前后
  - first-pass candidates
  - reranker top results
  - 最终 answer_mode

#### 为什么先做

因为后续 hybrid、reranker、chunking、query rewrite 都会互相影响，没有 benchmark 只会陷入“感觉有提升”。

#### 验收标准

- 可以稳定复现“哪些问题匹配不到”
- 任意一次检索优化都能跑对比报告

### Phase 1：把当前检索升级为真正的 hybrid retrieval

这是最高 ROI 的一阶段，应该最先提升线上效果。

#### 目标

解决“简单问题匹配不到”“术语命不中”“缩写命不中”的核心问题。

#### 建议改动

- 在 PostgreSQL 内补真正的 sparse 检索：
  - 优先 `tsvector + tsquery` / PostgreSQL full-text
  - 或同步一份 BM25 索引到专门搜索层
- dense 保留 pgvector 或兼容向量检索，不再在 Python 里把所有 chunk 拉出来全扫
- 用 RRF 或加权融合做 first-pass merge
- 召回结果扩大到例如 sparse top 20 + dense top 20
- 引入轻量 reranker：
  - 第一阶段可先用 API reranker
  - 或 cross-encoder/bge-reranker 一类模型

#### 对当前仓库的具体影响

- `paper_chunks.embedding` 重新回归 pgvector 存储
- 新增 sparse 索引字段与构建逻辑
- `backend/app/services/rag.py` 重写 `_search_chunks()`
- 增加 first-pass result logging

#### 验收标准

- exact-term 问题 hit@5 明显提升
- benchmark 上 retrieval hit@10 提升至少一个明显档位
- “明明文档里有，但没捞到”的 case 数量显著下降

### Phase 2：升级分块，从固定长度到结构感知 + 多粒度

这是第二高 ROI 工作，尤其适合论文 PDF。

#### 目标

解决 chunk 语义不完整、引用不稳定、长文跨段丢上下文的问题。

#### 建议改动

- 从 PDF 解析阶段保留更多结构：
  - page number
  - block bbox
  - section title
  - heading level
  - table / figure / caption 类型
- chunk 改成两层：
  - child chunks：小粒度检索单元
  - parent chunks：大粒度生成上下文
- 对论文优先做：
  - section-aware chunking
  - paragraph-aware chunking
  - sentence-window retrieval
- 为每个 chunk prepend 轻量 contextual header，例如：
  - paper title
  - section title
  - page range
  - optional summary sentence

#### 技术参考

- [Weaviate chunking guide](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Structure-aware retrieval discussion](https://medium.com/@yu-joshua/adding-structure-aware-retrieval-to-genai-stack-373976de14d6)
- [layout-aware document processing sample](https://github.com/aws-samples/layout-aware-document-processing-and-retrieval-augmented-generation/blob/main/README.md)
- [LLM Sherpa / layout-aware PDF discussion](https://ambikasukla.substack.com/p/efficient-rag-with-document-layout)

#### 验收标准

- chunk 平均可读性和语义完整性明显提升
- 引用能稳定给出页码或 section
- 总结类、多段推理类问题召回更稳定

### Phase 3：补 query understanding，别把原始用户话术直接拿去检索

这会显著改善多轮对话和口语提问。

#### 目标

让检索 query 更像“搜索表达”，而不是“聊天表达”。

#### 建议改动

- 增加 query rewrite 层，至少拆成两个产物：
  - retrieval query
  - generation instruction
- 对多轮对话做 decontextualization：
  - “它/这个方法/上面那篇论文”改写成显式实体
- 对复杂问题做 multi-query：
  - 原问题
  - keyword-heavy query
  - semantic paraphrase query
- 可选引入 HyDE，用于召回困难问题
- 对 query 和文档统一做 terminology normalization

#### 为什么重要

你现在的问题很可能不是“没有语义相似度”，而是“检索 query 被聊天语气和指代污染了”。这正是 query rewriting 的高 ROI 区间。

#### 验收标准

- 多轮追问 hit@k 提升
- pronoun / ellipsis 类问题的 miss 明显减少
- 复杂指令型问题比当前版本更稳定

### Phase 4：引入 attribution-first 的回答链路

这一阶段解决“用户信不信”和“系统能不能严格拒答”。

#### 目标

让引用不是装饰，而是系统控制 hallucination 的核心机制。

#### 建议改动

- 回答前先做 evidence selection，再生成
- chunk 级引用升级到 claim-supporting evidence
- 返回：
  - evidence ids
  - page range
  - section path
  - confidence / support score
- 增加 evidence sufficiency threshold：
  - 低于阈值时直接拒答
  - 或进入“知识库未命中”的显式路径
- 增加 answer verifier / groundedness checker

#### 技术参考

- [Attribute First, then Generate](https://aclanthology.org/2024.acl-long.182/)
- [AIS attribution framework](https://direct.mit.edu/coli/article/49/4/777/116438/Measuring-Attribution-in-Natural-Language)

#### 验收标准

- 用户能快速定位引用位置
- 拒答更稳定，不再“看似回答了但其实没依据”
- citation precision 成为正式评测指标

### Phase 5：把知识库从“chunk 集合”升级成“多维知识结构”

这是中长期能力建设，不建议在基础检索没打稳之前提前做重。

#### 目标

支持更复杂的问题，例如：

- 跨论文比较
- 方法演进
- 同概念在不同论文中的出现
- 围绕主题而不是围绕单段文本检索

#### 可选路线

##### 路线 A：轻量多粒度索引

- leaf chunks
- section summaries
- paper summaries
- corpus topic summaries

这是最推荐的中期路线，性价比高。

##### 路线 B：RAPTOR / hierarchical retrieval

- 对长文或专题集合做树状摘要索引
- 同时检索 summary nodes 和 leaf nodes

适合：

- 长论文
- 专题综述
- 跨章节 reasoning

##### 路线 C：graph / Obsidian-style retrieval

- note/paper nodes
- citation edges
- shared tags
- concept/entity links
- graph-boosted retrieval

适合未来做：

- 研究主题浏览
- 论文关系探索
- “相关工作/相似方法/引用链”问答

#### 结论

短期不要直接上最重的 graph RAG。先做“多粒度索引 + parent-child + metadata filters”，等评测证明确实需要，再逐步引入树状或图结构。

### Phase 6：把结构化抽取作为上游增强层

这是为论文知识库做“深水区优化”的关键一步。

#### 目标

让系统不只会检索文本，还会检索结构化知识。

#### 建议改动

- 用结构化抽取增强论文 schema，例如：
  - method name
  - task
  - dataset
  - metric
  - result value
  - limitation
  - claims
- 对表格做专门抽取，不再把表格简单展平成连续文本
- 对关键实体保留 source offsets / page grounding
- 评估引入 `LangExtract` 或同类方案做 extraction layer

#### 为什么值得做

论文问答里很多高价值问题其实是结构化问题：

- “这篇论文在 ImageNet 上 top-1 提升多少？”
- “方法 A 和 B 的主要区别是什么？”
- “哪些论文把 limitation 提到了数据规模？”

纯文本 RAG 对这类问题天然吃亏。

#### 参考资料

- [Google LangExtract blog](https://developers.googleblog.com/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [LangExtract README](https://github.com/google/langextract/blob/main/README.md)
- [Unstructured preprocessing guide](https://unstructured.io/blog/level-up-your-genai-apps-essential-data-preprocessing-for-any-rag-system)
- [Databricks unstructured data pipeline guide](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)

## 推荐的实际落地顺序

### 第一阶段：2 到 3 周，快速止血

- 建 benchmark 与日志
- 真正的 hybrid retrieval
- first-pass reranker
- 基础 query rewrite

这是最应该先做的，因为它最有机会直接把“简单问题匹配不到”拉回来。

### 第二阶段：2 到 4 周，提升稳定性

- 结构感知 chunking
- parent-child retrieval
- richer metadata
- 更好的 citation / page grounding

### 第三阶段：4 周以上，做差异化能力90.

- hierarchical summaries / RAPTOR-lite
- graph-boosted retrieval
- structured extraction
- query planning / multi-hop retrieval

## 不建议现在就做的事情

- 不建议一上来就做完整 RAPTOR
- 不建议先做大而全的知识图谱
- 不建议在没有 benchmark 的前提下反复调 chunk_size
- 不建议把主要希望寄托在“换更强模型”

你当前的问题更像 retrieval system 问题，不像 model intelligence 问题。

## 对 ResearchDock 的建议结论

如果目标是最快把效果从“明显不好”拉到“可用”，推荐优先级如下：

1. 评测与日志
2. BM25 / sparse + dense hybrid retrieval
3. reranker
4. query rewrite / decontextualization
5. 结构感知 chunking + parent-child
6. 更强 attribution
7. 层级索引 / graph / structured extraction

一句话概括：

当前系统不是“模型不够强”，而是“知识进入检索层后被过度扁平化了，query 也没有被优化成适合检索的表达”。未来 roadmap 应该优先把知识库从“固定长度 chunk 列表”升级成“多路召回 + 结构保留 + 可验证引用”的检索系统，再考虑树状结构、多维结构和结构化抽取。

## 参考资料

- [Enhancing RAG with contextual retrieval | Claude Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Introducing Contextual Retrieval | Anthropic Engineering](https://www.engineering.fyi/article/introducing-contextual-retrieval)
- [Hypothetical Document Embeddings (HyDE) | Haystack Documentation](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversational Passage Retrieval | SIGIR 2024](https://dl.acm.org/doi/10.1145/3626772.3657933)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [GitHub - parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)
- [Advanced RAG 01: Small-to-Big Retrieval](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4)
- [Optimizing RAG with Hybrid Search & Reranking | Superlinked](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Hybrid search benefits: Why your RAG system needs both keyword & vector search | Redis](https://redis.io/blog/hybrid-search-benefits-rag-systems)
- [Hybrid Search RAG That Actually Works: BM25 + Vectors + Reranking in Python | Towards AI](https://towardsai.net/p/machine-learning/hybrid-search-rag-that-actually-works-bm25-vectors-reranking-in-python)
- [Chunking Strategies to Improve LLM RAG Pipeline Performance | Weaviate](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Optimising retrieval performance in RAG systems: A new growing window semantic chunking strategy to address weak semantic boundaries](https://www.sciencedirect.com/science/article/pii/S0950705125019343)
- [Adding Structure-Aware Retrieval to GenAI Stack](https://medium.com/@yu-joshua/adding-structure-aware-retrieval-to-genai-stack-373976de14d6)
- [Using Document Layout Structure for Efficient RAG](https://ambikasukla.substack.com/p/efficient-rag-with-document-layout)
- [Introducing LangExtract: A Gemini powered information extraction library](https://developers.googleblog.com/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [google/langextract README](https://github.com/google/langextract/blob/main/README.md)
- [Data Preprocessing for RAG: A Complete Guide | Unstructured](https://unstructured.io/blog/level-up-your-genai-apps-essential-data-preprocessing-for-any-rag-system)
- [Build an unstructured data pipeline for RAG | Databricks](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
- [Attribute First, then Generate: Locally-attributable Grounded Text Generation](https://aclanthology.org/2024.acl-long.182/)
- [Measuring Attribution in Natural Language Generation Models | MIT Press](https://direct.mit.edu/coli/article/49/4/777/116438/Measuring-Attribution-in-Natural-Language)
- [obsidian-note-taking-assistant](https://github.com/sspaeti/obsidian-note-taking-assistant)
- [Man and machine: GPT for second brains](https://reasonabledeviations.com/2023/02/05/gpt-for-second-brain/)
- [r/Rag: What's your experience with hybrid retrieval](https://www.reddit.com/r/Rag/comments/1rf7xf6/whats_your_experience_with_hybrid_retrieval)

