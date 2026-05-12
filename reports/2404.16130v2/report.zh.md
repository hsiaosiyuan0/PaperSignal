# From Local to Global: A GraphRAG Approach to Query-Focused Summarization —— 研究简报

## 背景

向量 RAG —— "切块、嵌入、取 top-k" 那套经典管线 —— 在**局部**问题上表现很好：凡是答案藏在语料里某一小撮可定位的记录中的问题它都能答。它崩溃的地方是 **global sensemaking** 这类问题 —— 那些需要对整个数据集做推理才能回答的问题："这个语料的主要主题是什么？""这些文档整体上怎么看 X？""这些事件之间有什么模式？"

论文的诊断很犀利：向量 RAG 答不了全局问题是因为**没有任何单条记录包含答案 —— 答案是整个集合的属性**。早期的 query-focused summarization (QFS) 概念上能处理这个任务，但规模化扛不住几十篇文档以上的量。同时 LLM 的上下文窗口确实变长了，你也许会想把整个语料塞进去 —— 但在百万 token 量级既不划算也不可靠（长上下文的召回退化已经被反复验证）。GraphRAG 要占的就是这两种失效模式之间的位置。

事后看，这篇论文是**给"现代 GraphRAG"这个范式命名并下定义**的工作。2024–2025 年大多数 GraphRAG 生态（LightRAG、MedGraphRAG、StructRAG、整个 DEEP-PolyU 综述的分类学）的结构都是对这篇论文的变体或批判。

## 核心方案

GraphRAG 把文本语料分五步转成一个可查询的层级化制品：

**❶ 切块 → 实体 + 关系**。文档切成 600 token 的 chunk（重叠 100 token）。用 few-shot exemplar 提示 LLM 抽出带类型的实体、实体间的关系，以及每个实体 / 关系的简短描述。**Claims**（关于实体的可验证事实陈述）在并行一遍中抽出。换语料只需要换 few-shot 示例，不需要重训。

**❷ 实体 → 知识图谱**。跨 chunk 的重复合并（实现里用精确字符串匹配，软匹配也行）。边权用关系被抽出的次数；每个节点 / 边的描述聚合。论文实验里的图规模：Podcast 数据集 8,564 节点 / 20,691 边；News 数据集 15,754 节点 / 19,520 边。

**❸ 图 → 社区**。**Leiden** 社区检测按层级地切分图（C0 根 → C3 叶），每一层都是对所有节点的"互斥 + 集体穷尽"覆盖。这一步是让"全局 → 局部"导航成为可能的关键动作。

**❹ 社区 → 社区摘要**。自底向上摘要：先摘叶级社区；上层摘要在 token 限制压缩下递归从子社区摘要再生成。结果是一棵报告式的摘要层级 —— 你**不带任何查询**就能从顶往下浏览来理解整个语料。

**❺ 查询 → 全局答案（map-reduce）**。给定用户查询：
- *Map*：每个社区摘要独立生成一份部分答案，并给出 0-100 的"对该问题的帮助度"评分。
- *Reduce*：部分答案按帮助度降序排，塞进一个上下文窗口，LLM 产出最终的全局答案。

评估方面他们顺手提出了一个 **adaptive benchmarking** 方法：persona 驱动的问题生成（5 个 persona × 5 个任务 × 5 个问题 = 每语料 125 题），避开了"直接从语料里生成测语料的问题"这种陷阱。

## 与其他方案对比

在两个约 100 万 token 的语料（Podcast transcripts, News articles）上，用 LLM-as-judge 跑六组对照：

| 对比组 | Comprehensiveness | Diversity | Empowerment | Directness |
|---|---|---|---|---|
| GraphRAG (C0–C3) vs 向量 RAG (SS) | **72–83% 胜** | **62–82% 胜** | 混合 | 输给 SS（SS 在 directness 上赢）|
| GraphRAG C1–C3 vs 纯文本 map-reduce 摘要 (TS) | 略胜 | 略胜 | 混合 | 混合 |
| 根级社区 (C0) | 每次查询 token：**比 TS 少 9-43 倍** | — | — | — |

**胜率这一组就是核心结果**：在 global sensemaking 上 GraphRAG 对常规向量 RAG **不是好 5%，而是 3 次里有 2-3 次胜出** —— 而且赢的是用户真正在意的维度（comprehensiveness、diversity）。向量 RAG 在 directness 上还是赢，因为它产的答案更短更聚焦 —— 这一项作者刻意保留作"对照变量"，因为任何冗长但全面的方法都该在简洁度上输。

在 GraphRAG 内部，**C0（根级社区摘要）是甜点位**：成本碾压（比纯文本 map-reduce 便宜 9-43 倍），质量与更深层级竞争。这个层级**不只是实现细节** —— 不同层级实际上是不同的产品配置，便宜+好用选 C0，需要细节就 C2/C3。

论文里一条诚实的披露：Podcast 数据集的图索引在他们的实验机器上跑了 **281 分钟**（用公网 OpenAI 接口）。这就是这套方案的**核心 trade**：一次性繁重的索引成本换来后续便宜且高质量的查询。这也是 LightRAG 以及所有后继者对这篇论文最集中的批评点。

## 总结

这篇是你想真正理解"现代 GraphRAG 到底是什么"时该读的那篇。四点值得记住：

1. **真正的突破是 framing，不是工程**。把向量 RAG（局部）和 QFS（不可扩展）之间的空隙命名出来，并提出"社区摘要层级"作为桥梁 —— 这是改变了领域的事。单个技术点（LLM 抽实体、Leiden、map-reduce）单独看都是已知的。

2. **社区层级是一个真正的产品表面**，不是实现细节。C0 根摘要可以**不带查询**地浏览 —— 它本质上是对一个语料的结构化概览，独立于"回答问题"也有价值。

3. **Adaptive benchmarking 在方法论上有意思**：persona 驱动的问题生成绕开了"global 查询没有 ground truth"的难题；LLM-as-judge + directness 作对照变量是评估这类系统时**正确的形状**。

4. **索引成本是真问题、也是此后所有 GraphRAG 论文的中心张力**。LightRAG 砍掉社区摘要换成双层关键词索引就是冲这条来的；StructRAG 在多种结构间路由部分也是为了避开这条；MedGraphRAG 退到领域 KG 也是为了规避。读完这篇，紧接着读 LightRAG，你能看到"认真对待成本批判"会推出什么样的方案。

开源在 [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)，并已被 LangChain、LlamaIndex、NebulaGraph、Neo4J 的扩展集成 —— 是这个范式的**正典参考实现**。
