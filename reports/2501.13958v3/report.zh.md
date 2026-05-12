# A Survey of Graph Retrieval-Augmented Generation —— 研究简报

## 背景

LLM 是宽而不深：训练语料以通用 Web 文本为主，一旦推入专业语境 —— 医疗、法律、金融、代码 —— 就会塌方，因为这些场景下答错的代价高、规则又是局部且精确的。综述把这件事拆成三个维度：预训练**知识**广而浅、专业领域的**推理**需要带约束的多步逻辑一致性、**上下文敏感性**让 LLM 在"同一术语在不同场景下含义不同"时栽跟头。

之前两条主流路线都没干净地解决问题。在领域数据上 fine-tune 会带来灾难性遗忘，并且**新知识与预训练冲突时会产生新的幻觉**（这是综述引的一项 Google Research 的结果）。传统 RAG —— 文本切块 + 向量检索 —— 留下四个未解的限制：

1. **复杂查询理解**：基于 chunk 的关键词 / 向量相似度抓不住多跳推理。问 A 与 D 的关系，朴素 RAG 把 A、D 各自检索出来，**漏掉了桥接的 B 和 C**。
2. **分布式领域知识**：chunking 牺牲上下文完整性；概念之间的层级关系消失。
3. **LLM 自身约束**：向量检索召回过多，LLM 的固定上下文窗口（2K-32K tokens）装不下，关键信息被截断。
4. **效率与可扩展性**：知识库一大，无结构文本检索越来越慢。

综述的论断是：**把知识组织成图**能正面回应以上四点，而由此形成的范式 GraphRAG 已经成熟到值得做一份正经的分类学。

## 核心方案

综述最有价值的贡献是这套分类学。现有 GraphRAG 系统按"图扮演什么角色"分三类：

**❶ Knowledge-based GraphRAG** —— 图作为知识载体。要么从语料用 Open Information Extraction 构图（Microsoft GraphRAG、GraphReader、QUEST、AutoKG），要么挂载到已有的 KG（DBpedia、YAGO；或领域专用：生物医学的 SPOKE、学术界的 AceKG、法律界的 Lynx）。检索是**推理路径遍历**：ToG 在关系三元组上跑 beam search；RoG 和 KGR 把 LLM 当 planning agent；KnowGPT 把路径探索形式化为强化学习问题。

**❷ Index-based GraphRAG** —— 图作为原始文本块之上的索引工具。节点是文本 chunk，边编码语义相似度或共享实体。检索是**图遍历决定召回哪些 chunk**。GNN-ret、PG-RAG、KGP 是这个路线。图提供保留上下文的查找，事实内容仍然在底层文本里。

**❸ Hybrid GraphRAG** —— 两者都做，例如 Microsoft GraphRAG、MedGraphRAG、GoR。图既携带摘要后的知识，又索引到原文 chunk，让 LLM 在需要时还能回退到细粒度上下文。

每种范式都有同样的三阶段工作流：**知识组织 → 检索 → 集成**。检索这一段着墨最多，因为设计选择最多。综述沿两个维度把代表性系统排成表：

- **检索技术**：**相似度类**（BERT/SentenceBERT embedding + cosine、TF-IDF、PCST 剪枝）、**逻辑类**（关系路径生成、规则挖掘、MCTS 规划）、**GNN 类**（RGNN 做查询扩展、GAT 提图特征）、**LLM 类**（LLM 当图上的 agent：社区摘要、关系上的 beam search、关键词搜）、**RL 类**（KnowGPT、Spider 用深度 RL 学检索策略）。
- **检索策略**：**多轮**（DialogGSR、Graph-CoT、GoR 迭代扩张子图）、**后检索**（CoK、KGR 对检索到的事实做验证 / 主张核查）、**混合检索**（StructRAG 按查询在五种候选结构之间路由；ToG-2 在 KG 遍历和文档检索之间交替）。

知识集成 —— 怎么把检索到的子图喂给 LLM —— 单列一节：结构化 prompt 模板、把图节点 verbalize 成 token、训练直接消费图嵌入的 adapter。按作者自己的口吻，这块"作为研究方向比检索更不成熟"。

## 与其他方案对比

综述里 `GraphRAG vs 传统 RAG` 那张表给出四个胜场：

| 维度 | 传统 RAG | GraphRAG |
|---|---|---|
| 知识表示 | 扁平 chunk，语义关系隐式 | 显式关系、层级、多跳路径 |
| 灵活性 | 以文本为中心 | 一张图里同时容纳结构化 / 半结构化 / 无结构数据 |
| 效率 | 规模化慢；更新得重索引 | 图数据库为关系查询优化；已发表对比里**少用 26-97% tokens**；新增节点不需重索引 |
| 可解释性 | 不透明的向量匹配 | 图上的推理路径可追踪 |

**少用 26-97% tokens** 这条是综述里被引用最多的运营性收益 —— 来自"LLM 消费一个聚焦子图而非一堆文本块"。

一个有分量的差异化：作者把自己的分类学与之前的 GraphRAG 综述区分开 —— 之前的综述基本只描述工作流，而他们把 **Hybrid** 列为一等公民。论证是：纯 Knowledge-based GraphRAG（依赖 KG 质量）与纯 Index-based GraphRAG（依赖文本检索）相互制约 —— 2024–2025 收敛出来的生产级系统（LightRAG、StructRAG、MedGraphRAG、Microsoft GraphRAG）**全是 Hybrid**，因为纯路线都不够强。

综述还少见地坦白讲了**局限**：(i) 高质量 KG 稀缺，从语料构建 KG 代价高；(ii) 抽取出来的 KG 的粒度权衡 —— 细粒度意味着大图 + 高算力，粗粒度意味着摘要丢信息；(iii) 用 LLM 做 KG 摘要的方法（比如 Microsoft 的 GraphRAG）规模化时 token 成本难以承受。

## 总结

这是值得加书签的那种综述，主要做对了三件事：

1. **清晰的分类学**（Knowledge-based / Index-based / Hybrid）把每个系统都映射到一个可辨认的格子里，正交的工作流分解（组织 / 检索 / 集成）穿过它。
2. **完整的 Table I**，把 ~30 个代表性 GraphRAG 系统按预处理模型、匹配算法、剪枝方法、输出类型表格化。**这是大部分研究者真正会用的产物**。
3. **配套的 GitHub 仓库** ([DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG)) 收集论文、开源项目、benchmark 数据集 —— 明确设计成"活得比综述本身久"。

综述刻意不做的事：横向 benchmark 对比。作者指出 GraphRAG 各实现之间缺少标准化的 benchmark，把这当成开放问题，**没尝试钦点赢家**。这是诚实的处理 —— GraphRAG 的 benchmark 还没稳定到能可靠排名。

对 2025–2026 要落地 GraphRAG 的人，战略层面的 takeaway：**这个领域已经收敛到 Hybrid 是生产范式**。纯 Knowledge-based 在你已经有高质量领域 KG 时（生物医学、法律）很好用；纯 Index-based 在原文质量高、多跳需求不重时好用。Hybrid 是更安全的默认 —— 复杂度的成本你得付，但你不再被 KG 完整度或者 chunk 召回率单独卡住。综述结晶了这一收敛趋势，这是它对实践者的主要价值。
