# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval —— 研究简报

## 背景

RAPTOR（Stanford，Christopher Manning 组，ICLR 2024）比 Microsoft GraphRAG 早几个月发表，但**诊断完全一样**：标准的 retrieval-augmented LM 一次只取若干段连续短 chunk，这种结构和"需要跨长文本整合信息"的问题天然不匹配。论文里的引子是灰姑娘童话和那个问题 —— *"灰姑娘是怎么走到幸福结局的？"* —— 任何 top-k 的 100-token 片段都不可能包含答案，你需要的是故事的**整条弧**。

它的贡献是第一个被广泛接受的"多层级检索"答案：**递归地把文本 chunk 聚类、对每个 cluster 做摘要、再对摘要继续聚类**，直到聚不下去。最终输出是一棵树 —— 叶子是原文本，每一层往上都是 LLM 对下一层的抽象。查询时**所有层一起检索**。

放到完整的 GraphRAG 弧里看，RAPTOR 是**"树状前身"**：GraphRAG 后来把"层级摘要"这个思路从树挪到了图（社区摘要代替 cluster 摘要），而 LightRAG 干脆主张可以把摘要那层全砍了。

## 核心方案

**树的构建是整个方法的核心**。三层嵌套的循环，反复迭代直到收敛：

**❶ 切块 + embedding**。语料切成 ~100 token 的连续 chunk（句子保持完整 —— 一句话如果会越过 100-token 边界，整句移到下一个 chunk）。每个 chunk 用 SBERT（`multi-qa-mpnet-base-cos-v1`）做 embedding，构成**叶子层**。

**❷ 聚类**。在 embedding 上做 Gaussian Mixture Model 的软聚类。两个实现选择很关键：

- **UMAP** 先做降维再 GMM，因为高维欧氏距离会失真。通过调整 UMAP 的 `n_neighbors`，做**两遍**聚类 —— 先全局聚类，再在每个全局 cluster 内做局部聚类。
- **BIC**（Bayesian Information Criterion）自动选 cluster 数。然后 EM 求解 GMM 参数。

**软聚类很关键**：一个 chunk 可以同时属于多个 cluster，这吻合真实文本里"一段话承载多个主题"的现实。如果某个 cluster 合并后的上下文超过摘要模型的 token 上限，递归地在内部再聚一次。

**❸ 摘要 + 递归**。每个 cluster 的所有 chunk 喂给 `gpt-3.5-turbo`，产出一段摘要。这段摘要变成上一层的新节点，再用 SBERT embed，进入下一轮。继续聚类 + 摘要，直到聚不下去。

一个有针对性的人工标注研究发现：**~4% 的摘要含轻微幻觉**，但**这些错误不会向上层传播**，也对下游 QA 没有可测量的影响 —— 这是"摘要再摘要"这种 pipeline 安全性的有用经验证据。

**两种查询策略**搭在这棵树上做对比，最后**更简单的那个赢了**：

- **Tree traversal（树遍历）**。从根开始，按余弦相似度取 top-k。下钻到这些节点的子节点，再 top-k。重复 $d$ 层。结果拼起来。
- **Collapsed tree（坍塌树）**。把整棵树拍平成一袋节点。对 query 算余弦排序。从高到低取节点，直到达到 token 预算（论文用 2000 token，约等于 top-20）。

Collapsed tree 全面占优。论文给的理由有结构性意味：tree traversal 在结果里的"层数比例"是**预先固定**的，所以"抽象/细节"的混合度由树拓扑决定，而不是由 query 决定。Collapsed tree 让 query 自身的粒度去决定该取哪一层的节点。

## 与其他方案对比

三个数据集，都是专门挑出来压"长文档推理"的：**NarrativeQA**（书和电影脚本全文）、**QASPER**（针对完整 NLP 论文的提问）、**QuALITY**（~5000 token 段落上的多选题，外加一个 HARD 子集，人类在限时下都做错）。

**控制变量的检索消融实验**（UnifiedQA-3B 当 reader，SBERT/BM25/DPR 各自 ± RAPTOR）：

| Reader: UnifiedQA-3B | QuALITY Acc | QASPER F1 |
|---|---|---|
| SBERT + RAPTOR | **56.6%** | **36.7%** |
| SBERT alone | 54.9% | 36.2% |
| BM25 + RAPTOR | **52.1%** | **27.0%** |
| BM25 alone | 49.9% | 26.5% |
| DPR + RAPTOR | **54.7%** | **32.2%** |
| DPR alone | 53.1% | 31.7% |

在**任意 retriever** 上面套一棵 RAPTOR 树都能提升 —— 涨幅稳定、温和、与具体 retriever 无关。

**搭 GPT-4 时刷的 SOTA**：

| 数据集 | 前 SOTA | RAPTOR + GPT-4 | Δ |
|---|---|---|---|
| QASPER (F1) | CoLT5-XL 53.9% | **55.7%** | +1.8 |
| QuALITY (Acc) | CoLISA 62.3% | **82.6%** | **+20.3** |
| QuALITY-HARD | CoLISA 54.7% | **76.2%** | **+21.5** |
| NarrativeQA (METEOR) | Izacard&Grave 11.1 | **19.1** *(配 UnifiedQA)* | +8.0 |

QuALITY 上的数字是这篇论文的代表性成果 —— 在一个"多步推理多选题"上**绝对值 +20 个点**，HARD 子集 +21.5 点。正是这个数把 RAPTOR 钉进了之后每一篇 RAG 架构论文的 related work 章节。

**层数消融实验是方法论站得住脚的关键**（论文 Table 8）。同一个 QuALITY 故事上：只查叶子节点拿 57.9%；从第 2 层往下查 2 层拿 63.15%；从第 2 层往下查全部 3 层拿 **73.68%**。**整棵树确实在做事**，不是叶子信息的花式重复。

## 总结

RAPTOR 把一个具体的主张表达得很干净：**检索应当能选择抽象层级**。机制本身（递归 cluster + 摘要）机械上不复杂，但每个设计选择都有目的：

- **软聚类**（一个 chunk 可以进多个摘要）—— 承认真实文本的多主题性
- **GMM + UMAP + BIC** —— 站得住脚的概率框架，不是拍脑袋数字
- **Collapsed-tree 检索** —— 让 query 决定层级混合，不预先固定

实证站得住：QuALITY 上比前 SOTA 多 20 个点，这种跃迁级数才会被 ICLR 收、之后被反复引。

**这篇论文向前指了一条路**：RAPTOR 在文本上构建了**层级**，但这个层级是 **chunk-cluster 之上的树**。下一步自然的问题是 —— 如果底层不是扁平的 embedding 空间，而是一张**实体-关系图**呢？把"chunk-cluster 树"换成"LLM 抽取的知识图谱 + 社区检测层级"，这就是三个月后的 Microsoft GraphRAG。而一旦有了那张图，下一个论断（再六个月之后的 LightRAG）就是：图可以保留，社区摘要那层可以全砍掉。

所以三篇放一起读：**RAPTOR 是"摘要整个语料"的极**，GraphRAG 是"图 + 摘要"的中间态，LightRAG 是"只剩图"的另一极。按这个顺序读会发现 —— 2024 年这个领域真正的问题不是"要不要分层"，而是"分层里 LLM 摘要那层和底下的结构层，到底谁在干活？"RAPTOR 给的实证基线是：即便摘要本身有用，**完整树**比任何单层都更有用。后面所有论文都得在这个基线上往前走。

代码开源在 [github.com/parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)。Stanford NLP，Manning 组出品，代码质量和实验可复现性都靠谱。
