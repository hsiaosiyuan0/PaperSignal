# TrumorGPT — Research Brief

## Background

Health misinformation moves faster than fact-checkers can keep up with — the WHO formally declared a global infodemic during COVID-19, and generative AI has since added a steady supply of plausible-sounding fake news (NewsGuard tracked 614 unreliable AI-generated news sites across 15 languages by 2023). Manual fact-checking is too slow, and the obvious automation pieces each have a known failure mode: LLMs hallucinate or repeat outdated training data, and knowledge graphs lag behind whatever happened last week.

The paper's framing is direct: the failure modes of LLMs and KGs are complementary, so combine them — let the KG provide grounded, refreshable facts and let the LLM handle the language understanding. The application focus is health-related claim verification, the domain where being wrong has the highest direct cost.

## Approach

TrumorGPT is GPT-4 driven by a **Graph-based Retrieval-Augmented Generation (GraphRAG)** pipeline over *semantic health knowledge graphs*. Three pieces do most of the work.

**Topic-Enhanced Sentence Centrality.** To build a KG from an article, you first need to find the article's spine. They train a domain-specific LDA model to produce a per-sentence topic distribution, normalize it, and concatenate with a normalized BERT embedding:

$$v_s = [\eta \tilde{e}_s ; (1-\eta) \tilde{t}_s]$$

with $\eta=0.7$ giving semantics slightly more weight than topic. Cosine similarity between these vectors defines a sentence graph; PageRank picks the topically-aware central sentences.

**Topic-Specific TextRank (TST).** A modified PageRank in which both the teleportation distribution and the edge weights are biased toward a topic relevance score $R(v_i)$. Health-related topics get $\beta_k = \alpha = 1.5$ in the relevance score, while edge weights are averaged with adjacent vertex relevance:

$$w'_{j,i} = \frac{R(v_i) + R(v_j)}{2} \cdot w_{j,i}$$

The paper proves the resulting Markov chain is irreducible and aperiodic and bounds the convergence rate by $C \cdot d^t \cdot |\lambda_2(\mathbf{P})|^t$. With $d=0.85$ and a typical graph spectrum, it converges in tens of iterations.

**GraphRAG retrieval.** A user query is converted into a query KG $G_x$, then scored against each KG in the knowledge base using a weighted Jaccard over consecutive triples:

$$S(G_x, G_i) = \frac{\sum_{t \in T_x \cap T_i} f(t)}{\sum_{t \in T_x \cup T_i} f(t)}$$

A match above threshold returns True/False; no match returns Undetermined plus retrieved context for the user to judge. The KB is built from DBpedia's latest-core RDF triples filtered to health-related entities, which sidesteps GPT-4's December 2023 knowledge cutoff.

## Comparison

On 600 PolitiFact statements (300 true, 300 false) in Health Care and Coronavirus:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| GPT-3.5 | 72.7% | 75.8% | 66.7% | 70.9% |
| PaLM 2 | 76.8% | 77.9% | 75.0% | 76.4% |
| Claude 3.5 Sonnet | 77.2% | 78.4% | 75.0% | 76.7% |
| Gemini 1.5 | 81.7% | 83.9% | 78.3% | 81.0% |
| LLaMA 3.2 | 81.8% | 83.0% | 80.0% | 81.5% |
| GPT-4 | 83.3% | 85.7% | 80.0% | 82.8% |
| **TrumorGPT** | **88.5%** | **91.4%** | **85.0%** | **88.1%** |

The +5.2 point gain over its own GPT-4 backbone is the load-bearing result — it isolates what GraphRAG adds on top of a strong general-purpose LLM. TrumorGPT is also the most concise (avg 2.8 sentences per response), suggesting the KG retrieval surfaces precisely the relevant facts rather than padding with distractors.

In the discussion the authors briefly contrast their pure GraphRAG with **HybridRAG** (dual vector+graph retrieval, broader coverage but more compute) and **LightRAG** (lighter graph indexing, faster but shallower reasoning). Their position is that for high-stakes, structured domains like health, the graph-only path with curated semantic KGs gives better verifiable grounding than text-blob retrieval.

A real limitation: on the original PolitiFact 6-class rubric (True / Mostly True / Half True / Mostly False / False / Pants on Fire), accuracy drops to 49.3%. The binary collapse is doing a lot of work. Knowledge graph size also scales linearly with article length, and error rate rises with longer articles — the model struggles to maintain centrality on long inputs.

## Summary

The interesting move is recognizing that LLM hallucinations and KG staleness fail in different directions and constraining them against each other through GraphRAG. Most of the technical content (topic-enhanced centrality, TST with convergence bounds, weighted Jaccard scoring) is standard graph algorithms with the topic-aware weighting trick applied carefully — nothing exotic, which is part of why it works.

Worth noting from a reading-the-room perspective: by 2025, this hybrid pattern (LLM + structured KG retrieval) has effectively become the default for high-trust verticals (medicine, finance, legal). TrumorGPT is one well-engineered instance, not a paradigm shift. The value is in the careful end-to-end execution, especially the DBpedia-derived auto-refreshing KB, which addresses the "your KG is stale" failure mode that most academic GraphRAG papers wave at without solving.

Open questions the paper doesn't quite answer: how much of the 5-point gain over plain GPT-4 comes from the topic-enhanced centrality vs. the GraphRAG retrieval itself? An ablation would have helped. And the 49% multi-class accuracy hints that the system has learned to detect *direction* (true-leaning vs false-leaning) more than degree — a useful caveat for anyone deploying this kind of system in practice.
