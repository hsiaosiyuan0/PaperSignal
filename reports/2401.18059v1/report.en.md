# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval — Research Brief

## Background

RAPTOR (Stanford, Christopher Manning's group, ICLR 2024) was published a few months before Microsoft GraphRAG and pre-figures the same diagnosis: standard retrieval-augmented LMs fetch a handful of short, contiguous chunks, which is a structural mismatch for questions that require integrating information across a long document. The paper's running example is the Cinderella fairy tale and the question *"How did Cinderella reach her happy ending?"* — no top-k of 100-token snippets contains the answer; you need the *arc* of the story.

The contribution is the first widely-adopted answer to "what would multi-level retrieval look like": recursively cluster text chunks, summarize each cluster, recurse on the summaries, until you can't cluster anymore. The output is a tree where leaves are the original text and every higher layer is an LLM-generated abstraction of its children. At query time you retrieve from *all layers at once*.

Read alongside the rest of the GraphRAG arc, RAPTOR is the **tree-based predecessor**: the hierarchical-summary idea that GraphRAG later transposes onto a graph (community summaries instead of cluster summaries), and that LightRAG eventually argues you can drop entirely.

## Approach

**Tree construction is the core idea.** Three nested loops over the corpus, repeated until convergence:

**❶ Chunk + embed.** Split the corpus into ~100-token contiguous chunks (sentences kept whole — if a sentence would push past 100 tokens, the whole sentence moves to the next chunk). Embed each with SBERT (`multi-qa-mpnet-base-cos-v1`). These become the **leaf layer**.

**❷ Cluster.** Soft clustering with Gaussian Mixture Models on the embeddings. Two implementation choices matter:

- **UMAP** for dimensionality reduction before GMM, because high-dim Euclidean distance degrades. Vary UMAP's `n_neighbors` to do *two-pass* clustering — first global clusters, then local clustering inside each global cluster.
- **BIC** (Bayesian Information Criterion) to pick the cluster count automatically. EM solves the resulting GMM.

Soft clustering matters: a chunk can belong to multiple clusters, which matches the reality that a paragraph can carry several topics. If a cluster's combined context exceeds the summarization model's token limit, recurse with a tighter UMAP locally.

**❸ Summarize and recurse.** Each cluster's chunks are sent to `gpt-3.5-turbo`, which produces a single text summary. That summary becomes a new node, embedded by SBERT, and joins the next layer up. Repeat the cluster-and-summarize cycle on the new layer. Stop when clustering becomes infeasible.

A focused annotation study found **~4% of summaries contain minor hallucinations** that **do not propagate to parent nodes** and have no measurable downstream effect — useful empirical evidence for "summarize-then-summarize" being safer than feared.

**Two querying strategies** sit on top of the tree. They are evaluated head-to-head, and the simpler one wins:

- **Tree traversal.** Start at the root. Take top-k by cosine similarity. Descend into those nodes' children, top-k again. Repeat for $d$ layers. Concatenate.
- **Collapsed tree.** Flatten the entire tree into one bag of nodes. Cosine-rank against the query. Take top nodes until you hit a token budget (they use 2000 tokens, ≈ top-20).

The collapsed-tree approach consistently wins. The reason the authors give is structural: tree traversal has a *fixed* ratio of nodes-per-layer in its result, so the abstract/detail mix is decided by the topology rather than by the question. Collapsed tree lets the question's level of granularity decide which layer's nodes get retrieved.

## Comparison

Three datasets, picked specifically to stress long-document reasoning: **NarrativeQA** (books and movie scripts, full text), **QASPER** (questions over full NLP papers), **QuALITY** (multiple-choice over ~5,000-token passages, plus a HARD subset where humans fail under time pressure).

**Controlled retrieval-only ablation (UnifiedQA-3B as reader, SBERT/BM25/DPR ± RAPTOR):**

| Reader: UnifiedQA-3B | QuALITY Acc | QASPER F1 |
|---|---|---|
| SBERT + RAPTOR | **56.6%** | **36.7%** |
| SBERT alone | 54.9% | 36.2% |
| BM25 + RAPTOR | **52.1%** | **27.0%** |
| BM25 alone | 49.9% | 26.5% |
| DPR + RAPTOR | **54.7%** | **32.2%** |
| DPR alone | 53.1% | 31.7% |

Adding the RAPTOR tree on top of *any* retriever improves it — the gain is consistent, modest, retriever-agnostic.

**State-of-the-art when paired with GPT-4:**

| Dataset | Prev. SOTA | RAPTOR + GPT-4 | Δ |
|---|---|---|---|
| QASPER (F1) | CoLT5-XL 53.9% | **55.7%** | +1.8 |
| QuALITY (Acc) | CoLISA 62.3% | **82.6%** | **+20.3** |
| QuALITY-HARD | CoLISA 54.7% | **76.2%** | **+21.5** |
| NarrativeQA (METEOR) | Izacard&Grave 11.1 | **19.1** *(w/ UnifiedQA)* | +8.0 |

The QuALITY result is the headline — a 20-point absolute jump on a multi-step-reasoning multiple-choice benchmark, and 21.5 points on the human-hard subset. This is the result that put RAPTOR on every subsequent RAG-architecture paper's related-work section.

**The tree-layer ablation is what makes the methodology defensible** (Table 8 in the paper). On a single QuALITY story, querying only leaf nodes gives 57.9%; querying 2 layers from layer 2 gives 63.15%; querying all 3 layers from layer 2 gives **73.68%**. The full tree is doing real work; it isn't just a clever way of duplicating leaf information.

## Summary

RAPTOR is the cleanest statement of one specific idea: **retrieval should be able to choose the level of abstraction**. The mechanism — recursive cluster+summarize over text chunks — is mechanically simple, but the design choices are deliberate:

- **Soft clustering** (a chunk can be in multiple summaries) — handles the messiness of real text
- **GMM + UMAP + BIC** — a defensible probabilistic stack, not magic numbers
- **Collapsed-tree retrieval** — let the question decide the layer mix, don't preordain it

The empirical case is strong: +20 points on QuALITY against the previous SOTA is the kind of jump that gets a paper into ICLR and stays cited.

Where the paper points forward: RAPTOR builds a *hierarchy* on text, but the hierarchy is a *tree* over *clusters of chunks*. The natural next move is to ask: what if the lower level were an entity-relation **graph** rather than a flat embedding space? That move — replacing the chunk-cluster tree with an LLM-extracted knowledge graph and community-detection-based hierarchy — is exactly Microsoft's GraphRAG, three months later. And once you have that graph, the next argument (LightRAG, six months after that) is that you can keep the graph and drop the community summaries entirely.

So in the three-paper arc, RAPTOR is the **"summarize-the-corpus" pole**, GraphRAG is the **"graph + summaries" middle**, and LightRAG is the **"only the graph" pole**. Reading them in order makes it clear that the field's question for 2024 was not "do we need hierarchy?" but "how much of the hierarchy is the LLM-generated summarization layer doing the work, vs the structural layer underneath?" RAPTOR's evidence — that even tree-organized recursive summaries help, but the *full tree* helps more than any single layer — is the empirical baseline that everything after has to beat.

Open source at [github.com/parthsarthi03/raptor](https://github.com/parthsarthi03/raptor). Stanford NLP, Manning group; expect clean code and reproducible experiments.
