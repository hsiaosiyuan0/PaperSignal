# From Local to Global: A GraphRAG Approach to Query-Focused Summarization — Research Brief

## Background

Vector RAG — the canonical "embed-chunks-and-retrieve-top-k" pipeline — answers *local* questions well: anything where the answer lives in a small, identifiable subset of the corpus. Where it falls apart is on **global sensemaking** questions, the kind that require reasoning over an entire dataset: "What are the main themes here?", "How do these documents collectively view X?", "What patterns connect these events?".

The paper frames this with a sharp diagnosis. Vector RAG can't answer global questions because no individual record contains the answer — the answer is a property of the *collection*. Earlier query-focused summarization (QFS) methods could handle the conceptual task but don't scale past tens of documents. Meanwhile, LLM context windows have grown enough that you might naïvely stuff a corpus in — but at million-token scale that's neither affordable nor reliable (degraded recall in long contexts is well-documented). The space between these failure modes is what GraphRAG is built to occupy.

This paper is, in retrospect, the work that *named* and *defined* what the community now calls "the GraphRAG paradigm." Most of the 2024-2025 GraphRAG ecosystem (LightRAG, MedGraphRAG, StructRAG, the entire DEEP-PolyU survey taxonomy) is structured as variations on or critiques of what this paper proposed.

## Approach

GraphRAG converts a text corpus into a hierarchical, queryable artifact in five stages:

**❶ Chunks → Entities & Relationships.** Documents are split into 600-token chunks (with 100-token overlap). An LLM is prompted with few-shot exemplars to extract typed entities, the relationships between them, and short descriptions of each. *Claims* — verifiable factual statements about entities — are extracted in a parallel pass. Domain tuning happens by swapping out the exemplars, not by retraining.

**❷ Entities → Knowledge Graph.** Duplicates across chunks are merged (exact string match in their implementation, but soft matching works too). Edge weights record how often a relationship was extracted; descriptions are aggregated per node/edge. Resulting graphs in their experiments: Podcast dataset → 8,564 nodes / 20,691 edges; News dataset → 15,754 nodes / 19,520 edges.

**❸ Graph → Communities.** Hierarchical **Leiden** community detection partitions the graph at multiple levels (C0 root → C3 leaf), giving a mutually exclusive, collectively exhaustive coverage of every node at each level. This is the move that makes "global → local" navigation possible.

**❹ Communities → Community Summaries.** Bottom-up summarization: leaf communities are summarized first; higher-level summaries are recursively built from their children's summaries when token limits force compression. The result is a hierarchy of report-like summaries you can scan top-down to understand the corpus *without any query at all*.

**❺ Query → Global Answer (map-reduce).** Given a user query:
- *Map*: each community summary independently generates a partial answer plus a 0-100 helpfulness score.
- *Reduce*: partial answers are sorted by helpfulness, packed into one context window, and the LLM produces the final global answer.

For evaluation, they also propose an **adaptive benchmarking** method: persona-prompted question generation (5 personas × 5 tasks × 5 questions = 125 questions per corpus), avoiding the trap of generating questions directly from the corpus they're testing on.

## Comparison

Head-to-head LLM-as-judge against six conditions on two ~1M-token corpora (Podcast transcripts, News articles):

| Conditions compared | Comprehensiveness | Diversity | Empowerment | Directness |
|---|---|---|---|---|
| Any GraphRAG (C0–C3) vs Vector RAG (SS) | **72–83%** win | **62–82%** win | mixed | loses (SS wins on directness) |
| GraphRAG C1–C3 vs Text Summarization (TS) | slight win | slight win | mixed | mixed |
| Root-level community (C0) | tokens per query: **9× to 43× fewer than TS** | — | — | — |

The win-rate result is the load-bearing finding: against conventional vector RAG on global sensemaking, GraphRAG isn't 5% better, it's preferred 3-out-of-4 times on what users actually want from these questions (comprehensiveness, diversity). Vector RAG still wins on "directness" because it produces shorter, more focused answers — which the authors deliberately track as a control variable, since you'd expect any verbose-but-thorough approach to lose on conciseness.

Within GraphRAG itself, **C0 (root-level community summaries) is the sweet spot**: it dominates token cost (9-43× cheaper than text-only map-reduce) while being competitive on quality with deeper levels. The hierarchy isn't just an artifact — different levels are genuinely operationally different products, and you should pick C0 for cheap-and-good, C2/C3 for detail-heavy queries.

Important honest disclosure in the paper: graph indexing for the Podcast dataset took **281 minutes** on the experimental rig with a public OpenAI endpoint. This is the trade — heavy one-time indexing cost in exchange for cheap, high-quality queries afterwards. It's the single biggest critique LightRAG and successors have leveled at this paper.

## Summary

This paper is the right one to read if you want to internalize what "modern GraphRAG" actually means. Four things stand out:

1. **The framing is the breakthrough**, not the engineering. Naming the gap between vector RAG (local) and QFS (doesn't scale) and proposing community-summary hierarchies as the bridge is what changed the field. The individual techniques (LLM entity extraction, Leiden, map-reduce) are well-known on their own.

2. **The community hierarchy is a real product surface**, not an implementation detail. C0 root summaries can be browsed without queries — they're effectively a structured overview of a corpus, which has independent utility beyond answering questions.

3. **Adaptive benchmarking is methodologically interesting**: persona-driven question synthesis sidesteps the absence of ground-truth answers for global queries, and the LLM-as-judge framework with directness-as-control is the right shape for evaluating these systems honestly.

4. **The indexing cost is real and is the central tension** in everything that came after. LightRAG drops community summaries for double-level keyword indexing because of this; StructRAG routes between structures partly to avoid it; MedGraphRAG narrows to domain KGs. Read this paper, then read LightRAG to see what falls out when you take the cost critique seriously.

Released open-source at [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag), and incorporated into LangChain, LlamaIndex, NebulaGraph, Neo4J extensions — the canonical reference implementation for the paradigm.
