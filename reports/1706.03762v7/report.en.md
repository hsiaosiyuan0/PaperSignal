# Attention Is All You Need — Research Brief

## Background

The dominant approach to sequence transduction in 2017 — translation, summarization, speech recognition — relied on recurrent neural networks (LSTMs, GRUs) or convolutional architectures. RNNs forced computation to be **sequential along the input**: hidden state $h_t$ depends on $h_{t-1}$, which is fundamentally at odds with modern hardware's parallelism. This made long sequences slow to train and harder to scale.

Attention mechanisms had become a standard ingredient *alongside* recurrence, used mainly to bridge long distances in the input. But every model still kept the recurrent backbone, and so kept its bottleneck.

The paper's question is direct: **can we throw away the recurrence entirely and let attention carry the whole load?**

## Approach

The Transformer is built from three primitives, stacked into encoder–decoder layers:

1. **Scaled dot-product attention**. Given queries $Q$, keys $K$, values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor $1/\sqrt{d_k}$ keeps softmax gradients in a usable range as $d_k$ grows.

2. **Multi-head attention**. Project $Q$, $K$, $V$ into $h$ different subspaces, run attention in each, concatenate the results. This lets the model attend to different positions and different representational subspaces simultaneously, instead of averaging them into a single distribution.

3. **Positional encodings**. Without recurrence, the model is permutation-equivariant. The fix is a sinusoidal positional signal added to the input embedding:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

The choice of sinusoid (rather than learned) lets the model extrapolate to lengths longer than seen in training.

The encoder is six identical layers of [self-attention → feed-forward]. The decoder mirrors it with an added cross-attention sub-layer and masked self-attention so positions can't attend to the future.

## Comparison

| Layer Type | Complexity per layer | Sequential ops | Max path length |
|------------|----------------------|----------------|-----------------|
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| **Self-attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |

The trade-off is explicit: self-attention pays $n^2$ in compute but collapses the maximum dependency path between any two positions to $O(1)$, while staying perfectly parallelizable. For typical sentence lengths ($n < d$), this is cheaper than RNN per layer too.

Against the best 2017 models on WMT 2014 EN→DE: Transformer (big) reaches **28.4 BLEU**, beating the previous best ensemble by ~2 BLEU, while training in 3.5 days on 8 P100s — a fraction of competitors' cost.

## Summary

The Transformer's importance was clear in retrospect but not obvious at the time: a clean architectural simplification that turned out to be the right substrate for everything downstream — BERT, GPT, ViT, CLIP, the entire modern foundation-model stack. The mechanical contribution (scaled dot-product attention + multi-head + positional encoding) is small; the **conceptual move** — refusing to keep recurrence as a comfort blanket — is what mattered.

What still surprises on re-reading: how *little* tuning was needed. Three primitives, a clean residual stack, layer normalization. Most of the "tricks" are interpretability conveniences, not load-bearing.

Worth holding in mind: every modern attention variant (flash-attention, sliding window, MQA / GQA, rotary positional encoding) is a delta against this 2017 baseline.
