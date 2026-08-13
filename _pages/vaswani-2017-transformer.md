---
permalink: /notes/vaswani-2017-transformer/
title: "Attention Is All You Need"
date: 2026-08-11
paper_note: true
hide_date: true
note_area: "NLP · Transformer"
note_summary: "Why self-attention can replace recurrence, and how multi-head attention and positional encoding build contextual token representations."
paper_authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin"
paper_venue: "NeurIPS 2017"
paper_date: 2017-06-12
author_profile: true
lang: en
locale: en-US
og_locale: en_US
author: jeo_en
translations:
  en: /notes/vaswani-2017-transformer/
  ko: /ko/notes/vaswani-2017-transformer/
tags:
  - NLP
  - Transformer
  - Attention
  - Sequence-to-Sequence
paperurl: "https://arxiv.org/abs/1706.03762"
htmlpaperurl: "https://arxiv.org/html/1706.03762v7"
proceedingsurl: "https://papers.nips.cc/paper/7181-attention-is-all-you-need"
citation: "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. Attention Is All You Need. NeurIPS 2017."
---

{% include toc %}
{% include paper-note-toc-layout.html %}
{% include paper-note-head.html %}
{% include paper-note-meta.html %}

## Introduction

The Transformer replaces recurrence and convolution with stacked self-attention, letting every token directly read from other tokens and making sequence modeling much more parallelizable.

RNN and convolutional sequence models have two important limitations:

- RNNs process tokens sequentially, so training is hard to parallelize across sequence positions.
- Long-range dependencies require long computational paths in RNNs and multiple layers in local convolutional models.
- Encoder-decoder attention helps translation, but earlier systems still depend on recurrent or convolutional encoders and decoders.

The paper asks whether a sequence transduction model can be built using attention alone, while preserving strong translation quality and improving parallel training efficiency.

The Transformer represents each token as a vector and repeatedly updates those vectors using self-attention and position-wise feed-forward networks. In encoder self-attention, every position can attend to every other source position. In decoder self-attention, masking prevents a position from attending to future target tokens.

For a sequence of representations $X$, each layer constructs queries, keys, and values from the previous layer's representations. Attention scores compare queries and keys, and the output is a weighted sum of values:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This is the core change from RNN-based modeling: the path between any two tokens in a full self-attention layer is constant.

<div class="paper-note-table-wrap" markdown="1">

| Layer type | Complexity per layer | Sequential operations | Maximum path length |
|------------|----------------------|-----------------------|---------------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

</div>

The table above cites **Table 1** values directly from the paper. It is the main argument for why self-attention is attractive for long-range dependencies: all positions can interact in one layer, while recurrence needs a path that grows with sequence length.

## Method

### Encoder-Decoder Structure

The Transformer keeps the encoder-decoder framework but changes the layer type. The encoder is a stack of $N=6$ identical layers. Each encoder layer has:

- multi-head self-attention,
- a position-wise feed-forward network,
- residual connections and layer normalization around both sublayers.

The decoder is also a stack of $N=6$ layers, but each decoder layer has three sublayers:

- masked multi-head self-attention over previous target positions,
- encoder-decoder attention over encoder outputs,
- a position-wise feed-forward network.

<figure class="paper-note-figure paper-note-figure--narrow">
  <img src="/assets/images/paper-notes/vaswani-2017/figure-1-transformer-architecture.png" alt="Figure 1 from Vaswani et al. showing the Transformer model architecture">
  <figcaption>
    <strong>Figure 1.</strong> Original Transformer model architecture. The left side is the encoder stack, and the right side is the autoregressive decoder stack.
  </figcaption>
</figure>

### Scaled Dot-Product Attention

The attention equation compares each query with all keys using dot products. The scaling factor $\sqrt{d_k}$ keeps the logits from becoming too large when $d_k$ is large:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Without scaling, dot products can grow in magnitude with the key dimension, which can push softmax into low-gradient regions. The scaling term is therefore a stability device for dot-product attention.

### Multi-Head Attention

Instead of applying one attention operation in the full $d_{\text{model}}$ space, the model learns $h$ different projections of $Q$, $K$, and $V$. Each head attends in its own lower-dimensional representation subspace:

$$
\mathrm{head_i}=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

The head outputs are concatenated and projected back into the model dimension:

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head_1},...,\mathrm{head_h})W^O
$$

**The output projection $W^O$ linearly recombines information extracted by different heads from different representation subspaces into one learned $d_{\text{model}}$-dimensional representation.** Without this projection, concatenation would merely place head outputs side by side; $W^O$ lets the next layer receive a learned mixture of those head-specific signals.

<figure class="paper-note-figure paper-note-figure--medium">
  <div class="paper-note-figure-grid">
    <div>
      <img src="/assets/images/paper-notes/vaswani-2017/figure-2-scaled-dot-product-attention.png" alt="Figure 2 left from Vaswani et al. showing scaled dot-product attention">
      <div class="paper-note-panel-label">Scaled Dot-Product Attention</div>
    </div>
    <div>
      <img src="/assets/images/paper-notes/vaswani-2017/figure-2-multi-head-attention.png" alt="Figure 2 right from Vaswani et al. showing multi-head attention">
      <div class="paper-note-panel-label">Multi-Head Attention</div>
    </div>
  </div>
  <figcaption>
    <strong>Figure 2.</strong> Original diagrams for scaled dot-product attention and multi-head attention.
  </figcaption>
</figure>

### Positional Encoding

Because the Transformer has no recurrence or convolution, token order must be injected explicitly. The paper adds positional encodings to input embeddings at the bottom of the encoder and decoder:

$$
PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})
$$

$$
PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{\text{model}}})
$$

The important detail is that the token embedding and positional encoding have the same dimension, $d_{\text{model}}$, so they can be added element-wise:

$$
x_{pos}=E(x_{pos})+PE_{pos}
$$

**The Transformer does not concatenate the input embedding and positional encoding, and it does not take a dot product between them. It adds them in the same vector space, preserving the model dimension while placing semantic information and positional information in one representation that self-attention can use.**

Concatenation would increase the representation size, for example from $d_{\text{model}}$ to $2d_{\text{model}}$, or require another projection before entering layers that expect $d_{\text{model}}$. A dot product would collapse two vectors into a scalar similarity, which is not useful as the token representation passed into the encoder or decoder. Element-wise addition keeps the residual stream dimension fixed and lets later learned projections decide how to use semantic and positional coordinates.

### Position-Wise Feed-Forward Network

Each layer also contains a feed-forward network applied independently to every position:

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

The same feed-forward parameters are shared across positions within a layer, but each layer has its own parameters. In the base model, $d_{\text{model}}=512$ and $d_{\text{ff}}=2048$.

## Experiments

### Machine Translation Results

The main evaluation is WMT 2014 English-German and English-French translation. The base Transformer uses $N=6$, $d_{\text{model}}=512$, $d_{\text{ff}}=2048$, $h=8$, and $d_k=d_v=64$. The big model increases model size and trains longer.

The quantitative table below cites **Table 2** values directly from the paper.

<div class="paper-note-table-wrap" markdown="1">

| Model | EN-DE BLEU | EN-FR BLEU | Reported training cost |
|-------|-----------:|-----------:|------------------------|
| ByteNet | 23.75 | - | - |
| Deep-Att + PosUnk | - | 39.2 | EN-FR: $1.0 \times 10^{20}$ |
| GNMT + RL | 24.6 | 39.92 | EN-DE: $2.3 \times 10^{19}$; EN-FR: $1.4 \times 10^{20}$ |
| ConvS2S | 25.16 | 40.46 | EN-DE: $9.6 \times 10^{18}$; EN-FR: $1.5 \times 10^{20}$ |
| MoE | 26.03 | 40.56 | EN-DE: $2.0 \times 10^{19}$; EN-FR: $1.2 \times 10^{20}$ |
| Deep-Att + PosUnk Ensemble | - | 40.4 | EN-FR: $8.0 \times 10^{20}$ |
| GNMT + RL Ensemble | 26.30 | 41.16 | EN-DE: $1.8 \times 10^{20}$; EN-FR: $1.1 \times 10^{21}$ |
| ConvS2S Ensemble | 26.36 | 41.29 | EN-DE: $7.7 \times 10^{19}$; EN-FR: $1.2 \times 10^{21}$ |
| Transformer (base model) | 27.3 | 38.1 | $3.3 \times 10^{18}$ |
| Transformer (big) | 28.4 | 41.8 | $2.3 \times 10^{19}$ |

</div>

The key result is that Transformer (big) reaches 28.4 BLEU on English-German and 41.8 BLEU on English-French, while using far less reported training compute than strong recurrent, convolutional, and ensemble baselines.

### Ablation Study

The paper's **Table 3** varies Transformer components on the English-German development set. These are `newstest2013` development BLEU scores, so they should not be compared directly with the Table 2 test scores.

| Configuration | Dev BLEU |
|---------------|---------:|
| Base: 8 heads | 25.8 |
| 1 head | 24.9 |
| 4 heads | 25.5 |
| 16 heads | 25.8 |
| 32 heads | 25.4 |
| Learned positional embedding | 25.7 |
| No label smoothing | 25.3 |

One head is 0.9 BLEU below the base model, but adding more heads does not monotonically improve quality. Learned positional embeddings perform almost identically to sinusoidal encoding, and removing label smoothing lowers BLEU. The ablation supports multi-head attention as a useful design, but it does not prove that every head learns a guaranteed linguistic role.

### Interpretation

The Transformer is not just "attention instead of RNN." It changes how sequence representations are built. In an RNN, information from distant tokens must travel through many recurrent steps. In a Transformer layer, every token can directly compare itself with every other token, and the model learns how much information to retrieve from each position.

The architecture also standardizes the residual stream around one dimension, $d_{\text{model}}$. Embeddings, positional encodings, sublayer outputs, residual connections, and multi-head attention outputs all live in that dimension. This design choice makes the model stackable.

## Questions from My Study

### 1. Does self-attention model all token relations from the first layer?

Yes, in terms of connectivity. From the first self-attention layer, every token can compute an attention score against every other token in the same sequence. This is why Table 1 lists the maximum path length of self-attention as $O(1)$.

But this does not mean the first layer already understands every high-level relation. In the first layer, $Q$, $K$, and $V$ are projected from token embeddings plus positional encodings. Deeper layers then repeat self-attention over representations that are already contextualized by previous layers.

So the layer stack works like this:

1. Early layers compute direct token-to-token interactions from relatively local lexical representations.
2. Middle layers recompute attention over contextualized representations created by earlier layers.
3. Later layers can combine more abstract syntactic, semantic, and task-specific relations.

In short, **the first layer can connect all tokens, and deeper layers repeatedly recalculate and compose relations among increasingly contextualized representations.**

### 2. Why add input embeddings and positional encodings instead of concatenating them?

The model needs each token representation to contain both what the token means and where it appears. Positional encoding supplies order information that pure self-attention does not naturally have.

The paper makes the positional encoding dimension equal to $d_{\text{model}}$, the same as the token embedding dimension. This allows element-wise addition:

$$
\text{input representation}=\text{token embedding}+\text{positional encoding}
$$

This is not concatenation and not a dot product. Concatenation would expand the hidden size or require another projection before the first encoder/decoder layer. A dot product would compress two vectors into one scalar, losing the vector representation needed by attention.

Addition keeps the model dimension unchanged, so the same representation can flow through residual connections and later projections. The learned $W^Q$, $W^K$, and $W^V$ matrices can then decide how to use the combined semantic and positional signal.

### 3. Why does multi-head attention need $W^O$ after concatenating heads?

Each head attends with its own learned projections, so each head can focus on different relation types, positions, or representation subspaces. After attention, the model concatenates the head outputs:

$$
\mathrm{Concat}(\mathrm{head_1},...,\mathrm{head_h})
$$

If the model stopped there, the result would be a block-wise concatenation of independent head outputs. The final projection $W^O$ learns how to mix those blocks.

That means $W^O$ is not just a shape-fixing layer. It is the learned integration step that turns multiple head-specific representations into one $d_{\text{model}}$ vector for the residual stream and the next layer.

## Reference

### Follow-Up Reading

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Radford et al., "Improving Language Understanding by Generative Pre-Training"
- Shaw et al., "Self-Attention with Relative Position Representations"
- Harvard NLP, "The Annotated Transformer"

### Sources

- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [HTML version via arXiv](https://arxiv.org/html/1706.03762v7)
- [NeurIPS proceedings page](https://papers.nips.cc/paper/7181-attention-is-all-you-need)
- [Google Research publication page](https://research.google/pubs/attention-is-all-you-need/)
