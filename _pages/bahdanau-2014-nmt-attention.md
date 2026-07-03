---
permalink: /notes/bahdanau-2014-nmt-attention/
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
date: 2026-07-03
author_profile: true
tags:
  - NLP
  - Neural Machine Translation
  - Attention
  - Sequence-to-Sequence
paperurl: "https://arxiv.org/abs/1409.0473"
citation: "Bahdanau, D., Cho, K., & Bengio, Y. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015 oral."
---

{% include toc %}

<style>
  .paper-note-figure {
    margin: 2rem 0;
    text-align: center;
  }

  .paper-note-figure img {
    max-width: 100%;
    height: auto;
    padding: 0.75rem;
    border: 1px solid var(--global-border-color);
    background: #fff;
  }

  .paper-note-figure--narrow img {
    max-width: 280px;
  }

  .paper-note-figure--medium img {
    max-width: 680px;
  }

  .paper-note-figure-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    align-items: start;
  }

  .paper-note-figure-grid img {
    width: 100%;
  }

  .paper-note-panel-label {
    margin-top: 0.25rem;
    font-size: 0.85em;
    color: var(--global-text-color);
  }

  .paper-note-figure figcaption {
    margin-top: 0.75rem;
    font-size: 0.85em;
    color: var(--global-text-color);
    text-align: left;
  }

  .page__content {
    position: relative;
  }

  @media (min-width: 1280px) {
    .page__content > .sidebar__right {
      position: absolute !important;
      top: 0;
      right: -14.5rem;
      float: none !important;
      width: 13rem !important;
      margin: 0 !important;
    }
  }

  @media (max-width: 1279px) {
    .page__content > .sidebar__right {
      float: none !important;
      width: auto !important;
      margin: 0 0 1.5rem !important;
    }
  }

  @media (max-width: 700px) {
    .paper-note-figure-grid {
      grid-template-columns: 1fr;
    }
  }
</style>

## One-Line Takeaway

This paper introduces a neural machine translation model that learns **where to look in the source sentence while generating each target word**, removing the fixed-length context bottleneck of early encoder-decoder models.

## Metadata

| Item | Detail |
|------|--------|
| Paper | Neural Machine Translation by Jointly Learning to Align and Translate |
| Authors | Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio |
| Venue | ICLR 2015 oral presentation |
| First arXiv submission | 2014-09-01 |
| Task | English-to-French neural machine translation |
| Core keywords | RNN Encoder-Decoder, soft alignment, attention, BiRNN encoder, RNNsearch |
| Links | [arXiv](https://arxiv.org/abs/1409.0473) · [HTML paper](https://ar5iv.labs.arxiv.org/html/1409.0473) |

## Why I Read This

This is one of the papers that made **attention** a central idea in deep learning. Before reading Transformer-style self-attention, it is useful to understand why attention was introduced in sequence-to-sequence translation: early NMT models compressed the whole source sentence into one vector, and that single vector became a bottleneck, especially for long sentences.

## Problem

Earlier encoder-decoder NMT models encode the entire source sentence into one fixed-length vector <math><mi>c</mi></math>. The decoder then generates every target word from that same compressed representation.

The paper argues that this design is weak because:

- Long sentences contain too much information to store reliably in one vector.
- The decoder needs different source words at different generation steps.
- Translation often requires soft, non-monotonic alignment rather than a single hard word mapping.

## Key Idea

Instead of forcing the encoder to produce one sentence vector, the encoder produces a sequence of annotations:

<math display="block">
  <msub><mi>h</mi><mn>1</mn></msub>
  <mo>,</mo>
  <msub><mi>h</mi><mn>2</mn></msub>
  <mo>,</mo>
  <mo>...</mo>
  <mo>,</mo>
  <msub><mi>h</mi><msub><mi>T</mi><mi>x</mi></msub></msub>
</math>

At each target step <math><mi>i</mi></math>, the decoder computes attention weights over all source annotations and builds a step-specific context vector:

<math display="block">
  <msub><mi>c</mi><mi>i</mi></msub>
  <mo>=</mo>
  <munderover>
    <mo>&#x2211;</mo>
    <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow>
    <msub><mi>T</mi><mi>x</mi></msub>
  </munderover>
  <msub><mi>&#x03B1;</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub>
  <msub><mi>h</mi><mi>j</mi></msub>
</math>

The model therefore decides which source positions are relevant **for the current target word**, not once for the entire sentence. See **Figure 1** below for the original model diagram from the paper.

## Method

### Encoder

The encoder is a bidirectional RNN. For each source position <math><mi>j</mi></math>, it concatenates the forward and backward hidden states:

<math display="block">
  <msub><mi>h</mi><mi>j</mi></msub>
  <mo>=</mo>
  <mo>[</mo>
  <msub><mover><mi>h</mi><mo>&#x2192;</mo></mover><mi>j</mi></msub>
  <mo>;</mo>
  <msub><mover><mi>h</mi><mo>&#x2190;</mo></mover><mi>j</mi></msub>
  <mo>]</mo>
</math>

This gives each source-word annotation information from both left and right context.

### Alignment Model

For each target step <math><mi>i</mi></math> and source position <math><mi>j</mi></math>, the model computes an alignment score:

<math display="block">
  <msub><mi>e</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub>
  <mo>=</mo>
  <mi>a</mi>
  <mo>(</mo>
  <msub><mi>s</mi><mrow><mi>i</mi><mo>-</mo><mn>1</mn></mrow></msub>
  <mo>,</mo>
  <msub><mi>h</mi><mi>j</mi></msub>
  <mo>)</mo>
</math>

Here, <math><msub><mi>s</mi><mrow><mi>i</mi><mo>-</mo><mn>1</mn></mrow></msub></math> is the previous decoder hidden state and <math><mi>a</mi><mo>(</mo><mo>&#x22C5;</mo><mo>)</mo></math> is a learned feed-forward network. The scores are normalized with softmax:

<math display="block">
  <msub><mi>&#x03B1;</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub>
  <mo>=</mo>
  <mfrac>
    <mrow><mi>exp</mi><mo>(</mo><msub><mi>e</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>)</mo></mrow>
    <mrow>
      <munderover>
        <mo>&#x2211;</mo>
        <mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow>
        <msub><mi>T</mi><mi>x</mi></msub>
      </munderover>
      <mi>exp</mi><mo>(</mo><msub><mi>e</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><mo>)</mo>
    </mrow>
  </mfrac>
</math>

These <math><msub><mi>&#x03B1;</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></math> values are differentiable soft-alignment weights, so the alignment mechanism can be trained jointly with the translation model by backpropagation.

### Decoder

The decoder predicts the next target word using:

- the previous target word,
- the previous decoder state,
- the current context vector <math><msub><mi>c</mi><mi>i</mi></msub></math>.

This differs from the basic encoder-decoder because every decoding step receives a different context vector.

## Paper Figures and Tables

I cite and render the original paper's figures rather than recreating them.

<figure class="paper-note-figure paper-note-figure--narrow">
  <img src="/assets/images/paper-notes/bahdanau-2014/figure-1-model.png" alt="Figure 1 from Bahdanau et al. showing the proposed attention-based neural machine translation model">
  <figcaption>
    <strong>Figure 1.</strong> Original model diagram from Bahdanau et al. showing the proposed attention-based decoder. Source: <a href="https://ar5iv.labs.arxiv.org/html/1409.0473#S3.F1">ar5iv rendering of arXiv:1409.0473</a>.
  </figcaption>
</figure>

<figure class="paper-note-figure paper-note-figure--medium">
  <img src="/assets/images/paper-notes/bahdanau-2014/figure-2-bleu-by-length.png" alt="Figure 2 from Bahdanau et al. showing BLEU scores by source sentence length">
  <figcaption>
    <strong>Figure 2.</strong> Original BLEU-by-sentence-length plot. This is the main evidence that RNNsearch handles long sentences better than the fixed-vector encoder-decoder. Source: <a href="https://ar5iv.labs.arxiv.org/html/1409.0473#S4.F2.fig1">ar5iv rendering of arXiv:1409.0473</a>.
  </figcaption>
</figure>

<figure class="paper-note-figure">
  <div class="paper-note-figure-grid">
    <div>
      <img src="/assets/images/paper-notes/bahdanau-2014/figure-3a-alignment.png" alt="Figure 3a from Bahdanau et al. showing a sample soft alignment matrix">
      <div class="paper-note-panel-label">(a)</div>
    </div>
    <div>
      <img src="/assets/images/paper-notes/bahdanau-2014/figure-3b-alignment.png" alt="Figure 3b from Bahdanau et al. showing a sample soft alignment matrix">
      <div class="paper-note-panel-label">(b)</div>
    </div>
    <div>
      <img src="/assets/images/paper-notes/bahdanau-2014/figure-3c-alignment.png" alt="Figure 3c from Bahdanau et al. showing a sample soft alignment matrix">
      <div class="paper-note-panel-label">(c)</div>
    </div>
    <div>
      <img src="/assets/images/paper-notes/bahdanau-2014/figure-3d-alignment.png" alt="Figure 3d from Bahdanau et al. showing a sample soft alignment matrix">
      <div class="paper-note-panel-label">(d)</div>
    </div>
  </div>
  <figcaption>
    <strong>Figure 3.</strong> Original RNNsearch-50 soft-alignment examples. The axes correspond to source English words and generated French words; brighter cells indicate larger attention weights. Source: <a href="https://ar5iv.labs.arxiv.org/html/1409.0473#S4.F3">ar5iv rendering of arXiv:1409.0473</a>.
  </figcaption>
</figure>

The quantitative table below cites **Table 1** values directly from the paper.

## Experiments

The authors evaluate on WMT 2014 English-to-French translation. They compare a basic RNN encoder-decoder with the proposed attention-based model, called **RNNsearch**.

Values cited from **Table 1** of the paper:

| Model | BLEU, all sentences | BLEU, no UNK |
|------|---------------------:|-------------:|
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* | 28.45 | 36.15 |
| Moses phrase-based SMT | 33.30 | 35.63 |

The important result is not only the overall BLEU improvement. As shown in **Figure 2** of the paper, the attention model is much more robust for longer sentences, while the fixed-vector encoder-decoder degrades as sentence length increases.

## What I Learned

The main contribution is the shift from **sentence-level compression** to **step-level retrieval**. The encoder stores a sequence of useful representations, and the decoder retrieves what it needs at each generation step.

This also makes the model more interpretable than a plain encoder-decoder. **Figure 3** in the paper visualizes attention weights as source-target alignment matrices, which helps explain how the translation was produced.

For my Transformer study, this paper is a good bridge:

- Bahdanau attention uses an RNN decoder state to attend over encoder states.
- Transformer attention removes recurrence and uses attention as the main computation.
- The motivation remains similar: let the model select relevant context instead of relying on one compressed representation.

## Implementation Notes

If I implement this in PyTorch, I would split the model into four modules:

1. `Encoder`: bidirectional GRU/LSTM that returns all hidden states.
2. `AdditiveAttention`: feed-forward alignment network for <math><msub><mi>e</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></math>.
3. `Decoder`: recurrent decoder that consumes <math><msub><mi>c</mi><mi>i</mi></msub></math> at each step.
4. `Seq2Seq`: training wrapper with teacher forcing and beam-search inference.

Important checks:

- Verify attention weights sum to 1 over source positions.
- Compare attention weights against the qualitative alignment examples in Figure 3 of the paper.
- Compare short-sentence and long-sentence performance separately.
- Track unknown-token behavior, because rare words remain a key limitation in this paper.

## Limitations and Questions

- The shortlist vocabulary maps rare words to `UNK`, so rare-word translation is still weak.
- Attention requires computing source-target pair scores, which scales with both input and output length.
- Attention heatmaps are useful diagnostics, but they should not automatically be treated as a complete explanation of model behavior.
- The paper focuses on English-to-French; I want to compare whether the same behavior appears in Korean-English or lower-resource settings.

## Follow-Up Reading

- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- Sutskever et al., "Sequence to Sequence Learning with Neural Networks"
- Vaswani et al., "Attention Is All You Need"

## Reference

- [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- [HTML version via ar5iv](https://ar5iv.labs.arxiv.org/html/1409.0473)
