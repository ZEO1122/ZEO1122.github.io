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
{% include paper-note-toc-layout.html %}

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

  .page__content mjx-container[display="true"] {
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 0.15rem;
  }

  @media (max-width: 700px) {
    .paper-note-figure-grid {
      grid-template-columns: 1fr;
    }

    .masthead,
    .masthead__inner-wrap,
    .masthead__menu,
    .greedy-nav,
    .greedy-nav .visible-links {
      max-width: 100vw;
      overflow-x: hidden;
    }
  }
</style>

<script>
  window.MathJax = window.MathJax || {};
  window.MathJax.tex = window.MathJax.tex || {};
  window.MathJax.tex.inlineMath = [['$', '$'], ['\\(', '\\)']];
  window.MathJax.tex.displayMath = [['$$', '$$'], ['\\[', '\\]']];
</script>

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

The model therefore decides which source positions are relevant **for the current target word**, not once for the entire sentence.

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

<figure class="paper-note-figure paper-note-figure--narrow">
  <img src="/assets/images/paper-notes/bahdanau-2014/figure-1-model.png" alt="Figure 1 from Bahdanau et al. showing the proposed attention-based neural machine translation model">
  <figcaption>
    <strong>Figure 1.</strong> Original model diagram from Bahdanau et al. showing the proposed attention-based decoder. Source: <a href="https://ar5iv.labs.arxiv.org/html/1409.0473#S3.F1">ar5iv rendering of arXiv:1409.0473</a>.
  </figcaption>
</figure>

## Experiments

The authors evaluate on WMT 2014 English-to-French translation. They compare a basic RNN encoder-decoder with the proposed attention-based model, called **RNNsearch**.

The quantitative table below cites **Table 1** values directly from the paper.

| Model | BLEU, all sentences | BLEU, no UNK |
|------|---------------------:|-------------:|
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* | 28.45 | 36.15 |
| Moses phrase-based SMT | 33.30 | 35.63 |

The important result is not only the overall BLEU improvement. As shown in **Figure 2**, the attention model is much more robust for longer sentences, while the fixed-vector encoder-decoder degrades as sentence length increases.

<figure class="paper-note-figure paper-note-figure--medium">
  <img src="/assets/images/paper-notes/bahdanau-2014/figure-2-bleu-by-length.png" alt="Figure 2 from Bahdanau et al. showing BLEU scores by source sentence length">
  <figcaption>
    <strong>Figure 2.</strong> Original BLEU-by-sentence-length plot. This is the main evidence that RNNsearch handles long sentences better than the fixed-vector encoder-decoder. Source: <a href="https://ar5iv.labs.arxiv.org/html/1409.0473#S4.F2.fig1">ar5iv rendering of arXiv:1409.0473</a>.
  </figcaption>
</figure>

## What I Learned

The main contribution is the shift from **sentence-level compression** to **step-level retrieval**. The encoder stores a sequence of useful representations, and the decoder retrieves what it needs at each generation step.

This also makes the model more interpretable than a plain encoder-decoder because its attention weights can be inspected as source-target alignment matrices.

For my Transformer study, this paper is a good bridge:

- Bahdanau attention uses an RNN decoder state to attend over encoder states.
- Transformer attention removes recurrence and uses attention as the main computation.
- The motivation remains similar: let the model select relevant context instead of relying on one compressed representation.

## Questions I Worked Through

This section is not a PyTorch implementation checklist. It collects the questions I had while studying the paper and trying to connect the equations, the official GroundHog code, and a modern PyTorch-style reproduction.

### 1. How should I interpret soft-search and soft-alignment?

In this paper, **soft-search** means that the decoder does not rely on one fixed vector for the whole source sentence. Instead, at each target step it scans the source-side annotation sequence and assigns a soft relevance weight to every source position. Here, "search" does not mean choosing one source token with a hard decision. It means scoring all source positions and normalizing those scores with softmax.

At decoding step $i$, the alignment score is computed roughly as:

$$
e_{ij} = a(s_{i-1}, h_j)
$$

Here, $s_{i-1}$ is the previous decoder hidden state, and $h_j$ is the annotation at source position $j$. The model then normalizes the scores into attention weights:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

The context vector is the weighted sum of all source annotations:

$$
c_i = \sum_j \alpha_{ij} h_j
$$

The **soft-alignment** is the interpretation of $\alpha_{ij}$ as a source-target alignment weight. It tells us how much the annotation at source position $j$ contributed when generating target word $y_i$. The paper describes the context vector as an expected annotation over possible alignments.

The attention heatmap should not be treated as a gold word alignment or a complete explanation of model behavior. It is better to read it as a diagnostic signal: at a given decoding step, which source representations did the model rely on most strongly? For language pairs with very different word order or tokenization units, such as Korean-English, it is safer to interpret the heatmap as token-level soft correspondence rather than exact word-level alignment.

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

### 2. Why do BiRNN encoder states need transpose or reshape?

At the paper level, the important point is that the BiRNN encoder creates an annotation for every source position. The forward RNN reads the source sentence from left to right, and the backward RNN reads it from right to left. The annotation at each position concatenates both directional states:

$$
h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]
$$

This lets each source-word annotation contain both left and right context.

In implementation, transpose or reshape operations are usually about **tensor layout**, not about changing the mathematical meaning of the hidden states.

The key distinction is between `encoder_outputs` and `h_n`. The annotation sequence used by attention usually comes from `encoder_outputs`. In the [PyTorch GRU docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html), a bidirectional encoder's `output` has shape `(seq_len, batch, 2 * hidden)` when `batch_first=False`, and this corresponds to the source position-wise annotations. By contrast, `h_n` has shape `(num_layers * num_directions, batch, hidden)` and stores the final hidden state, which is usually used to initialize the decoder or summarize final states.

For example, if a PyTorch BiGRU uses `batch_first=False`, `encoder_outputs` comes out as `(src_len, batch, 2 * hidden)`. If the attention module expects `(batch, src_len, 2 * hidden)`, then `encoder_outputs.transpose(0, 1)` is needed.

For the decoder initial state, `h_n` may need to be reshaped from `(num_layers * 2, batch, hidden)` into `(num_layers, 2, batch, hidden)` so that the final layer's forward and backward states can be concatenated or projected. In short, one transpose/reshape aligns the annotation sequence layout, while the other separates layer and direction axes in the final hidden state.

In PyTorch, tensors can become non-contiguous after `transpose`, so it is safer to use `.contiguous().view(...)` before `.view()`, or to use `.reshape(...)` when appropriate.

The important point is that transpose and reshape are not learned operations. The learned parts are the embeddings, RNN parameters, attention scoring network, and decoder parameters. Transpose and reshape only arrange tensors so that those learned values keep the intended axis semantics.

The official GroundHog implementation also builds source batches so that each column is one sentence, giving tensors organized around `(max_seq_len, batch_size)`. Its encoder hidden layer is processed around `(max_seq_len, batch_size, dim)`. Therefore, a modern PyTorch reproduction that uses `batch_first=True` or implements attention with `torch.bmm` may contain more transpose operations than the paper equations suggest.

### 3. How are annotations learned?

The annotation $h_j$ is not a human-provided label. It is a hidden state produced by the encoder while reading the source sentence, and it is learned end-to-end through the translation loss.

The training flow is:

1. The encoder creates an annotation $h_j$ for each source position.
2. The decoder uses the previous state $s_{i-1}$ and annotation $h_j$ to compute score $e_{ij}$.
3. Softmax turns the scores into attention weights $\alpha_{ij}$.
4. The model builds the context vector $c_i = \sum_j \alpha_{ij} h_j$.
5. The decoder predicts the next target-word distribution using $c_i$, the previous target word, and the previous hidden state.
6. The negative log-likelihood loss for the gold target word backpropagates through the encoder, attention module, and decoder.

The key point is that alignment is not a hard latent variable here. It is a differentiable soft alignment, so gradients can flow not only into the attention scoring network but also into the BiRNN parameters that produced the encoder annotations.

The official GroundHog [`RecurrentLayerWithSearch`](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/encdec.py) follows this structure. The code projects the source annotation `c` and previous decoder state to build energy values, normalizes them with `probs = energy / normalizer`, and computes the weighted context with `ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)`. In PyTorch terms, this is roughly `ctx = weighted_sum(probs, annotations)`. Because this context enters the decoder update, the annotations are learned in the direction that improves translation likelihood.

### 4. Is the RNNencdec-30/50 vs RNNsearch-30/50 experiment design fair?

The short answer: the design is reasonable for comparing attention vs no attention within the same sentence-length cutoff, but it is not enough to isolate the effect of the `-30` vs `-50` cutoff itself.

In the paper, `-30` and `-50` are not just model-name suffixes. They indicate the maximum sentence length included during training. The paper compares two model families:

- `RNNencdec`: the baseline encoder-decoder model without attention
- `RNNsearch`: the proposed model with attention

Each model is trained under a maximum 30-word setting and a maximum 50-word setting. This gives `RNNencdec-30`, `RNNsearch-30`, `RNNencdec-50`, and `RNNsearch-50`.

In the official GroundHog [`state.py`](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/state.py), `prototype_encdec_state()` is documented in its docstring as the `RNNenc-30` setting, and the [README](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/README.md) describes it as the training setup for the paper's `RNNencdec-30`. This setting uses `seqlen = 30`, `bs = 80`, and `dim = 1000`. `prototype_search_state()` is documented as the `RNNsearch-50` setting and sets `search = True`, `forward = True`, `backward = True`, `seqlen = 50`, and `sort_k_batches = 20`. The README also explains that the default prototype corresponds to `RNNsearch-50`, while `RNNencdec-50` is trained by overriding `prototype_encdec_state` with `seqlen=50, sort_k_batches=20`.

Within the same length cutoff, comparing `RNNencdec-30` with `RNNsearch-30`, or `RNNencdec-50` with `RNNsearch-50`, is a useful way to measure the effect of adding attention. The results show that RNNsearch gets higher BLEU than RNNencdec under the same cutoff.

However, directly comparing `-30` and `-50` is more delicate. The `seqlen=50` setting includes longer training sentences than `seqlen=30`, so the length distribution and training difficulty are different. GroundHog's `sort_k_batches` is a batching strategy that groups several minibatches and sorts them by length to reduce padding; it is not a balanced length-bucket experiment design.

The summary is:

- **Reasonable comparison:** `RNNencdec` vs `RNNsearch` under the same cutoff
- **Comparison requiring caution:** direct `-30` vs `-50` comparison
- **Main interpretation:** RNNsearch reduces the fixed-length vector bottleneck and is more robust for long sentences, but the 30/50 setup is not a fully length-controlled experiment.

## Limitations and Questions

* Because the shortlist vocabulary maps rare words to `UNK`, rare-word translation remains weak.
* Attention computes source-target pair scores, so computation grows with both input length and output length.
* Attention heatmaps are useful diagnostics, but they should not be automatically treated as complete explanations of model behavior.

## Follow-Up Reading

- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- Sutskever et al., "Sequence to Sequence Learning with Neural Networks"
- Vaswani et al., "Attention Is All You Need"

## Reference

- [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- [HTML version via ar5iv](https://ar5iv.labs.arxiv.org/html/1409.0473)
- [Official GroundHog NMT implementation README](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/README.md)
- [GroundHog NMT state.py](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/state.py)
- [GroundHog NMT encdec.py](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/encdec.py)
