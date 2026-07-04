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
      bottom: 0;
      right: -14.5rem;
      float: none !important;
      width: 13rem !important;
      height: auto;
      margin: 0 !important;
    }

    .page__content > .sidebar__right .toc {
      position: sticky;
      top: 1.5rem;
    }
  }

  @media (max-width: 1279px) {
    .page__content > .sidebar__right {
      float: none !important;
      width: auto !important;
      margin: 0 0 1.5rem !important;
    }

    .page__content > .sidebar__right .toc {
      position: static;
    }
  }

  @media (max-width: 700px) {
    .paper-note-figure-grid {
      grid-template-columns: 1fr;
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

이 섹션은 단순한 PyTorch 구현 체크리스트가 아니라, 논문을 직접 구현하거나 재현할 때 헷갈릴 수 있는 개념을 정리한 메모이다.

### 1. Soft-search와 soft-alignment는 어떻게 해석해야 하는가?

Bahdanau et al.이 말하는 **soft-search**는 decoder가 다음 target word를 생성할 때 source sentence 전체를 하나의 고정 벡터로만 사용하는 대신, source의 각 위치별 annotation을 다시 훑으면서 현재 step에 필요한 정보를 확률적으로 찾는 과정이다. 여기서 “search”는 하나의 source token을 hard하게 선택한다는 뜻이 아니라, 모든 source position에 대해 relevance score를 계산한 뒤 softmax로 weight를 부여한다는 뜻이다.

각 decoding step $i$에서 alignment score는 대략 다음처럼 계산된다.

$$
e_{ij} = a(s_{i-1}, h_j)
$$

여기서 $s_{i-1}$는 이전 decoder hidden state이고, $h_j$는 source position $j$의 annotation이다. 이후 softmax를 통해 attention weight를 만든다.

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

그리고 context vector는 모든 source annotation의 weighted sum으로 계산된다.

$$
c_i = \sum_j \alpha_{ij} h_j
$$

이때 **soft-alignment**는 $\alpha_{ij}$를 source-target alignment처럼 해석한 것이다. 즉 target word $y_i$를 만들 때 source position $j$의 annotation이 얼마나 많이 사용되었는지 보여주는 가중치이다. 논문은 이를 possible alignments에 대한 expected annotation으로 해석할 수 있다고 설명한다.

다만 attention heatmap을 “정답 단어 정렬”이나 “완전한 설명”으로 단정하면 안 된다. 이 값은 모델이 그 step에서 어떤 source representation을 상대적으로 크게 사용했는지 보여주는 진단 신호에 가깝다. 특히 Korean-English처럼 어순, 조사, 어미, 형태소 단위가 크게 다른 언어쌍에서는 word-level alignment보다 tokenization 단위에 따른 soft correspondence로 보는 편이 안전하다.

### 2. BiRNN encoder에서 hidden state를 왜 transpose 또는 reshape하는가?

논문 수준에서 중요한 것은 BiRNN encoder가 각 source position마다 annotation을 만든다는 점이다. forward RNN은 source sentence를 왼쪽에서 오른쪽으로 읽고, backward RNN은 오른쪽에서 왼쪽으로 읽는다. 각 위치의 annotation은 두 방향의 hidden state를 concatenate해서 만든다.

$$
h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]
$$

이렇게 하면 각 source word annotation이 앞쪽 문맥과 뒤쪽 문맥을 모두 담을 수 있다.

구현에서 hidden state를 transpose하거나 reshape하는 이유는 대부분 **수식의 의미 때문이 아니라 tensor layout을 맞추기 위해서**이다.

주의할 점은 `encoder_outputs`와 `h_n`을 구분하는 것이다. Attention에서 사용하는 annotation sequence는 보통 `encoder_outputs`에서 온다. [PyTorch GRU docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) 기준으로 bidirectional encoder의 `output`은 `batch_first=False`일 때 `(seq_len, batch, 2 * hidden)` 형태이고, 이것이 source position별 annotation에 해당한다. 반면 `h_n`은 `(num_layers * num_directions, batch, hidden)` 형태의 final hidden state로, 주로 decoder initial state를 만들 때 사용한다.

예를 들어 PyTorch에서 `batch_first=False`로 BiGRU를 쓰면 `encoder_outputs`는 `(src_len, batch, 2 * hidden)` 형태로 나온다. Attention module이 `(batch, src_len, 2 * hidden)`을 기대한다면 `encoder_outputs.transpose(0, 1)`이 필요하다.

반면 decoder initial state를 만들 때는 `h_n`의 `(num_layers * 2, batch, hidden)` 축을 `(num_layers, 2, batch, hidden)`처럼 분리한 뒤, 마지막 layer의 forward/backward state를 concatenate할 수 있다. 따라서 하나의 transpose/reshape는 annotation sequence layout을 맞추기 위한 것이고, 다른 하나는 final hidden state에서 layer/direction 축을 분리하기 위한 것이다.

PyTorch에서는 `transpose` 이후 tensor가 non-contiguous일 수 있으므로, `.view()`를 써야 한다면 `.contiguous().view(...)`를 사용하거나 `.reshape(...)`를 쓰는 편이 안전하다.

중요한 점은 transpose/reshape 자체가 학습되는 연산은 아니라는 것이다. 학습되는 것은 embedding, RNN parameter, attention scoring network, decoder parameter이고, transpose/reshape는 이 값들이 올바른 축 의미를 유지하도록 배치하는 구현상의 조작이다.

공식 GroundHog 구현에서도 source batch는 각 column이 하나의 sentence가 되도록 `(max_seq_len, batch_size)` 형식으로 구성되고, encoder hidden layer는 `(max_seq_len, batch_size, dim)` 형태를 기준으로 처리된다. 따라서 현대 PyTorch 구현에서 `batch_first=True`를 쓰거나 attention 계산을 `torch.bmm`으로 구현한다면, 논문 수식과 달리 transpose가 더 자주 보일 수 있다.

### 3. Annotation은 어떻게 학습되는가?

Annotation $h_j$는 사람이 부여한 정답 label이 아니다. Encoder가 source sentence를 읽으면서 만든 hidden state이며, translation loss를 통해 end-to-end로 학습된다.

학습 흐름은 다음과 같다.

1. Encoder가 각 source position의 annotation $h_j$를 만든다.
2. Decoder가 이전 state $s_{i-1}$와 annotation $h_j$를 이용해 score $e_{ij}$를 계산한다.
3. Softmax를 통해 attention weight $\alpha_{ij}$를 만든다.
4. Context vector $c_i = \sum_j \alpha_{ij} h_j$를 만든다.
5. Decoder가 $c_i$, 이전 target word, 이전 hidden state를 이용해 다음 target word probability를 예측한다.
6. 정답 target word에 대한 negative log-likelihood loss가 encoder, attention, decoder 전체로 역전파된다.

논문에서 중요한 점은 alignment가 hard latent variable이 아니라 differentiable soft alignment라는 것이다. 따라서 gradient가 attention scoring network뿐 아니라 encoder annotation을 만든 BiRNN parameter까지 전달될 수 있다.

공식 GroundHog 구현의 [`RecurrentLayerWithSearch`](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/encdec.py)도 이 구조를 따른다. 코드에서는 source annotation `c`와 이전 decoder state를 projection해 energy를 만들고, `probs = energy / normalizer`로 normalize한 뒤, `ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)`로 weighted context를 계산한다. PyTorch식으로 쓰면 이는 대략 `ctx = weighted_sum(probs, annotations)`에 해당한다. 이 context가 decoder update에 들어가기 때문에 annotation은 번역 likelihood를 높이는 방향으로 함께 학습된다.

### 4. RNNencdec-30/50, RNNsearch-30/50 실험 설계는 적합한가?

결론부터 말하면, 이 설계는 같은 cutoff 안에서 attention 유무를 비교하기에는 적합하지만, `-30`과 `-50`을 서로 직접 비교해 length cutoff 자체의 효과를 결론내리기에는 부족하다.

논문에서 `-30`과 `-50`은 단순한 모델 이름이 아니라 training에 포함되는 sentence length cutoff를 의미한다. 논문은 두 모델군을 비교한다.

- `RNNencdec`: attention이 없는 기존 encoder-decoder 모델
- `RNNsearch`: attention을 사용하는 제안 모델

각 모델은 sentence length를 최대 30 words로 제한한 setting과 최대 50 words로 제한한 setting에서 학습된다. 그래서 `RNNencdec-30`, `RNNsearch-30`, `RNNencdec-50`, `RNNsearch-50`이 나온다.

공식 GroundHog 구현의 [`state.py`](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/state.py)를 보면 `prototype_encdec_state()`는 docstring상 `RNNenc-30` 설정으로 적혀 있으며, [README](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/README.md)에서는 이를 논문의 `RNNencdec-30` 학습 설정으로 안내한다. 이 설정은 `seqlen = 30`, `bs = 80`, `dim = 1000`을 사용한다. `prototype_search_state()`는 `RNNsearch-50` 설정으로 문서화되어 있으며, `search = True`, `forward = True`, `backward = True`, `seqlen = 50`, `sort_k_batches = 20`을 설정한다. README도 기본 prototype이 `RNNsearch-50`에 해당하고, `RNNencdec-50`은 `prototype_encdec_state`에서 `seqlen=50, sort_k_batches=20`을 override해서 학습한다고 설명한다.

같은 length cutoff 안에서 `RNNencdec-30`과 `RNNsearch-30`, `RNNencdec-50`과 `RNNsearch-50`을 비교하면 fixed-vector encoder-decoder와 attention-based decoder의 차이를 볼 수 있다. 실제 결과에서도 같은 cutoff 기준으로 RNNsearch가 RNNencdec보다 높은 BLEU를 보인다.

하지만 `30`과 `50`을 직접 비교해 “길이를 늘리면 성능이 좋아진다/나빠진다”라고 단순 결론을 내리기에는 한계가 있다. `seqlen=50` setting은 `seqlen=30` setting보다 더 긴 문장들을 training에 포함하므로, 학습 데이터의 길이 분포와 난이도가 달라진다. 또한 GroundHog의 `sort_k_batches`는 여러 minibatch를 모아 길이순으로 정렬한 뒤 padding을 줄이기 위한 batching 전략이지, length bucket별 데이터 분포를 균등하게 맞추는 실험 설계는 아니다.

정리하면 다음과 같다.

- **적합한 비교:** 같은 cutoff 안에서 `RNNencdec` vs `RNNsearch`
- **주의할 비교:** `-30` vs `-50` 자체의 직접 비교
- **핵심 해석:** 논문의 주장은 “RNNsearch가 fixed-length vector bottleneck을 완화해 긴 문장에 더 robust하다”는 것이며, 30/50 setting은 이를 보여주는 근거이지만 완전히 균등한 length-controlled 실험은 아니다.

## Limitations and Questions

* Shortlist vocabulary 때문에 rare word가 `UNK`로 매핑되며, rare-word translation은 여전히 약하다.
* Attention은 source-target pair score를 계산하므로 입력 길이와 출력 길이가 모두 길어질수록 계산량이 증가한다.
* Attention heatmap은 유용한 진단 도구이지만, 모델 행동에 대한 완전한 설명으로 자동 해석해서는 안 된다.

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
