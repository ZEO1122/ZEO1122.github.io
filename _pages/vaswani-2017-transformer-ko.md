---
permalink: /ko/notes/vaswani-2017-transformer/
title: "Attention Is All You Need"
description: "Transformer 논문의 문제 정의, 구조, 실험 결과와 공부하며 정리한 질문을 다룬 한국어 노트입니다."
date: 2026-08-11
author_profile: true
lang: ko
locale: ko-KR
og_locale: ko_KR
author: jeo_ko
translations:
  en: /notes/vaswani-2017-transformer/
  ko: /ko/notes/vaswani-2017-transformer/
tags:
  - 자연어 처리
  - Transformer
  - Attention
  - Sequence-to-Sequence
paperurl: "https://arxiv.org/abs/1706.03762"
citation: "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. Attention Is All You Need. NeurIPS 2017."
---

{% include toc %}
{% include paper-note-toc-layout.html %}
{% include paper-note-head.html %}

## 한 줄 요약

Transformer는 recurrence와 convolution을 여러 층의 self-attention으로 대체해 모든 token이 다른 token을 직접 참고할 수 있게 하고, 시퀀스 모델의 병렬 학습 효율을 크게 높였다.

## 논문 정보

| 항목 | 내용 |
|------|------|
| 논문 | Attention Is All You Need |
| 저자 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin |
| 학회 | NeurIPS 2017 |
| arXiv 최초 제출 | 2017-06-12 |
| 과제 | 신경망 기계번역, 추가 constituency parsing 평가 |
| 핵심 키워드 | Transformer, self-attention, scaled dot-product attention, multi-head attention, positional encoding |
| 링크 | [arXiv](https://arxiv.org/abs/1706.03762) · [HTML 논문](https://arxiv.org/html/1706.03762v7) · [NeurIPS proceedings](https://papers.nips.cc/paper/7181-attention-is-all-you-need) |

## 읽은 이유

Bahdanau attention은 여전히 RNN decoder state를 query로 사용해 각 목표 시점에서 encoder state를 참고한다. Transformer는 attention이라는 아이디어를 유지하면서 recurrence를 완전히 제거한다. Sequence-to-sequence attention 다음에 이 논문을 읽으면 attention이 RNN을 보조하는 module에서 모델의 주된 연산으로 확장되는 과정을 이해할 수 있다.

## 문제 정의

RNN과 convolution 기반 시퀀스 모델에는 다음 한계가 있다.

- RNN은 token을 순서대로 처리하므로 sequence position을 가로질러 학습을 병렬화하기 어렵다.
- RNN에서 멀리 떨어진 token의 정보는 여러 recurrent step을 거쳐야 하고, local convolution에서도 여러 layer가 필요하다.
- Encoder-decoder attention을 사용하더라도 기존 시스템의 encoder와 decoder는 여전히 recurrence 또는 convolution에 의존한다.

논문은 번역 성능을 유지하면서 병렬 학습 효율을 높일 수 있도록 attention만으로 sequence transduction model을 구성할 수 있는지 묻는다.

## 핵심 아이디어

Transformer는 각 token을 vector로 표현한 뒤 self-attention과 position-wise feed-forward network를 반복 적용해 표현을 갱신한다. Encoder self-attention에서는 모든 위치가 다른 모든 원문 위치를 참고할 수 있다. Decoder self-attention에서는 mask를 사용해 미래 목표 token을 보지 못하게 한다.

표현 시퀀스 $X$가 주어지면 각 layer는 이전 layer의 표현에서 query, key, value를 만든다. Attention score는 query와 key를 비교하고, 출력은 value의 가중합이다.

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

RNN 기반 모델과 달리 full self-attention layer에서는 어떤 두 token 사이의 경로 길이도 상수다.

<div class="paper-note-table-wrap" markdown="1">

| Layer 유형 | Layer당 복잡도 | 순차 연산 수 | 최대 경로 길이 |
|------------|----------------|--------------|----------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

</div>

위 표는 논문의 **Table 1** 수치를 직접 옮긴 것이다. 모든 위치가 한 layer 안에서 상호작용할 수 있으므로, 시퀀스 길이에 따라 경로가 길어지는 recurrence보다 장거리 의존성을 다루는 데 유리하다는 논거를 보여 준다.

## 방법

### Encoder-Decoder 구조

Transformer는 encoder-decoder 틀을 유지하되 layer의 종류를 바꾼다. Encoder는 동일한 구조의 layer $N=6$개를 쌓으며, 각 layer는 다음 요소로 구성된다.

- multi-head self-attention
- position-wise feed-forward network
- 각 sublayer를 감싸는 residual connection과 layer normalization

Decoder 역시 $N=6$개 layer를 쌓지만, 각 layer에는 세 sublayer가 있다.

- 이전 목표 위치만 보는 masked multi-head self-attention
- encoder output을 참고하는 encoder-decoder attention
- position-wise feed-forward network

<figure class="paper-note-figure paper-note-figure--narrow">
  <img src="/assets/images/paper-notes/vaswani-2017/figure-1-transformer-architecture.png" alt="Transformer 모델 구조를 보여 주는 Vaswani 등의 Figure 1">
  <figcaption>
    <strong>Figure 1.</strong> 논문에 수록된 Transformer 구조. 왼쪽은 encoder stack, 오른쪽은 autoregressive decoder stack이다. 출처: <a href="https://arxiv.org/html/1706.03762v7#S3.F1">arXiv:1706.03762 HTML 렌더링</a>.
  </figcaption>
</figure>

### Scaled Dot-Product Attention

Attention 식은 각 query와 모든 key를 dot product로 비교한다. $d_k$가 커질 때 logit이 지나치게 커지는 것을 막기 위해 $\sqrt{d_k}$로 나눈다.

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Scaling이 없으면 key dimension이 증가할수록 dot product의 크기가 커져 softmax가 gradient가 작은 영역으로 들어갈 수 있다. 따라서 scaling term은 dot-product attention의 학습 안정성을 높이는 장치다.

### Multi-Head Attention

하나의 attention을 전체 $d_{\text{model}}$ 공간에서 수행하는 대신, $Q$, $K$, $V$를 서로 다르게 projection한 $h$개의 head를 학습한다. 각 head는 더 작은 representation subspace에서 attention을 수행한다.

$$
\mathrm{head_i}=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

Head의 출력을 concatenate한 뒤 model dimension으로 projection한다.

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head_1},...,\mathrm{head_h})W^O
$$

**출력 projection $W^O$는 서로 다른 representation subspace에서 각 head가 추출한 정보를 학습 가능한 방식으로 다시 조합해 하나의 $d_{\text{model}}$ 차원 표현으로 통합한다.** 단순 concatenate는 head 출력을 나란히 배치할 뿐이지만, $W^O$는 다음 layer가 head별 신호의 학습된 혼합을 받도록 한다.

<figure class="paper-note-figure paper-note-figure--medium">
  <div class="paper-note-figure-grid">
    <div>
      <img src="/assets/images/paper-notes/vaswani-2017/figure-2-scaled-dot-product-attention.png" alt="Scaled dot-product attention을 보여 주는 Figure 2 왼쪽">
      <div class="paper-note-panel-label">Scaled Dot-Product Attention</div>
    </div>
    <div>
      <img src="/assets/images/paper-notes/vaswani-2017/figure-2-multi-head-attention.png" alt="Multi-head attention을 보여 주는 Figure 2 오른쪽">
      <div class="paper-note-panel-label">Multi-Head Attention</div>
    </div>
  </div>
  <figcaption>
    <strong>Figure 2.</strong> 논문에 수록된 scaled dot-product attention과 multi-head attention 구조. 출처: <a href="https://arxiv.org/html/1706.03762v7#S3.F2">arXiv:1706.03762 HTML 렌더링</a>.
  </figcaption>
</figure>

### Positional Encoding

Transformer에는 recurrence와 convolution이 없으므로 token의 순서를 별도로 주입해야 한다. 논문은 encoder와 decoder의 입력 embedding에 positional encoding을 더한다.

$$
PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})
$$

$$
PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{\text{model}}})
$$

Token embedding과 positional encoding은 같은 $d_{\text{model}}$ 차원이므로 element-wise addition이 가능하다.

$$
x_{pos}=E(x_{pos})+PE_{pos}
$$

**Transformer는 input embedding과 positional encoding을 concatenate하거나 dot product하지 않는다. 같은 차원에서 원소별로 더해 model dimension을 늘리지 않으면서 token의 의미와 위치 정보를 하나의 representation에 함께 담고, 이후 self-attention이 이를 활용하게 한다.**

Concatenation은 표현 크기를 $d_{\text{model}}$에서 $2d_{\text{model}}$로 늘리거나 첫 layer 전에 별도 projection을 요구한다. Dot product는 두 vector를 scalar similarity로 축소하므로 attention에 전달할 token representation으로 쓸 수 없다. Addition은 residual stream의 차원을 고정하고, 이후의 학습 가능한 projection이 의미와 위치 신호를 어떻게 사용할지 결정하게 한다.

### Position-Wise Feed-Forward Network

각 layer는 모든 위치에 독립적으로 같은 feed-forward network를 적용한다.

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

한 layer 안에서는 모든 위치가 같은 parameter를 공유하지만 layer끼리는 서로 다른 parameter를 사용한다. Base model은 $d_{\text{model}}=512$, $d_{\text{ff}}=2048$을 사용한다.

## 실험

주요 평가는 WMT 2014 영어-독일어와 영어-프랑스어 번역이다. Base Transformer는 $N=6$, $d_{\text{model}}=512$, $d_{\text{ff}}=2048$, $h=8$, $d_k=d_v=64$를 사용한다. Big model은 모델 크기를 키우고 더 오래 학습한다.

아래 수치는 논문의 **Table 2**를 직접 옮긴 것이다.

<div class="paper-note-table-wrap" markdown="1">

| 모델 | EN-DE BLEU | EN-FR BLEU | 보고된 학습 비용 |
|------|-----------:|-----------:|------------------|
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

Transformer (big)은 영어-독일어에서 28.4 BLEU, 영어-프랑스어에서 41.8 BLEU를 기록했다. 강한 recurrent, convolutional, ensemble baseline보다 훨씬 적은 학습 연산량으로 경쟁력 있는 성능을 달성한 것이 핵심 결과다.

## 배운 점

Transformer는 단순히 "RNN 대신 attention을 사용한 모델"이 아니다. 시퀀스 표현을 만드는 방식 자체를 바꾼다. RNN에서는 먼 token의 정보가 여러 recurrent step을 지나야 하지만, Transformer layer에서는 모든 token이 다른 모든 token과 직접 비교하고 각 위치에서 가져올 정보의 양을 학습한다.

또한 architecture 전체의 residual stream을 $d_{\text{model}}$이라는 하나의 차원으로 통일한다. Embedding, positional encoding, sublayer output, residual connection, multi-head attention output이 모두 같은 차원에 있으므로 layer를 안정적으로 반복해 쌓을 수 있다.

## 공부하며 정리한 질문

### 1. Self-attention은 첫 layer부터 모든 token 관계를 학습하는가?

연결 가능성만 보면 그렇다. 첫 self-attention layer부터 모든 token이 같은 시퀀스의 다른 모든 token과 attention score를 계산할 수 있다. Table 1에서 self-attention의 최대 경로 길이를 $O(1)$로 둔 이유다.

그러나 첫 layer가 곧바로 모든 고수준 관계를 이해한다는 뜻은 아니다. 첫 layer의 $Q$, $K$, $V$는 token embedding과 positional encoding을 더한 비교적 초기 표현에서 projection된다. 더 깊은 layer는 앞선 layer가 이미 문맥화한 표현을 대상으로 self-attention을 반복한다.

1. 초기 layer는 어휘와 위치 정보에 가까운 표현에서 직접적인 token 간 상호작용을 만든다.
2. 중간 layer는 이전 layer가 만든 문맥화된 표현 사이의 관계를 다시 계산한다.
3. 후반 layer는 이를 조합해 더 추상적인 구문, 의미, 과제 특화 관계를 형성할 수 있다.

정리하면 **Transformer는 첫 layer부터 모든 token 사이의 관계를 계산할 수 있으며, layer가 깊어질수록 이전 layer에서 형성된 문맥화된 표현 사이의 관계를 반복적으로 재계산하고 조합해 더 복합적이고 고차원적인 관계를 학습한다.**

### 2. Input embedding과 positional encoding을 concatenate하지 않고 더하는 이유는 무엇인가?

각 token representation에는 token의 의미와 위치가 모두 필요하다. Pure self-attention에는 순서 개념이 자연스럽게 들어 있지 않으므로 positional encoding이 이를 보완한다.

논문은 positional encoding의 차원을 token embedding과 같은 $d_{\text{model}}$로 맞추고 원소별 덧셈을 사용한다.

$$
\text{input representation}=\text{token embedding}+\text{positional encoding}
$$

Concatenation은 hidden size를 늘리거나 첫 encoder·decoder layer 전에 별도 projection을 요구한다. Dot product는 두 vector를 하나의 scalar로 줄여 attention이 필요한 vector representation을 잃게 한다.

Addition은 model dimension을 유지하므로 같은 표현을 residual connection과 이후 projection에 그대로 전달할 수 있다. 학습되는 $W^Q$, $W^K$, $W^V$가 결합된 의미·위치 신호를 어떻게 사용할지 결정한다.

### 3. 여러 head를 concatenate한 뒤 왜 $W^O$가 필요한가?

각 head는 서로 다른 projection을 사용하므로 다른 관계 유형, 위치, representation subspace에 집중할 수 있다. Attention 이후 head 출력은 다음처럼 concatenate된다.

$$
\mathrm{Concat}(\mathrm{head_1},...,\mathrm{head_h})
$$

여기서 멈추면 독립적인 head 출력이 block 단위로 나란히 놓인 상태다. $W^O$는 이 block을 학습 가능한 방식으로 섞는다.

따라서 $W^O$는 단순히 shape을 맞추는 layer가 아니다. **각 head가 서로 다른 representation subspace에서 추출한 정보를 다시 조합해 residual stream과 다음 layer가 사용할 하나의 $d_{\text{model}}$ 표현으로 통합하는 학습 단계**다.

### 4. 하나의 full-dimensional head 대신 여러 head를 사용하는 이유는 무엇인가?

논문은 multi-head attention이 서로 다른 representation subspace와 서로 다른 위치의 정보를 동시에 참고하게 한다고 설명한다. 하나의 head도 여러 위치에 attention을 분배할 수 있지만, 모든 정보가 하나의 attention pattern을 통해 평균된다.

여러 head를 사용하면 어떤 head는 지역적인 구문 관계, 다른 head는 장거리 일치 관계, 또 다른 head는 번역 alignment에 집중할 수 있다. 역할을 사람이 지정하는 것은 아니지만, architecture가 여러 개의 병렬 attention 관점을 학습할 여지를 제공한다.

### 5. Bahdanau attention과 어떤 관계인가?

Bahdanau attention은 decoder 시점마다 encoder annotation의 가중합으로 context vector를 만든다.

$$
c_i=\sum_j\alpha_{ij}h_j
$$

Transformer attention도 weighted sum이라는 핵심을 유지하지만 범위를 일반화한다. Self-attention에서는 query, key, value가 모두 같은 시퀀스에서 나오고, encoder-decoder attention에서는 query는 decoder에서, key와 value는 encoder에서 나온다.

- Bahdanau attention: RNN decoder가 한 목표 시점에 필요한 원문 context를 검색한다.
- Transformer self-attention: 모든 위치가 다른 위치의 context를 병렬로 검색한다.
- Transformer encoder-decoder attention: decoder 위치가 encoder output에서 원문 context를 검색한다.

## 한계와 남은 질문

- Full self-attention은 모든 위치 쌍의 score를 계산하므로 긴 시퀀스에서 $O(n^2)$ 비용이 크다.
- Absolute sinusoidal positional encoding은 순서를 주입하지만, 후속 연구에서는 relative 또는 learned positional scheme을 탐구한다.
- Attention map은 해석에 도움이 되지만 모델 행동의 완전한 설명으로 볼 수 없다.
- 원 논문은 기계번역에 초점을 맞췄으며, 이후 language model은 학습 목적과 배포 방식이 달라졌다.

## 후속 읽기

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Radford et al., "Improving Language Understanding by Generative Pre-Training"
- Shaw et al., "Self-Attention with Relative Position Representations"
- Harvard NLP, "The Annotated Transformer"

## 참고문헌

- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [arXiv HTML 버전](https://arxiv.org/html/1706.03762v7)
- [NeurIPS proceedings 페이지](https://papers.nips.cc/paper/7181-attention-is-all-you-need)
- [Google Research publication 페이지](https://research.google/pubs/attention-is-all-you-need/)
