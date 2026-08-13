---
permalink: /ko/notes/bahdanau-2014-nmt-attention/
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
description: "Bahdanau attention 논문의 문제 정의, 모델 구조, 실험 결과와 공부하며 정리한 질문을 다룬 한국어 노트입니다."
date: 2026-07-03
paper_note: true
hide_date: true
note_area: "자연어 처리 · 신경망 기계번역"
note_summary: "학습 가능한 soft alignment가 recurrent encoder-decoder의 고정 길이 context 병목을 어떻게 해소하는지 정리한 노트입니다."
paper_authors: "Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio"
paper_venue: "ICLR 2015 Oral"
paper_date: 2014-09-01
author_profile: true
lang: ko
locale: ko-KR
og_locale: ko_KR
author: jeo_ko
translations:
  en: /notes/bahdanau-2014-nmt-attention/
  ko: /ko/notes/bahdanau-2014-nmt-attention/
tags:
  - 자연어 처리
  - 신경망 기계번역
  - Attention
  - Sequence-to-Sequence
paperurl: "https://arxiv.org/abs/1409.0473"
htmlpaperurl: "https://ar5iv.labs.arxiv.org/html/1409.0473"
citation: "Bahdanau, D., Cho, K., & Bengio, Y. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015 oral."
---

{% include toc %}
{% include paper-note-toc-layout.html %}
{% include paper-note-head.html %}
{% include paper-note-meta.html %}

## 개요

초기 encoder-decoder의 고정 길이 context 병목을 없애기 위해, 목표 단어를 생성할 때마다 원문에서 **어디를 참고할지 학습하는** 신경망 기계번역 모델을 제안한 논문이다.

기존 encoder-decoder NMT는 원문 전체를 하나의 고정 길이 벡터 $c$로 인코딩한다. Decoder는 모든 목표 단어를 같은 압축 표현에 의존해 생성한다.

논문은 이 구조에 다음 문제가 있다고 본다.

- 긴 문장의 정보를 하나의 벡터에 안정적으로 담기 어렵다.
- 생성 시점마다 필요한 원문 단어가 다른데도 같은 context를 사용한다.
- 번역에는 단일한 hard word mapping보다 유연하고 비단조적인 alignment가 필요하다.

Encoder가 문장 하나를 대표하는 벡터를 만드는 대신, 원문의 각 위치에 대응하는 annotation의 시퀀스를 출력한다.

$$
h_1,h_2,\ldots,h_{T_x}
$$

목표 시점 $i$마다 decoder는 모든 원문 annotation에 attention weight를 계산하고, 해당 시점에 필요한 context vector를 만든다.

$$
c_i=\sum_{j=1}^{T_x}\alpha_{ij}h_j
$$

즉, 문장 전체에 대해 한 번 참고 위치를 정하는 것이 아니라 **현재 목표 단어를 생성할 때 중요한 원문 위치**를 매 시점 다시 결정한다.

## 방법

### Encoder

Encoder는 bidirectional RNN이다. 원문의 위치 $j$마다 정방향 hidden state와 역방향 hidden state를 이어 붙인다.

$$
h_j=[\overrightarrow{h_j};\overleftarrow{h_j}]
$$

따라서 각 annotation은 해당 단어의 왼쪽 문맥과 오른쪽 문맥을 모두 포함한다.

### Alignment Model

목표 시점 $i$와 원문 위치 $j$에 대해 alignment score를 계산한다.

$$
e_{ij}=a(s_{i-1},h_j)
$$

$s_{i-1}$은 이전 decoder hidden state이고, $a(\cdot)$는 학습 가능한 feed-forward network다. 이 점수에 softmax를 적용해 attention weight를 얻는다.

$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}
$$

$\alpha_{ij}$는 미분 가능한 soft-alignment weight이므로 번역 모델과 alignment mechanism 전체를 backpropagation으로 함께 학습할 수 있다.

### Decoder

Decoder는 다음 정보를 사용해 다음 목표 단어를 예측한다.

- 이전 목표 단어
- 이전 decoder state
- 현재 시점의 context vector $c_i$

기본 encoder-decoder와 달리 decoding 시점마다 서로 다른 context vector를 받는다는 점이 핵심이다.

<figure class="paper-note-figure paper-note-figure--narrow">
  <img src="/assets/images/paper-notes/bahdanau-2014/figure-1-model.png" alt="Bahdanau 등의 attention 기반 신경망 기계번역 모델을 보여 주는 Figure 1">
  <figcaption>
    <strong>Figure 1.</strong> Bahdanau 등의 논문에 수록된 attention 기반 decoder 구조.
  </figcaption>
</figure>

## 실험

### 번역 성능

WMT 2014 영어-프랑스어 번역에서 기본 RNN encoder-decoder와 제안 모델인 **RNNsearch**를 비교한다. 아래 수치는 논문의 **Table 1**을 직접 옮긴 것이다.

| 모델 | BLEU, 전체 문장 | BLEU, UNK 제외 |
|------|----------------:|---------------:|
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* | 28.45 | 36.15 |
| Moses phrase-based SMT | 33.30 | 35.63 |

### 문장 길이와 학습된 정렬

중요한 결과는 전체 BLEU 향상만이 아니다. **Figure 2**에서 고정 벡터 encoder-decoder는 문장이 길어질수록 성능이 크게 떨어지지만, attention 모델은 긴 문장에서도 훨씬 안정적이다.

<figure class="paper-note-figure paper-note-figure--medium">
  <img src="/assets/images/paper-notes/bahdanau-2014/figure-2-bleu-by-length.png" alt="원문 문장 길이에 따른 BLEU 점수를 보여 주는 Bahdanau 등의 Figure 2">
  <figcaption>
    <strong>Figure 2.</strong> 논문의 문장 길이별 BLEU 그래프. RNNsearch가 고정 벡터 encoder-decoder보다 긴 문장을 잘 처리한다는 핵심 근거다.
  </figcaption>
</figure>

논문은 번역 성능뿐 아니라 학습된 soft-alignment도 직접 확인한다. Figure 3에서는 영어 원문과 생성된 프랑스어 단어 사이에 대체로 대각선 형태의 대응이 나타나며, 어순이 바뀌는 구문에서는 비단조적인 정렬도 형성된다.

<figure class="paper-note-figure">
  <div class="paper-note-figure-grid">
    <div><img src="/assets/images/paper-notes/bahdanau-2014/figure-3a-alignment.png" alt="RNNsearch-50 soft-alignment 예시 Figure 3a"><div class="paper-note-panel-label">(a)</div></div>
    <div><img src="/assets/images/paper-notes/bahdanau-2014/figure-3b-alignment.png" alt="RNNsearch-50 soft-alignment 예시 Figure 3b"><div class="paper-note-panel-label">(b)</div></div>
    <div><img src="/assets/images/paper-notes/bahdanau-2014/figure-3c-alignment.png" alt="RNNsearch-50 soft-alignment 예시 Figure 3c"><div class="paper-note-panel-label">(c)</div></div>
    <div><img src="/assets/images/paper-notes/bahdanau-2014/figure-3d-alignment.png" alt="RNNsearch-50 soft-alignment 예시 Figure 3d"><div class="paper-note-panel-label">(d)</div></div>
  </div>
  <figcaption>
    <strong>Figure 3.</strong> 논문의 RNNsearch-50 soft-alignment 예시. 축은 영어 원문 단어와 생성된 프랑스어 단어에 대응하며, 밝은 셀일수록 attention weight가 크다.
  </figcaption>
</figure>

### 결과 해석

가장 큰 기여는 **문장 단위 압축**을 **시점별 검색**으로 바꾼 것이다. Encoder는 유용한 표현의 시퀀스를 저장하고, decoder는 생성 시점마다 필요한 정보를 꺼내 쓴다.

Attention weight를 원문-번역문 alignment matrix로 확인할 수 있어 일반 encoder-decoder보다 모델의 판단을 진단하기도 쉽다.

Transformer를 공부하는 과정에서는 다음 연결이 중요했다.

- Bahdanau attention은 RNN decoder state를 query처럼 사용해 encoder state를 참고한다.
- Transformer는 recurrence를 제거하고 attention을 주된 연산으로 사용한다.
- 하나의 압축 표현에 의존하지 않고 필요한 문맥을 선택한다는 동기는 이어진다.

## 공부하며 정리한 질문

### 1. Soft-search와 soft-alignment를 어떻게 해석해야 하는가?

**Soft-search**는 decoder가 원문 전체를 대표하는 하나의 벡터에 의존하지 않고, 목표 단어를 생성할 때마다 모든 원문 위치를 점수화한다는 뜻이다. 여기서 search는 원문 단어 하나를 hard decision으로 선택한다는 뜻이 아니다. 모든 위치의 점수를 구한 뒤 softmax로 연속적인 가중치를 만드는 과정이다.

**Soft-alignment**는 $\alpha_{ij}$를 원문 위치 $j$와 목표 단어 $y_i$ 사이의 정렬 정도로 해석한 것이다. Context vector는 가능한 alignment에 대한 annotation의 기댓값으로 볼 수 있다.

Attention heatmap을 정답 word alignment나 모델 행동의 완전한 설명으로 간주해서는 안 된다. 특정 decoding 시점에 어떤 원문 표현을 강하게 참고했는지 보여 주는 진단 신호로 보는 편이 타당하다. 한국어-영어처럼 어순과 tokenization 단위가 크게 다른 언어 쌍에서는 정확한 단어 대응보다 token-level soft correspondence로 해석하는 것이 안전하다.

### 2. BiRNN encoder state에 transpose나 reshape가 필요한 이유는 무엇인가?

논문에서 중요한 것은 BiRNN encoder가 원문 위치마다 annotation을 만든다는 점이다. 구현에서 transpose와 reshape는 hidden state의 수학적 의미를 바꾸는 연산이 아니라 **tensor layout을 맞추는 연산**이다.

`encoder_outputs`와 `h_n`을 구분해야 한다. Attention에 쓰는 annotation 시퀀스는 보통 `encoder_outputs`에서 가져온다. [PyTorch GRU 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)에서 `batch_first=False`인 bidirectional encoder의 `output` shape은 `(seq_len, batch, 2 * hidden)`이다. 반면 `h_n`은 `(num_layers * num_directions, batch, hidden)`이며 decoder 초기화 등에 사용할 최종 hidden state를 담는다.

Attention module이 `(batch, src_len, 2 * hidden)`을 요구하면 `encoder_outputs.transpose(0, 1)`이 필요하다. Decoder 초기 state를 만들 때는 `h_n`을 `(num_layers, 2, batch, hidden)`으로 reshape해 layer 축과 direction 축을 분리한 뒤 마지막 정방향·역방향 state를 결합할 수 있다.

`transpose` 이후 tensor는 non-contiguous일 수 있으므로 `.view()` 전에 `.contiguous()`를 호출하거나 상황에 맞게 `.reshape()`를 사용하는 것이 안전하다. 이 연산들은 학습되지 않으며, 각 축의 의미를 유지한 채 다음 연산이 기대하는 배치 형태로 배열할 뿐이다.

### 3. Annotation은 어떻게 학습되는가?

$h_j$는 사람이 제공한 label이 아니라 encoder가 만든 hidden state이며, 번역 loss를 통해 end-to-end로 학습된다.

1. Encoder가 각 원문 위치의 annotation $h_j$를 만든다.
2. Decoder가 $s_{i-1}$과 $h_j$로 score $e_{ij}$를 계산한다.
3. Softmax가 score를 attention weight $\alpha_{ij}$로 바꾼다.
4. $c_i=\sum_j\alpha_{ij}h_j$로 context vector를 만든다.
5. Decoder가 context와 이전 단어·state를 사용해 다음 단어의 확률 분포를 예측한다.
6. 정답 단어의 negative log-likelihood가 decoder, attention module, encoder 전체로 역전파된다.

Alignment가 hard latent variable이 아니라 미분 가능한 soft alignment이므로, gradient는 attention scoring network뿐 아니라 annotation을 만든 BiRNN parameter까지 전달된다.

공식 GroundHog의 [`RecurrentLayerWithSearch`](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/encdec.py)도 같은 구조를 따른다. 원문 annotation과 이전 decoder state로 energy를 만들고, 정규화한 `probs`로 weighted context를 계산한다. 이 context가 decoder update에 들어가므로 번역 likelihood를 높이는 방향으로 annotation도 함께 학습된다.

### 4. RNNencdec-30/50과 RNNsearch-30/50의 비교는 공정한가?

같은 최대 문장 길이 안에서 attention 유무를 비교하기에는 합리적이지만, `-30`과 `-50` 자체의 효과를 분리하기에는 충분하지 않다.

`-30`과 `-50`은 학습에 포함한 최대 문장 길이를 뜻한다. 따라서 `RNNencdec-30` 대 `RNNsearch-30`, `RNNencdec-50` 대 `RNNsearch-50`은 같은 cutoff에서 attention의 효과를 비교하는 데 유용하다. 두 경우 모두 RNNsearch의 BLEU가 더 높다.

반면 `-30`과 `-50`을 직접 비교하면 학습 문장의 길이 분포와 난도가 달라진다. GroundHog의 `sort_k_batches`는 padding을 줄이기 위해 여러 minibatch를 길이순으로 묶는 전략이지, 길이 구간을 통제한 실험 설계는 아니다.

정리하면 다음과 같다.

- **합리적인 비교:** 같은 cutoff에서 `RNNencdec`와 `RNNsearch` 비교
- **주의가 필요한 비교:** `-30`과 `-50`의 직접 비교
- **핵심 해석:** RNNsearch는 고정 길이 벡터 병목을 줄이고 긴 문장에 더 강하지만, 30/50 설정은 문장 길이만 완전히 통제한 실험이 아니다.

### 5. 왜 alignment score는 현재 state가 아니라 이전 decoder state로 계산하는가?

논문의 alignment model은 $e_{ij}=a(s_{i-1},h_j)$처럼 이전 decoder state를 사용한다. $s_{i-1}$은 지금까지 생성한 목표 prefix를 요약하므로, 다음 단어를 만들기 위해 원문의 어느 위치가 필요한지 묻는 query 역할을 한다.

반대로 $s_i$를 곧바로 사용하면 순환 의존성이 생긴다. 현재 state $s_i$를 계산하려면 context $c_i$가 필요하고, $c_i$를 만들려면 $s_i$로 계산한 attention weight가 필요해진다. 이전 state를 사용하면 `이전 state → alignment → context → 현재 state`라는 계산 순서가 명확해지고 전체 과정이 미분 가능하게 유지된다.

## 참고문헌

### 후속 읽기

- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- Sutskever et al., "Sequence to Sequence Learning with Neural Networks"
- Vaswani et al., "Attention Is All You Need"

### 출처

- [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- [ar5iv HTML 버전](https://ar5iv.labs.arxiv.org/html/1409.0473)
- [GroundHog NMT 공식 구현 README](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/README.md)
- [GroundHog NMT state.py](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/state.py)
- [GroundHog NMT encdec.py](https://github.com/lisa-groundhog/GroundHog/blob/master/experiments/nmt/encdec.py)
