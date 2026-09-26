---
permalink: /ko/portfolio/lg-aimers/
title: "LG Aimers 8기 모델 경량화"
description: "EXAONE-4.0-1.2B를 대상으로 진행한 팀 기반 LLM 경량화 프로젝트입니다."
date: 2025-12-01
author_profile: true
lang: ko
locale: ko-KR
og_locale: ko_KR
author: jeo_ko
translations:
  en: /portfolio/lg-aimers/
  ko: /ko/portfolio/lg-aimers/
---

**활동 기간: 2025.12~2026.02 · 역할: 팀원 · 성과: 40위**

## 성능을 유지하면서 모델을 경량화하기

LG Aimers 8기에서 EXAONE-4.0-1.2B를 대상으로 모델 경량화 프로젝트에 참여했습니다. 모델의 크기를 줄이는 것뿐 아니라, 응답 성능과 추론 효율을 함께 고려해야 하는 과제였습니다.

프로젝트에서는 제한된 자원으로 모델을 학습시키는 방법과 학습한 모델에 Quantization을 적용하는 방법을 나누어 접근했습니다. QLoRA를 활용한 Fine-Tuning을 실험하고, 대화 데이터의 학습 구간과 길이를 조정하면서 주어진 학습 자원을 어떻게 사용할지 고민했습니다.

## 대화 전체가 아니라 응답을 학습 대상으로

먼저 대화 데이터에서 모델이 무엇을 학습해야 하는지 구분했습니다. 이전 대화는 마지막 응답을 이해하는 데 필요한 문맥이므로 입력으로 제공하되, Loss는 마지막 Assistant 응답에 대해서만 계산하도록 구성했습니다.

이를 위해 Prompt와 Padding을 Loss 계산에서 제외했습니다. 같은 데이터를 사용하더라도 모든 문장을 학습 대상으로 삼는 것과, 문맥에 맞는 응답을 생성하도록 학습시키는 것은 다르다고 보았기 때문입니다. 데이터 형식을 맞추는 데 그치지 않고, 학습 목적에 맞게 입력과 학습 대상을 나누는 데 초점을 뒀습니다.

## 더 긴 입력보다 학습 비용과 데이터 보존 범위를 고려

대화 길이가 서로 달라 학습에 사용할 Sequence Length를 정해야 했습니다. 길이를 늘리면 더 많은 내용을 보존할 수 있지만, 메모리 사용량과 계산 비용도 함께 증가했습니다.

3,000개 샘플의 길이를 분석했을 때, 1,024 Tokens에서는 약 80%의 대화를 자르지 않고 포함할 수 있었고, 2,048 Tokens에서는 약 99.5%까지 포함할 수 있었습니다. 프로젝트에서는 데이터 보존 범위와 실험 비용을 고려해 1,024 Tokens를 학습 길이로 선택했습니다.

이 과정에서 Sequence Length를 단순히 크게 설정하는 것보다, 제한된 길이 안에 어떤 문맥과 응답이 남는지 함께 살펴보는 것이 중요했습니다. 특히 응답에 대해서만 Loss를 계산하는 구조에서는 입력 길이뿐 아니라 실제 학습에 사용되는 응답의 양도 고려해야 했습니다.

## QLoRA로 학습 부담을 줄이고 Quantization으로 확장

학습 단계에서는 Base Model을 4-bit NF4로 불러오고 LoRA Adapter를 학습하는 QLoRA 방식을 사용했습니다. 전체 Parameter를 업데이트하는 대신 학습 대상을 줄이고, Gradient Checkpointing과 Gradient Accumulation을 함께 적용해 메모리 부담을 조절했습니다.

또한 학습 데이터와 검증 데이터를 분리하고 Training Loss와 Validation Loss를 확인할 수 있도록 구성했습니다. 학습에 사용한 데이터에 잘 맞는 것과 새로운 데이터에서도 응답을 잘 생성하는 것을 구분해서 살펴보기 위해서였습니다.

학습 이후에는 Adapter를 Base Model에 병합하고 GPTQ W8A8을 적용할 수 있도록 구현했습니다. QLoRA는 학습에 필요한 자원을 줄이는 방법으로, GPTQ는 학습 이후 모델의 추론 효율을 고려하는 방법으로 구분해 접근했습니다.

## 결과와 배운 점

팀은 대회에서 **40위, 0.63166점**을 기록했습니다.

이번 프로젝트를 통해 모델 경량화는 Quantization 설정 하나를 선택하는 일이 아니라, 데이터 구성과 학습 방식, 추론 단계를 함께 고려하는 과정임을 배웠습니다. 특히 학습 메모리를 줄였다는 사실이 곧 추론 속도 개선을 의미하지는 않으므로, 각 방법이 어느 단계의 비용을 줄이는지 구분하는 것이 중요했습니다.

대화 데이터의 전처리 역시 학습 결과를 좌우하는 선택이었습니다. 모델에 어떤 문맥을 보여주고 어느 응답에 대해 Loss를 계산할지 결정하면서, 모델 설정뿐 아니라 데이터가 학습에 사용되는 방식까지 살펴보는 경험을 쌓았습니다.

## 사용 기술

Python · PyTorch · Transformers · PEFT · bitsandbytes · QLoRA · GPTQ · llmcompressor

## 관련 자료

- [프로젝트 저장소](https://github.com/ZEO1122/LG_Aimers_8th)
- [학습 및 Quantization 코드](https://github.com/ZEO1122/LG_Aimers_8th/blob/main/qlora_baseline_3000_val_local.py)
