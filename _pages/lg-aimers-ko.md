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

## 모델 성능과 추론 효율을 함께 고려한 경량화

LG Aimers 8기 온라인 해커톤에 참가해 EXAONE-4.0-1.2B의 경량화를 실험했습니다. 과제는 모델 크기를 줄이는 데 그치지 않고, 고정된 vLLM 추론 환경에서 모델 성능과 추론 효율을 함께 고려하는 것이었습니다.

학습 단계에서는 QLoRA로 메모리 부담을 줄이고, 학습 이후에는 Adapter를 병합해 GPTQ Quantization으로 이어갈 수 있도록 구성했습니다. 대화 데이터의 학습 구간을 정하는 전처리부터 Hugging Face 형식의 제출물 생성까지 연결했습니다.

## 대화에서 학습할 응답 구분

대화 데이터에는 사용자의 질문과 이전 응답, 새로 학습할 응답이 함께 들어 있습니다. 이 중 마지막 Assistant 응답을 학습 대상으로 정하고, 앞선 대화는 응답을 생성하기 위한 Prompt로 사용했습니다.

Prompt에는 Chat Template을 적용하고, Prompt와 Padding의 Label을 -100으로 설정해 Loss 계산에서 제외했습니다. 마지막 Assistant 응답에 대해서만 Loss를 계산하고 EOS를 붙여 응답의 끝을 처리했습니다. 단순히 대화 전체를 입력하는 대신, 주어진 문맥에 이어 어떤 응답을 생성해야 하는지 학습하도록 구성한 것입니다.

## 데이터 길이와 학습 비용 사이의 선택

긴 대화를 그대로 사용하면 문맥을 더 많이 보존할 수 있지만, 학습에 필요한 메모리와 계산량도 늘어납니다. 3,000개 샘플의 길이 분포를 살펴 어느 정도의 데이터를 보존할 수 있는지 확인했습니다.

| 최대 Sequence Length | 전체 대화를 자르지 않고 포함하는 샘플 비율 |
|---|---|
| 512 Tokens | 13.23% |
| 1,024 Tokens | 80.47% |
| 2,048 Tokens | 99.53% |

학습 길이는 데이터 보존 범위와 실험 비용을 고려해 1,024 Tokens로 정했습니다. 길이를 초과하는 경우에는 Prompt의 뒤쪽 문맥을 남기고, 남은 공간에 응답을 배치하도록 처리했습니다. Sequence Length는 단순한 설정값이 아니라 실제로 학습에 들어가는 문맥과 응답의 양을 결정하는 요소였습니다.

## 제한된 메모리에서 QLoRA로 학습

Base Model은 4-bit NF4로 불러오고 LoRA Adapter를 학습하도록 구성했습니다. 전체 Parameter를 업데이트하는 대신 학습 대상을 Adapter로 제한하고, Gradient Checkpointing과 Gradient Accumulation을 함께 사용해 메모리 부담을 조절했습니다.

데이터는 3,000개를 추출해 전처리 전에 학습용 2,850개와 검증용 150개로 나눴습니다. 학습 중 Training Loss와 Validation Loss를 확인할 수 있도록 평가와 로그 출력을 구성했습니다.

여기서 QLoRA는 학습 자원을 절약하기 위한 선택이었습니다. 학습 메모리를 줄이는 것과 실제 제출 모델의 추론 효율을 높이는 것은 구분해서 다뤘습니다.

## Adapter 병합부터 제출물 구성까지

학습 이후에는 저장한 Adapter를 Base Model에 병합하고, 필요에 따라 GPTQ W8A8을 적용할 수 있도록 구현했습니다. Calibration에는 마지막 Assistant 응답을 제외한 Prompt를 사용했습니다.

Quantization은 Linear 계층을 대상으로 하되 Embedding과 LM Head는 제외하도록 설정했습니다. 이후 모델과 Tokenizer를 Hugging Face 형식으로 저장하고, 제출에 필요한 디렉터리 구조를 갖춘 ZIP 파일을 생성하도록 연결했습니다.

## 결과와 배운 점

팀은 대회에서 **40위, 0.63166점**을 기록했습니다.

프로젝트를 통해 학습 효율과 추론 효율을 서로 다른 단계에서 고려하는 경험을 쌓았습니다. QLoRA로 학습 부담을 줄이는 접근과 학습 이후의 Quantization을 구분하고, 두 과정을 제출물 생성까지 연결했습니다.

데이터 전처리에서도 길이 제한과 Loss 계산 구간을 함께 봐야 했습니다. 같은 대화 데이터라도 어떤 문맥을 남기고 어느 응답을 학습 대상으로 삼는지에 따라 모델이 학습하는 내용이 달라진다는 점을 배웠습니다.

## 사용 기술

Python · PyTorch · Transformers · PEFT · bitsandbytes · QLoRA · GPTQ · llmcompressor

## 관련 자료

- [프로젝트 저장소](https://github.com/ZEO1122/LG_Aimers_8th)
- [학습 및 Quantization 코드](https://github.com/ZEO1122/LG_Aimers_8th/blob/main/qlora_baseline_3000_val_local.py)
