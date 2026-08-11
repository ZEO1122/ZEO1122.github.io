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

## 프로젝트 개요

모델 성능과 추론 효율을 함께 최적화하는 온라인 해커톤 [LG Aimers 8기](https://www.lgaimers.ai/)에 팀장으로 참가했습니다. LG AI연구원의 EXAONE-4.0-1.2B를 대상으로 실험했으며, Hugging Face 모델 형식과 고정된 추론 환경 등 DACON 방식의 제출 조건을 준수했습니다.

## 성과

**600여 팀 중 40위**

공개 리더보드 결과는 다음과 같습니다.

- 순위: 40위
- 점수: 0.63166
- 제출 횟수: 69회

## 접근 방법

- 팀원 간 의견을 조율하고 전체 실험 방향을 설정했습니다.
- 학습 메모리 사용량을 줄이면서 과제 적응 성능을 유지하기 위해 QLoRA 기반 지도 미세조정 전략을 설계했습니다.
- 프롬프트 토큰은 마스킹하고 어시스턴트 답변 토큰만 손실 계산에 사용하는 대화형 전처리를 적용했습니다.
- 데이터 포함 범위와 연산 비용을 비교해 학습 시퀀스 길이를 1,024 토큰으로 정했습니다.
- LoRA 어댑터 병합 후 GPTQ W8A8을 적용할 수 있는 선택적 후처리 양자화 경로를 준비했습니다.
- 제출 결과물을 Hugging Face 호환 모델 패키지로 구성했습니다.

## 링크

- [대회 저장소](https://github.com/ZEO1122/LG_Aimers_8th)

## 키워드

LLM 경량화 · QLoRA · GPTQ · EXAONE · Hugging Face · PyTorch · 해커톤
