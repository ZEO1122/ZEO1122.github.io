---
title: "LG Aimers 8th Model Compression"
excerpt: "Team-led LLM compression project using QLoRA and optional GPTQ W8A8 quantization for EXAONE-4.0-1.2B."
collection: portfolio
date: 2025-12-01
---

## Overview

Participated as team leader in [LG Aimers 8th](https://www.lgaimers.ai/), a model compression online hackathon focused on optimizing both model performance and inference efficiency. The project targeted LG AI Research's EXAONE-4.0-1.2B and followed DACON-style submission constraints, including Hugging Face model format and fixed inference-environment requirements.

## Achievement

**Ranked Top 40 out of 600+ teams**

Public portfolio result:

- Rank: 40th
- Score: 0.63166
- Submissions: 69

## Approach

- Led team coordination, aligned member opinions, and set the overall experiment direction
- Designed a strategy around QLoRA-based supervised fine-tuning to reduce training memory cost while preserving task adaptation
- Used conversation-style preprocessing where prompt tokens were masked and assistant-answer tokens were used for loss calculation
- Selected a 1024-token training sequence length based on coverage and compute trade-offs
- Prepared an optional post-training quantization path using GPTQ W8A8 after merging LoRA adapters
- Organized the output as a Hugging Face-compatible model package for submission

## Links

- [Competition Repository](https://github.com/ZEO1122/LG_Aimers_8th)

## Keywords

LLM Compression · QLoRA · GPTQ · EXAONE · Hugging Face · PyTorch · Hackathon
