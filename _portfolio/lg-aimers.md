---
title: "LG Aimers 8th Model Compression"
excerpt: "EXAONE-4.0-1.2B compression with conversational preprocessing, QLoRA training, and an optional GPTQ W8A8 pipeline."
collection: portfolio
date: 2025-12-01
lang: en
locale: en-US
og_locale: en_US
author: jeo_en
translations:
  en: /portfolio/lg-aimers/
  ko: /ko/portfolio/lg-aimers/
---

**December 2025-February 2026 · Team Member · 40th place**

## Compressing a Model While Preserving Performance

I participated in an EXAONE-4.0-1.2B model compression project for LG Aimers 8th. The task required considering response quality and inference efficiency together, rather than reducing model size alone.

We approached resource-efficient training and post-training Quantization as separate concerns. I experimented with QLoRA Fine-Tuning and considered how to use the available training resources by adjusting which parts of conversations became training targets and how much text each input retained.

## Training on Responses Rather Than Entire Conversations

The first decision was what the model should learn from each conversation. Earlier turns provided the context needed to understand the final response, but Loss was calculated only on the final Assistant response.

Prompt and Padding tokens were excluded from Loss calculation. Training on every sentence is different from training a model to respond appropriately to the preceding context, even with the same source data. The focus was therefore on separating inputs and targets to match the learning objective, rather than simply formatting the data.

## Balancing Training Cost and Data Coverage

Conversations varied in length, so the training Sequence Length required a deliberate choice. Longer inputs preserved more text but increased memory use and computation.

Analysis of 3,000 samples showed that approximately 80% of conversations fit within 1,024 Tokens without truncation, compared with approximately 99.5% at 2,048 Tokens. The project used a training length of 1,024 Tokens to balance data coverage and experiment cost.

This made it important to consider which context and response tokens remained within the limit, rather than simply choose a larger Sequence Length. Because Loss was calculated only on responses, the amount of response text available for learning mattered alongside the total input length.

## Reducing Training Requirements with QLoRA and Extending to Quantization

For training, I used QLoRA to load the Base Model in 4-bit NF4 and train LoRA Adapters. Instead of updating all model Parameters, this reduced the trainable Parameter set. Gradient Checkpointing and Gradient Accumulation further helped manage memory requirements.

Training and validation data were separated, with logging configured to track Training Loss and Validation Loss. This provided a way to distinguish fitting the training examples from generating responses on unseen data.

After training, I implemented a path to merge the Adapter into the Base Model and apply GPTQ W8A8. QLoRA addressed training resource requirements, while GPTQ provided a separate approach to considering inference efficiency after training.

## Results and Lessons

The team finished **40th with a score of 0.63166**.

The project taught me that model compression involves data preparation, training choices, and inference, rather than a single Quantization setting. Reducing training memory does not itself imply faster inference, so it was important to distinguish which stage each method addressed.

Conversational preprocessing was also a consequential choice. Deciding which context to show the model and which response tokens to include in Loss calculation gave me experience examining how data is used for learning, not just how the model is configured.

## Technologies

Python · PyTorch · Transformers · PEFT · bitsandbytes · QLoRA · GPTQ · llmcompressor

## Resources

- [Project repository](https://github.com/ZEO1122/LG_Aimers_8th)
- [Training and Quantization code](https://github.com/ZEO1122/LG_Aimers_8th/blob/main/qlora_baseline_3000_val_local.py)
