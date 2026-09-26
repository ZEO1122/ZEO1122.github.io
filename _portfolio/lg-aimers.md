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

## Balancing Model Performance and Inference Efficiency

I participated in the LG Aimers 8th online hackathon, experimenting with compression of EXAONE-4.0-1.2B. The task required considering both model performance and inference efficiency in a fixed vLLM environment, rather than reducing model size alone.

The approach used QLoRA to reduce training memory requirements, with an optional path to merge the trained Adapter and apply GPTQ Quantization. The pipeline connected conversational preprocessing to the creation of a Hugging Face-format submission.

## Choosing Which Response to Train On

Each conversation contains user questions, earlier responses, and the response to be learned. The final Assistant response served as the training target, while earlier turns provided the Prompt.

A Chat Template formatted the Prompt. Prompt and Padding Labels were set to -100 to exclude them from Loss calculation, leaving only the final Assistant response as the target. EOS marked the end of that response. This made the training objective specific: generate a response given the preceding conversation.

## Trading Data Coverage Against Training Cost

Longer sequences preserve more context but require more memory and computation. Length analysis of 3,000 samples helped quantify how much data each limit could retain.

| Maximum Sequence Length | Samples fitting without truncation |
|---|---|
| 512 Tokens | 13.23% |
| 1,024 Tokens | 80.47% |
| 2,048 Tokens | 99.53% |

The training limit was set to 1,024 Tokens to balance coverage and experiment cost. For overlength samples, preprocessing retained the tail of the Prompt and placed the response in the remaining space. Sequence Length therefore determined how much context and response text actually reached training.

## Training with QLoRA Under Memory Constraints

The Base Model was loaded in 4-bit NF4, with trainable LoRA Adapters rather than updates to all model Parameters. Gradient Checkpointing and Gradient Accumulation helped manage memory requirements.

Before preprocessing, 3,000 sampled records were split into 2,850 training and 150 validation examples. Evaluation and logging were configured to track Training Loss and Validation Loss.

QLoRA addressed training resource requirements. Reducing training memory and improving the submitted model's inference efficiency were treated as separate concerns.

## From Adapter Merging to Submission Packaging

An optional post-training stage merged the saved Adapter into the Base Model and applied GPTQ W8A8. Calibration used Prompts with the final Assistant response removed.

Quantization targeted Linear layers while excluding Embedding and the LM Head. The pipeline then saved the model and Tokenizer in Hugging Face format and packaged them into the required ZIP directory structure.

## Results and Lessons

The team finished **40th with a score of 0.63166**.

The project gave me experience addressing training and inference efficiency at different stages. I connected memory-efficient QLoRA training with a separate post-training Quantization path and submission packaging.

Preprocessing also required considering sequence limits together with the Loss mask. Which context is retained and which response tokens become targets determine what the model learns, even when the source conversations are unchanged.

## Technologies

Python · PyTorch · Transformers · PEFT · bitsandbytes · QLoRA · GPTQ · llmcompressor

## Resources

- [Project repository](https://github.com/ZEO1122/LG_Aimers_8th)
- [Training and Quantization code](https://github.com/ZEO1122/LG_Aimers_8th/blob/main/qlora_baseline_3000_val_local.py)
