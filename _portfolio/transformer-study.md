---
title: "Transformer Study Materials"
excerpt: "Connecting Transformer concepts with PyTorch implementation and small experiments."
collection: portfolio
date: 2026-02-13
lang: en
locale: en-US
og_locale: en_US
author: jeo_en
translations:
  en: /portfolio/transformer-study/
  ko: /ko/portfolio/transformer-study/
---

**Started February 2026 · Study Materials Lead**

## From Reading the Paper to Implementing the Model

I led the production of Transformer study materials for A.ing's junior track. The aim was to help members explain why each operation is needed and how it appears in code, rather than only use a pretrained model.

The paper's equations can be difficult to map to code. Even after understanding the architecture diagram, implementation requires tracking operation order and tensor dimensions. I organized the materials around prerequisites, paper reading, implementation, and comparison experiments.

## Connecting Explanations and Code

Cheat sheets, a PyTorch cookbook, notebooks, and quizzes serve different purposes. If each points only to the paper, learners still have to locate the relevant code themselves.

The paper guide serves as a shared reference. Its R01-R13 sections bring together paper locations, questions, explanations, and links to other materials. Those materials link back to the corresponding guide sections.

For Multi-Head Attention, learners can read the paper and cheat sheet, inspect code that splits and recombines heads in the cookbook, then complete the relevant exercise. Links lead back to the explanation when they get stuck.

I also distinguished reading order from implementation order. Input Embeddings and position information appear after Attention in the paper, but are introduced earlier along the implementation path.

## Checking Tensor Shapes and Masking Step by Step

Implementing Attention requires identifying batch, sequence, head, and feature dimensions. Self-Attention and Cross-Attention also have different relationships between input lengths. A mask can have the expected shape while blocking the wrong positions.

The notebook contains 36 blanks to implement. Checks cover input and output dimensions, finite values, Cross-Attention lengths, and padding behavior.

When a check fails, the material points to the relevant exercise and cookbook section rather than only displaying an answer. The cookbook explains dimensions, operations, and common mistakes alongside API usage.

## Small Experiments Without Model Training

Downloading translation data and training a model can distract from learning individual operations. A final score also says little about what each design choice does.

I added small experiments that run without dataset downloads or model training.

| Experiment | Comparison | Question |
|---|---|---|
| Scaling | Score variance and softmax distributions before and after scaling at different head dimensions | Why scale the dot product? |
| Position information | Encoder outputs after reordering inputs, with and without position information | Why does Attention need position information? |

These experiments let learners change a condition and inspect the output before training a translation model.

## Comparing Models Under Shared Conditions

The Multi30k translation exercise lets learners change the architecture and training settings. Changing data and evaluation rules at the same time would make results harder to interpret.

The notebook separates FIXED conditions from TUNE settings. Data splits and evaluation rules remain shared, while permitted architecture and training choices can vary. Model selection using Validation is kept separate from final Test evaluation.

The goal is to explain what changed and how it affected results, not just obtain the highest score.

## Materials Produced

The materials connect prerequisites, a paper guide, cheat sheets, a PyTorch cookbook, blank and solution notebooks, quizzes and answers, and translation exercises.

**Prerequisites → paper and concept mapping → implementation exercises → stepwise checks → small comparisons → translation experiments and interpretation**

Each resource points to what to read or implement next, with links back to earlier explanations.

## Lessons

Explaining familiar operations such as `reshape` and Masking made me revisit their details: what each axis means, when an operation applies, and what happens if it is used incorrectly.

Building the materials strengthened my own understanding of the connection between equations and code. It also taught me that approachable explanations need examples and checks that help learners find their own mistakes.

## Technologies

Python · PyTorch · Jupyter Notebook · Multi30k · Markdown

## Resources

- [Study repository](https://github.com/aing-gachon/26-Spring-Transformer-Study)
- [Paper guide](https://github.com/aing-gachon/26-Spring-Transformer-Study/blob/main/Week1/transformer_paper_guide.md)
- [PyTorch cookbook](https://github.com/aing-gachon/26-Spring-Transformer-Study/blob/main/Week2/A.ing_Transformer_Cookbook.md)
- [Implementation exercises](https://github.com/aing-gachon/26-Spring-Transformer-Study/blob/main/Week2/A.ing_Transformer_from_scratch_blank.ipynb)
