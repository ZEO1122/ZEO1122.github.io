---
title: "LG Aimers 9th: Pitch Control Success Prediction"
excerpt: "Team-led baseball probability prediction project with chronological validation, tree-based and deep-learning models, and ensemble experiments."
collection: portfolio
date: 2026-09-02
lang: en
locale: en-US
og_locale: en_US
author: jeo_en
translations:
  en: /portfolio/lg-aimers-9th/
  ko: /ko/portfolio/lg-aimers-9th/
---

## Overview

I participated as **team leader** in the LG Aimers 9th online hackathon, leading team **떡잎마을방범대** on a pitch control success prediction task. Phase 2 ran from August 5 to September 2, 2026, hosted by LG AI Research and organized by DACON.

The task was to estimate the probability of successful pitch control using game context and player history available **before the pitch**. This was not simply strike classification: the competition's target also accounted for pitches missing the intended location. The team explored data interpretation, feature engineering, machine-learning and deep-learning models, and ensemble strategies.

## Achievement

**63rd place · Score 1,162.43634**

I also completed **LG Aimers 9th Phase 1 & 2**, an 11-week program from June 22 to September 2, 2026.

## Approach

- **Team leadership:** Led the team through the pitch control prediction project, working together on modeling and ensemble experiments.
- **Data analysis:** Examined missing player histories, season shifts, game types, and unseen players. Distinguished unavailable history from a low historical success rate.
- **Feature integrity:** Checked when information became available and avoided directly joining player IDs across the main and TrackMan datasets without establishing correspondence.
- **Model comparison:** Compared CatBoost, LightGBM, and XGBoost with tabular deep-learning approaches including FT-Transformer, RealMLP, SAINT, and TabM.
- **Ensemble experiments:** Explored residual learning, batter-handedness effects, and conditional blending to assess whether models complemented one another's errors.
- **Submission preparation:** Checked inference-package execution separately from model quality, keeping executable but weaker configurations as alternatives rather than claiming them as final improvements.

## Validation and Lessons

The team used chronological validation: training through 2021, 2022, and 2023 and evaluating on 2022, 2023, and 2024, respectively. Evaluation considered Brier error, performance by game type, player-history coverage, and variation across random seeds. Internal validation and leaderboard scores were kept separate.

A monthly residual adjustment improved an internal score but hurt the actual submission result, so it was not adopted. Conditional blending was also evaluated by player-history coverage rather than applying the same correction to every player. These experiments highlighted the importance of checking whether a local improvement generalizes to later seasons.

The main lesson was to distinguish a strong standalone model from a useful ensemble component, and a working submission package from a model supported by sufficient performance evidence. Repeatedly inspected 2024 results were treated as retrospective checks, not independent unseen evaluations.

## Links

- [Competition repository](https://github.com/ZEO1122/LG_Aimers_9th)
- [Modeling process](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/modeling.md)
- [Ensemble experiments](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/experiments.md)
- [Results and retrospective](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/results.md)

## Keywords

Sports Analytics · Probability Prediction · CatBoost · LightGBM · XGBoost · PyTorch · Tabular Deep Learning · Ensemble · Team Leadership
