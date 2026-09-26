---
title: "LG Aimers 9th: Pitch Control Success Prediction"
excerpt: "Team leadership, temporal validation, and conditional ensembles for pitch control prediction."
collection: portfolio
date: 2026-08-05
lang: en
locale: en-US
og_locale: en_US
author: jeo_en
translations:
  en: /portfolio/lg-aimers-9th/
  ko: /ko/portfolio/lg-aimers-9th/
---

**August 5-September 2, 2026 · Team Lead · 63rd place**

## Predicting Success Before the Pitch

I led team 떡잎마을방범대 in the LG Aimers 9th online hackathon. The task was to predict pitch control success using game context and player history available before a pitch, rather than classify an outcome from the ball's observed location.

The team explored tree-based models, deep-learning models, and ensembles. I managed members' schedules, reports and documents in Notion, meetings, and code reviews, offering suggestions while coordinating the project.

## My Role as Team Lead

Parallel experiments required shared visibility into progress: one member's predictions could become another member's ensemble inputs. I used meetings and Notion documents to keep track of the work.

| Responsibility | What I did |
|---|---|
| Scheduling | Checked members' availability and coordinated progress |
| Documentation | Managed reports and project documents in Notion |
| Meetings | Discussed progress, experiment results, and next steps |
| Code review | Reviewed members' code and suggested improvements |
| Project coordination | Followed individual experiments and overall progress |

During reviews, I looked at what had changed in the code and how the results differed from earlier experiments. These checks helped me make concrete suggestions for the next discussion.

## When Validation Gains Did Not Transfer to Submissions

A monthly adjustment was motivated by changes in player condition and game patterns over a season. Varying its strength by month improved an internal Validation score but performed worse than the baseline on submission.

The team did not adopt it. Patterns from one year's monthly data could not be assumed to repeat in the next season. We examined results by game type and used temporal splits: training on 2019-2021 and evaluating 2022, then expanding the training window to evaluate 2023 and 2024.

This taught me to ask whether a gain reflected a useful pattern or a method that fit only a particular period.

## Adjusting Ensemble Weights by Player History

XGBoost performed worse than LightGBM alone, but could still complement its errors in some conditions. Increasing XGBoost's weight for players with limited history increased errors for new pitchers. The team then shifted toward giving it more weight in Regular games with sufficient player history.

A model's standalone score was not enough to judge its value in an ensemble. We also needed to examine where its errors occurred and whether another model could compensate.

## Checking Consistency in the Submission Environment

Predictions needed to remain consistent when processing one row at a time, splitting batches, or changing input order. The team reused preprocessing settings learned from training data during Inference and added tests for these cases.

These checks were necessary to preserve the chosen model's behavior when moving from experiments to submission code.

## Results and Lessons

The team finished **63rd with a score of 1,162.43634**.

Leading the project taught me that coordinating schedules and experiments requires understanding the work itself. Reading both code and shared results helped me contribute specific suggestions in meetings.

Dropping the monthly adjustment and refining ensemble conditions also showed me the value of removing ineffective approaches. I now consider the data period, evaluation conditions, and model-specific errors alongside the overall score.

## Technologies

Python · pandas · NumPy · scikit-learn · PyTorch · CatBoost · LightGBM · XGBoost · Notion

## Resources

- [Project Notion](https://valiant-crowd-006.notion.site/LG-Aimers-9th-3b2b6928b66c81dd95f7e2b9dcde038c?pvs=74)
- [Project repository](https://github.com/ZEO1122/LG_Aimers_9th)
- [Modeling process](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/modeling.md)
- [Ensemble experiments](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/experiments.md)
- [Shared preprocessing and tests](https://github.com/ZEO1122/LG_Aimers_9th/tree/main/Preprocess)
