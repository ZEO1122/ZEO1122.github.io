---
title: "LG Aimers 9th: Pitch Control Success Prediction"
excerpt: "A project that predicts the probability of pitch control success for the next pitch using game context and player history."
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

The team began by examining baseball game context and player histories, then worked through preprocessing, feature engineering, individual model comparisons, and ensemble experiments. I managed members' schedules, reports and documents in Notion, meetings, and code reviews, offering suggestions while coordinating the project.

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

## Understanding the Baseball Domain and Available Information

We examined what ball-strike counts, outs, baserunners, and score differences represent in a game. A player's control probability can vary with the situation, so we considered current game context alongside historical performance.

Player histories required more than reading the success rate alone. The same rate based on a few observations and on a long history should not carry the same confidence. Missing history also differs from an observed low success rate. We compared the previous one, three, and five games with longer-term records to investigate recent changes.

Information timing constrained the inputs. We excluded the current pitch's observed location and outcome, using only pre-pitch context and historical aggregates. The main and TrackMan datasets used different player ID systems, so matching numeric IDs were not treated as a valid join.

## Data Preprocessing and Feature Engineering

History-availability indicators and observation counts distinguished missing information from observed values. For categorical inputs, the team experimented with pooling rare IDs and handling unseen IDs separately. Statistics and category dictionaries were fitted on each fold's training data and reused for validation.

Feature engineering translated domain hypotheses into inputs that could be tested.

| Feature group | Construction and comparison |
|---|---|
| Long-term and season history | Pitcher and batter success rates with observation counts, shrinking small-sample estimates toward a prior |
| Recent form | Previous one, three, and five games, including differences from long-term success rates |
| Game context | Ball-strike combinations, outs, baserunners, and score differences |
| Matchups | Pitcher-batter handedness and team/opponent interactions |
| History reliability | History availability, sample counts, and availability of recent records |

More features did not always help. We added or removed groups and checked whether a feature set that worked for one model transferred to another. For example, a reduced feature set selected with DCN improved CatBoost's pooled result but worsened some individual years, so it was not adopted.

## Individual Model Experiments

The team experimented with CatBoost, LightGBM, and XGBoost, alongside tabular deep-learning models including FT-Transformer, RealMLP, and SAINT. Within each family, we examined how inputs and training choices affected predictions.

| Model family | Main experiments |
|---|---|
| CatBoost, LightGBM, XGBoost | Direct probability prediction versus residual learning, recent-season weights, rare-category pooling, and feature sets |
| FT-Transformer | Providing the player-history baseline as an input versus adding a bounded correction to its probability |
| RealMLP | Combining game-context and recent-form features with baseline probabilities and tree-model predictions |
| SAINT | Examining feature attention and the effect of attention across evaluation rows |

We compared predicting the full probability with correcting only the errors of a player-history baseline. The question was whether a model captured useful additional information, not simply whether it was larger or used more inputs.

## Temporal Validation and Monthly Adjustments

A monthly adjustment was motivated by changes in player condition and game patterns over a season. Varying its strength by month improved an internal Validation score but performed worse than the baseline on submission.

The team did not adopt it. Patterns from one year's monthly data could not be assumed to repeat in the next season. We examined results by game type and used temporal splits: training on 2019-2021 and evaluating 2022, then expanding the training window to evaluate 2023 and 2024.

This taught me to ask whether a gain reflected a useful pattern or a method that fit only a particular period.

## Adjusting Ensemble Weights by Player History

After examining individual models, we combined their predictions through fixed probability blends, weights conditioned on game type and player history, and small residual corrections to existing predictions.

XGBoost performed worse than LightGBM alone, but could still complement its errors in some conditions. Increasing XGBoost's weight for players with limited history increased errors for new pitchers. The team then shifted toward giving it more weight in Regular games with sufficient player history.

A model's standalone score was not enough to judge its value in an ensemble. We also needed to examine where its errors occurred and whether another model could compensate.

## Checking Consistency in the Submission Environment

Predictions needed to remain consistent when processing one row at a time, splitting batches, or changing input order. The team reused preprocessing settings learned from training data during Inference and added tests for these cases.

These checks were necessary to preserve the chosen model's behavior when moving from experiments to submission code.

## Results and Lessons

The team finished **63rd with a score of 1,162.43634**.

Leading the project taught me that coordinating schedules and experiments requires understanding the work itself. Reading both code and shared results helped me contribute specific suggestions in meetings.

Domain understanding guided preprocessing and model experiments: how to represent missing history, how much to trust recent records, and when to combine models depended on what the data meant. Comparing hypotheses and removing ineffective approaches mattered more than adding features or complexity alone.

## Technologies

Python · pandas · NumPy · scikit-learn · PyTorch · CatBoost · LightGBM · XGBoost · Notion

## Resources

- [Project Notion](https://valiant-crowd-006.notion.site/LG-Aimers-9th-3b2b6928b66c81dd95f7e2b9dcde038c?pvs=74)
- [Project repository](https://github.com/ZEO1122/LG_Aimers_9th)
- [Modeling process](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/modeling.md)
- [Ensemble experiments](https://github.com/ZEO1122/LG_Aimers_9th/blob/main/docs/experiments.md)
- [Shared preprocessing and tests](https://github.com/ZEO1122/LG_Aimers_9th/tree/main/Preprocess)
