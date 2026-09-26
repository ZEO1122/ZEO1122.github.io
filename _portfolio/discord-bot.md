---
title: "A.ing Discord Learning Support Bot"
excerpt: "Learning content and a Discord bot for foundational study and research exploration."
collection: portfolio
date: 2026-09-13
lang: en
locale: en-US
og_locale: en_US
author: jeo_en
translations:
  en: /portfolio/discord-bot/
  ko: /ko/portfolio/discord-bot/
---

**Started September 2026 · Learning Content and Service Design**

## Supporting Different Learning Goals

I designed a Discord bot to help junior A.ing members learn Deep Learning fundamentals and senior members explore research interests. Discord offered a familiar place for members to receive study materials.

I separated foundational lessons from recent-paper summaries so that each channel could use an appropriate level of explanation and delivery schedule.

## Separate Channels for Different Backgrounds

Recent papers can be difficult to approach without foundational knowledge, while introductory explanations alone offer limited support for exploring research interests.

| | Foundational lessons | Recent papers |
|---|---|---|
| Purpose | Learn concepts in sequence | Explore research interests |
| Content | 36 lessons across 12 weeks | Top three weekly papers from Hugging Face |
| Explanation | Motivation, mechanics, examples, misconceptions, review questions | Background, prior methods, key approach, results, limitations |
| Planned schedule | Three lessons per week | One research update per week |

The curriculum progresses from tensors and matrix operations through neural-network training, CNNs, RNNs, Attention, and Transformers. Each lesson builds on earlier concepts.

Paper explanations are in Korean, with technical terms retained in English so members can recognize them when reading the originals.

## Fixing Quotations That Did Not Match the Source

Generated quotations sometimes did not match the original paper. Fluent summaries alone were not enough for learning materials.

I assigned identifiers to collected source sentences and changed the pipeline so the model selects supporting sentences rather than writes quotations. The selected identifiers are checked before the original text is attached. Summary generation and quotation retrieval are separate steps.

Foundational lessons use prewritten Markdown rather than newly generated explanations each time. The manuscripts include learning objectives, prerequisites, explanations, review questions, and references. Separate checks verify numerical examples and Discord message length limits.

New papers are collected and summarized; recurring foundational content is maintained as reviewed manuscripts.

## Handling Delivery Failures and Semester Changes

I used Cloudflare Workers, Workflows, and D1 to avoid maintaining an always-on server.

If a delivery request receives no response, the message may still have been sent. Retrying unconditionally could create duplicates. The pipeline records a delivery reservation before sending and does not automatically resend requests with uncertain outcomes.

For foundational lessons, a semester uses a fixed content release, and breaks can shift the schedule. Editing a manuscript therefore does not immediately change what an active semester delivers.

## Implementation

The project includes 36 foundational lessons and a paper collection, summarization, and delivery pipeline, along with manuscript checks, previews, semester registration, and delivery-history management.

- Separate channels for foundational concepts and recent papers
- Paper summaries linked to source sentences
- Markdown checks and Discord message conversion
- Delivery-history-based duplicate control
- Semester schedules and study breaks
- Documentation for content maintenance and handover

## Lessons

Defining what each group needed helped me decide the depth, sequence, writing method, and schedule of the content.

I also learned that a natural explanation does not guarantee an accurate quotation. Learning materials need a way to verify their claims, not just readable prose.

Designing the bot meant considering more than successful delivery: uncertain responses, exam breaks, and handover to the next maintainer also shaped the implementation.

## Technologies

TypeScript · Cloudflare Workers · Workflows · D1 · OpenAI API · Discord Webhook · Markdown · Python

## Resources

- [Project repository](https://github.com/aing-gachon/A.ing-Discord-Bot)
- [Learning content](https://github.com/aing-gachon/A.ing-Discord-Bot/tree/main/content/dl-foundations)
- [Service architecture](https://github.com/aing-gachon/A.ing-Discord-Bot/blob/main/docs/ARCHITECTURE.md)
