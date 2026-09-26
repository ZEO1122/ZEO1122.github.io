---
title: "A.ing Discord Learning Support Bot"
excerpt: "A Discord bot that supports learning Deep Learning fundamentals and exploring research interests through recent papers."
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

## Different Content Pipelines for Different Purposes

Foundational lessons need consistent explanations that later lessons can build on. Instead of generating a new explanation at delivery time, I organized Markdown manuscripts with objectives, prerequisites, examples, and review questions. A lesson on Tensor Shape, for example, asks members to calculate sums along different Axes in a small array and interpret the results.

Manuscripts are checked and converted into Discord messages before deployment. Checks cover numerical examples and message lengths, and semester registration uses a reviewed content release. Delivery uses the compiled content without generating a new explanation or downloading a manuscript.

The paper channel serves a different purpose: introducing new research over time. It collects the top three weekly papers from Hugging Face and summarizes the research problem, prior approaches, and key method from the source text. Rather than receiving only titles and links, members can first understand the problem a paper addresses and then follow the original if it interests them.

## Connecting Explanations to Source Evidence

A learning summary needs more than fluent prose: readers should be able to check its supporting text. I assigned identifiers to collected source sentences so the model could select evidence from them.

Instead of asking the model to write quotations, the pipeline retrieves the original sentence for each selected identifier. Results and comparisons with prior methods are also linked to evidence, and summaries with invalid evidence selections are rejected during validation.

## Making the Bot Reusable Across Semesters

I used Cloudflare Workers, Workflows, and D1 to avoid maintaining an always-on server.

Workflows handle paper collection and summarization, while D1 stores processing results and delivery history. To limit duplicate notifications, the pipeline records a reservation before sending and does not automatically resend requests with uncertain outcomes.

For foundational lessons, a semester uses a fixed content release, and breaks can shift the schedule. Editing a manuscript therefore does not immediately change what an active semester delivers.

## Implementation

The project includes 36 foundational lessons and a paper collection, summarization, and delivery pipeline, along with manuscript checks, previews, semester registration, and delivery-history management.

Foundational lessons follow a reviewed sequence, while recent-paper summaries connect explanations to collected source evidence. Maintenance documentation covers content updates and semester setup so the next maintainer can continue the work.

## Lessons

Defining what each group needed helped me decide the depth, sequence, writing method, and schedule of the content.

Not every type of content needed automatic generation. Maintaining reusable foundational explanations as manuscripts and automating the collection and summarization of new papers helped me choose methods that fit each learning goal. Linking paper explanations to source evidence provided a way to check the content as well as read it.

Designing the bot meant considering more than successful delivery: uncertain responses, exam breaks, and handover to the next maintainer also shaped the implementation.

## Technologies

TypeScript · Cloudflare Workers · Workflows · D1 · OpenAI API · Discord Webhook · Markdown · Python

## Resources

- [Project repository](https://github.com/aing-gachon/A.ing-Discord-Bot)
- [Learning content](https://github.com/aing-gachon/A.ing-Discord-Bot/tree/main/content/dl-foundations)
- [Service architecture](https://github.com/aing-gachon/A.ing-Discord-Bot/blob/main/docs/ARCHITECTURE.md)
