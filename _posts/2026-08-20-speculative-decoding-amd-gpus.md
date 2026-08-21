---
layout: post
title: "Exploring Speculative Decoding in vLLM on AMD GPUs"
author: "AMD and Embedded LLM"
summary: "A practical guide to speculative decoding in vLLM on AMD GPUs, covering draft-and-verify mechanics, MTP, EAGLE-3, DFlash, DSpark, configuration, tuning, and benchmark results."
image: /assets/figures/2026-08-20-speculative-decoding-amd-gpus/figure-01.svg
tags:
  - speculative-decoding
  - amd
---

<style>

.config-table td {
    vertical-align: middle;
}

.config-table code {
    white-space: nowrap;
}

.config-table__config {
    line-height: 1.55;
}

.experiment-coverage-table {
    width: 100%;
    max-width: 100%;
    margin-right: auto;
    margin-left: auto;
    table-layout: fixed;
    font-size: 0.88rem;
}

.experiment-coverage-table th,
.experiment-coverage-table td {
    vertical-align: middle;
    padding-right: 0.7rem;
    padding-left: 0.7rem;
}

.experiment-coverage-table th:first-child,
.experiment-coverage-table td:first-child,
.experiment-coverage-table__target {
    white-space: nowrap;
}

.experiment-coverage-table th:first-child {
    width: 31%;
}

.experiment-coverage-table th:not(:first-child) {
    width: 13.8%;
}

.experiment-coverage-table__entry {
    display: inline-flex;
    flex-direction: column;
    gap: 0.1rem;
    align-items: flex-start;
    line-height: 1.2;
    white-space: nowrap;
}

.experiment-coverage-table__dash {
    color: #475569;
}

.key-question {
    border-left: 4px solid #0b51b7;
    background: #f5f8ff;
    margin: 1rem 0;
    padding: 0.8rem 1rem;
    color: #1f2937;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.45;
}

.message-line {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    align-items: center;
    margin: 0.75rem 0 1.35rem;
    border: 1px solid #d8dee9;
    border-radius: 7px;
    background: #f8fafc;
    padding: 0.65rem 0.75rem;
}

.message-token,
.token-chip {
    border: 1px solid #d8dee9;
    border-radius: 6px;
    background: #ffffff;
    color: #111827;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-weight: 600;
    line-height: 1.25;
}

.message-token {
    padding: 0.35rem 0.55rem;
    font-size: 0.9rem;
}

.token-chip {
    display: inline-block;
    padding: 0.08rem 0.32rem;
    font-size: 0.92em;
    vertical-align: baseline;
}

.message-token--context,
.token-chip--context {
    border-color: #cbd5e1;
    background: #ffffff;
    color: #334155;
}

.message-token--accept,
.token-chip--accept {
    border-color: #6bbf9a;
    background: #ecfdf5;
    color: #0b7d33;
}

.message-token--reject,
.token-chip--reject {
    border-color: #f29b9b;
    background: #fff5f5;
    color: #dc2626;
}

.message-token--target,
.token-chip--target {
    border-color: #8fb5ff;
    background: #f3f7ff;
    color: #0b51b7;
}

.message-token--muted,
.token-chip--muted {
    border-color: #d1d5db;
    background: #f3f4f6;
    color: #6b7280;
}

.baseline-trace {
    display: grid;
    gap: 0.25rem;
    margin: 0.9rem 0 1.15rem;
    border: 1px solid #d8dee9;
    border-radius: 7px;
    background: #f8fafc;
    padding: 0.65rem 0.75rem;
}

.baseline-trace__line {
    display: grid;
    grid-template-columns: 4.6rem 16rem 1.2rem 4.3rem 1.2rem 2.4rem;
    gap: 0.25rem;
    align-items: center;
    justify-content: start;
}

.baseline-trace span {
    min-width: 0;
    color: #111827;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.25;
}

.baseline-trace__step {
    color: #475569 !important;
    text-align: right;
}

.baseline-trace__context {
    color: #334155 !important;
    text-align: left;
}

.baseline-trace__arrow {
    color: #64748b !important;
    text-align: center;
}

.baseline-trace__model,
.baseline-trace__token {
    color: #0b51b7 !important;
    text-align: center;
}

.dflash-block {
    display: grid;
    gap: 0.25rem;
    margin: 0.9rem 0 1.15rem;
    border: 1px solid #d8dee9;
    border-radius: 7px;
    background: #f8fafc;
    padding: 0.65rem 0.75rem;
}

.dflash-block__row {
    display: grid;
    grid-template-columns: 5.6rem repeat(7, minmax(3.6rem, 1fr));
    gap: 0.28rem;
    align-items: center;
}

.dflash-block span {
    min-width: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.86rem;
    font-weight: 600;
    line-height: 1.25;
    text-align: center;
}

.dflash-block__label {
    color: #475569 !important;
    text-align: right !important;
}

.dflash-block__cell {
    border: 1px solid #d8dee9;
    border-radius: 6px;
    background: #ffffff;
    padding: 0.3rem 0.25rem;
    color: #111827;
}

.dflash-block__cell--target {
    border-color: #8fb5ff;
    background: #f3f7ff;
    color: #0b51b7;
}

.dflash-block__cell--draft {
    border-color: #6bbf9a;
    background: #ecfdf5;
    color: #0b7d33;
}

.dflash-block__cell--muted {
    border-color: #d1d5db;
    background: #f3f4f6;
    color: #6b7280;
}

.step-trace {
    display: grid;
    gap: 0.25rem;
    margin: 0.9rem 0 1.15rem;
    border: 1px solid #d8dee9;
    border-radius: 7px;
    background: #f8fafc;
    padding: 0.65rem 0.75rem;
}

.step-trace__line {
    display: grid;
    grid-template-columns: max-content 1.25rem max-content;
    gap: 0.55rem;
    align-items: center;
    justify-content: start;
}

.step-trace__grid-line {
    display: grid;
    grid-template-columns: 8.8rem repeat(4, minmax(3.7rem, 1fr));
    gap: 0.35rem;
    align-items: center;
}

.step-trace__five-line {
    display: grid;
    grid-template-columns: 11.5rem minmax(3.2rem, 1fr) minmax(3.2rem, 1fr) minmax(8.5rem, 1.45fr) minmax(4rem, 1fr) minmax(4rem, 1fr);
    gap: 0.35rem;
    align-items: center;
}

.step-trace span {
    min-width: 0;
    color: #111827;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.25;
    text-align: center;
    white-space: nowrap;
}

.step-trace__actor,
.step-trace__label {
    color: #475569 !important;
    font-weight: 700 !important;
}

.step-trace__label {
    text-align: right !important;
}

.step-trace__token--with-arrow {
    position: relative;
}

.step-trace__inline-arrow {
    position: absolute;
    left: calc(50% - 3.2rem);
    width: 0.8rem;
    color: #64748b !important;
    text-align: center !important;
}

.step-trace__arrow {
    color: #64748b !important;
}

.step-trace__accept {
    color: #0b7d33 !important;
}

.step-trace__reject {
    color: #dc2626 !important;
}

.step-trace__target {
    color: #0b51b7 !important;
}

.step-trace__muted {
    color: #64748b !important;
}

.process-diagram {
    --process-state-bg: #f8fafc;
    --process-state-border: #cbd5e1;
    --process-state-text: #1f2937;
    --process-draft-bg: #ecfdf5;
    --process-draft-border: #6bbf9a;
    --process-token-bg: #ecfdf5;
    --process-token-border: #6bbf9a;
    --process-verify-bg: #f3f7ff;
    --process-verify-border: #8fb5ff;
    --process-group-bg: #f9fafb;
    --process-arrow: #64748b;
    max-width: 680px;
    margin: 1rem auto 1.25rem;
}

.process-diagram--native-mtp {
    max-width: 760px;
}

.process-diagram__lane {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
}

.process-diagram__row {
    display: flex;
    width: 100%;
    align-items: stretch;
    justify-content: center;
    gap: 0.5rem;
}

.process-diagram__group {
    width: 100%;
    border: 1px solid #d8dee9;
    border-radius: 8px;
    background: var(--process-group-bg);
    padding: 0.6rem 0.7rem 0.7rem;
}

.process-diagram__group-title {
    margin: 0 0 0.45rem;
    color: #4b5563;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}

.process-diagram__step {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--process-state-border);
    border-radius: 7px;
    background: var(--process-state-bg);
    padding: 0.45rem 0.65rem;
    color: var(--process-state-text);
    font-size: 0.82rem;
    font-weight: 600;
    text-align: center;
}

.process-diagram__step--draft {
    border-color: var(--process-draft-border);
    background: var(--process-draft-bg);
}

.process-diagram__step--token {
    border-color: var(--process-token-border);
    background: var(--process-token-bg);
}

.process-diagram__step--verify {
    border-color: var(--process-verify-border);
    background: var(--process-verify-bg);
}

.process-diagram__step--small {
    flex: 1 1 0;
    min-width: 0;
}

.process-diagram__arrow {
    display: flex;
    width: 100%;
    height: 1rem;
    align-items: center;
    justify-content: center;
    color: var(--process-arrow);
    font-size: 1rem;
    line-height: 1;
    text-align: center;
}

.process-diagram__operator {
    display: flex;
    min-width: 1.5rem;
    align-items: center;
    justify-content: center;
    color: var(--process-arrow);
    font-weight: 700;
}

.process-diagram__tokens {
    display: flex;
    width: 100%;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.35rem;
}

.process-diagram__token {
    border: 1px solid var(--process-token-border);
    border-radius: 999px;
    background: var(--process-token-bg);
    padding: 0.25rem 0.55rem;
    color: var(--process-state-text);
    font-size: 0.78rem;
    font-weight: 700;
}

.process-diagram__token--anchor {
    border-color: var(--process-verify-border);
    background: var(--process-verify-bg);
}

.process-diagram__token--target {
    border-color: var(--process-verify-border);
    background: var(--process-verify-bg);
    color: #0b51b7;
}

.process-diagram__token--mask {
    border-color: var(--process-state-border);
    background: var(--process-state-bg);
    color: #64748b;
}

.drafting-style {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.45rem;
    align-items: center;
    margin: 0.9rem 0 1.15rem;
    border: 1px solid #d8dee9;
    border-radius: 7px;
    background: #f8fafc;
    padding: 0.65rem 0.75rem;
}

.drafting-style--single + .drafting-style--single {
    margin-top: -0.65rem;
}

.drafting-style__row {
    display: grid;
    grid-template-columns: 12rem 1fr;
    gap: 0.75rem;
    align-items: center;
}

.drafting-style__label {
    color: #475569;
    font-weight: 700;
    text-align: right;
}

.drafting-style__tokens {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    align-items: center;
}

.drafting-style__arrow {
    color: #64748b;
    font-weight: 700;
}

.drafting-style__note {
    margin-left: 0.25rem;
    color: #64748b;
    font-size: 0.86rem;
    font-weight: 700;
}

.plotly-chart-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.3rem 0 0.35rem;
}

.plotly-chart-heading {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 0.15rem;
}

.plotly-chart-heading strong {
    color: #1f2937;
    font-size: 1.05rem;
    line-height: 1.25;
}

.plotly-chart-heading span,
.plotly-chart-select span {
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.3;
}

.plotly-chart-controls {
    display: flex;
    align-items: flex-end;
    flex: 0 0 auto;
    gap: 0.6rem;
}

.plotly-chart-select {
    display: flex;
    min-width: min(19rem, 100%);
    flex: 0 0 auto;
    flex-direction: column;
    gap: 0.2rem;
}

.plotly-chart-select select {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid #b8c7dc;
    border-radius: 6px;
    background: #ffffff;
    color: #1f2937;
    font: inherit;
    font-size: 0.85rem;
    line-height: 1.3;
    padding: 0.42rem 2rem 0.42rem 0.55rem;
}

.plotly-chart-select--metric {
    min-width: 11rem;
}

.appendix-acceptance-view {
    margin: 1rem 0 1.25rem;
}

.appendix-acceptance-controls {
    display: flex;
    align-items: flex-end;
    gap: 0.6rem;
    margin-bottom: 0.9rem;
}

.appendix-acceptance-panel {
    display: none;
}

.appendix-acceptance-panel.is-active {
    display: block;
}

.appendix-acceptance-panel h3 {
    margin-top: 0.75rem;
}

.acceptance-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.8rem;
}

.acceptance-card {
    min-width: 0;
    border: 1px solid #d8dee9;
    border-radius: 8px;
    background: #ffffff;
    padding: 0.55rem;
}

.acceptance-card--single {
    padding: 0.6rem;
}

.acceptance-card h4 {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
    margin: 0 0 0.45rem;
    color: #334155;
    font-size: 0.82rem;
    line-height: 1.25;
}

.acceptance-card h4 span {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 500;
}

.acceptance-table {
    overflow-x: auto;
}

.acceptance-table table {
    width: 100%;
    min-width: 0;
    margin: 0;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: fixed;
    font-size: 0.78rem;
}

.acceptance-table th,
.acceptance-table td {
    padding: 0.26rem 0.2rem;
    text-align: center;
    vertical-align: middle;
}

.acceptance-table th:first-child,
.acceptance-table td:first-child {
    position: sticky;
    left: 0;
    z-index: 1;
    width: 10rem;
    background: #ffffff;
    text-align: left;
}

.acceptance-table td:first-child strong,
.acceptance-table td:first-child small {
    display: block;
    line-height: 1.2;
}

.acceptance-table td:first-child small {
    color: #64748b;
    font-size: 0.68rem;
    white-space: normal;
}

.acceptance-table td:first-child small span {
    display: block;
}

.acceptance-cell {
    border: 1px solid rgba(22, 101, 52, calc(0.08 + var(--accept) * 0.38));
    background: rgba(34, 197, 94, calc(0.04 + var(--accept) * 0.36));
    color: #102a1c;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

.appendix-raw-data {
    margin: 1rem 0;
}

.appendix-metric-note {
    margin: -0.35rem 0 1rem;
    color: #64748b;
    font-size: 0.82rem;
}

.appendix-raw-data summary {
    cursor: pointer;
    color: #1f2937;
    font-weight: 700;
}

@media (max-width: 640px) {
    .process-diagram {
        max-width: 100%;
    }

    .process-diagram__row {
        flex-direction: column;
    }

    .process-diagram__operator {
        min-height: 0.75rem;
    }

    .process-diagram__step {
        font-size: 0.78rem;
    }

    .plotly-chart-header {
        align-items: stretch;
        flex-direction: column;
    }

    .plotly-chart-controls {
        align-items: stretch;
        flex-direction: column;
    }

    .plotly-chart-select {
        min-width: 0;
    }

    .step-trace {
        padding: 0.55rem 0.45rem;
    }

    .step-trace__line {
        grid-template-columns: max-content 1rem max-content;
        gap: 0.25rem;
    }

    .step-trace__grid-line {
        grid-template-columns: 6.6rem repeat(4, minmax(2.5rem, 1fr));
        gap: 0.18rem;
    }

    .step-trace__five-line {
        grid-template-columns: 5.5rem minmax(1.8rem, 1fr) minmax(1.8rem, 1fr) minmax(4.7rem, 1.55fr) minmax(2.1rem, 1fr) minmax(2.1rem, 1fr);
        gap: 0.12rem;
    }

    .step-trace span {
        font-size: 0.68rem;
    }

    .baseline-trace {
        padding: 0.55rem 0.45rem;
    }

    .baseline-trace__line {
        grid-template-columns: 3.5rem minmax(7.4rem, 1fr) 0.8rem 3.1rem 0.8rem 1.8rem;
        gap: 0.16rem;
    }

    .baseline-trace span {
        font-size: 0.66rem;
    }

    .dflash-block {
        padding: 0.55rem 0.45rem;
    }

    .dflash-block__row {
        grid-template-columns: 3.7rem repeat(7, minmax(2.1rem, 1fr));
        gap: 0.12rem;
    }

    .dflash-block span {
        font-size: 0.58rem;
    }

    .dflash-block__cell {
        padding: 0.22rem 0.08rem;
    }

    .drafting-style {
        grid-template-columns: 1fr;
        gap: 0.3rem;
        padding: 0.55rem 0.45rem;
    }

    .drafting-style__label {
        font-size: 0.72rem;
        text-align: right;
    }

    .drafting-style__tokens {
        gap: 0.18rem;
    }

    .drafting-style .message-token {
        padding: 0.22rem 0.28rem;
        font-size: 0.66rem;
    }

    .drafting-style__note {
        margin-left: 0;
        font-size: 0.68rem;
    }

    .appendix-acceptance-controls {
        align-items: stretch;
        flex-direction: column;
    }

    .acceptance-grid {
        grid-template-columns: 1fr;
    }

    .acceptance-table table {
        font-size: 0.68rem;
    }

    .acceptance-table th,
    .acceptance-table td {
        padding: 0.2rem 0.1rem;
    }

    .acceptance-table th:first-child,
    .acceptance-table td:first-child {
        width: 7.4rem;
    }

    .acceptance-table td:first-child small {
        font-size: 0.62rem;
    }
}

</style>

**TL;DR:** Speculative decoding allows vLLM to verify multiple drafted tokens in a single target-model pass. In our experiments, its effect on output-token throughput varied across drafting methods and proposal lengths, and also depended on the model family, draft checkpoint, workload, and acceptance behavior.

---

## Introduction

Large language models support a wide range of applications, but serving them at scale requires careful optimization. Standard autoregressive decoding is the baseline used by most LLM serving systems: the model generates one token, appends it to the sequence, and then uses the updated sequence to generate the next token. This process is simple and reliable, but the serving loop still advances one committed token at a time because output tokens must be produced in strict left-to-right order.

Speculative decoding [[1]](#ref-1) builds on this baseline through a draft-and-verify mechanism. A lightweight draft component proposes candidate future tokens, and the target model verifies those candidates before they are committed. When several draft tokens are accepted, the system can commit multiple output tokens from a single target-model verification step while preserving the target model's output behavior.

This post explores how speculative decoding works in vLLM and shares measurements from our test environment. We first review the autoregressive decoding baseline and the draft-and-verify process. We then examine five speculative-drafting approaches: native MTP, Gemma 4 MTP, EAGLE-3, DFlash, and DSpark. These methods differ in how the draft component receives information from the target model and whether candidate tokens are generated sequentially, autoregressively, in parallel, or through a hybrid approach. Finally, we show how to enable the methods tested in our environment, report measurements from our experiments on AMD Instinct™ MI300X and MI355X GPUs using the ROCm™ open software platform, and discuss practical tuning and observability considerations.

---

## The autoregressive decoding baseline

In standard autoregressive decoding, each decode step produces and commits one new token. For example, generating four output tokens requires four sequential decode steps:

<div class="baseline-trace" role="img" aria-label="Standard autoregressive decoding appends each generated token to the next step's context">
  <div class="baseline-trace__line">
    <span class="baseline-trace__step">Step 1:</span>
    <span class="baseline-trace__context">context</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__model">model</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__token">T1</span>
  </div>
  <div class="baseline-trace__line">
    <span class="baseline-trace__step">Step 2:</span>
    <span class="baseline-trace__context">context + T1</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__model">model</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__token">T2</span>
  </div>
  <div class="baseline-trace__line">
    <span class="baseline-trace__step">Step 3:</span>
    <span class="baseline-trace__context">context + T1 T2</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__model">model</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__token">T3</span>
  </div>
  <div class="baseline-trace__line">
    <span class="baseline-trace__step">Step 4:</span>
    <span class="baseline-trace__context">context + T1 T2 T3</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__model">model</span>
    <span class="baseline-trace__arrow">→</span>
    <span class="baseline-trace__token">T4</span>
  </div>
</div>

After each step, the generated token is appended to the sequence and becomes part of the input for the next step. This makes the decoding loop straightforward, but it also requires one model decode step for every output token. During long generations, this token-by-token loop can dominate latency and limit serving throughput.

The key question behind speculative decoding is therefore:

<p class="key-question">Can we preserve the output behavior of the original model while reducing how often generation advances by only one token at a time?</p>

Speculative decoding addresses this by separating proposal from verification. A draft component first proposes several candidate future tokens. The original model, acting as the target model, then verifies those candidates before they are committed.

---

## Core idea of speculative decoding

Speculative decoding does not replace the original model. Instead, it keeps the original model as the target model, which remains responsible for the final output, and adds a faster proposal stage in front of it.

The process has two parts:

- Draft: propose several candidate future tokens.
- Verify: use the target model to check those candidates.

During each speculative decoding round, as illustrated in Figure 1, a lightweight draft component proposes one or more future tokens. These tokens are only candidates and are not committed immediately. The target model then evaluates the candidate token sequence in one verification pass.

Verification proceeds from left to right. Each draft token is checked using the target model's result at the corresponding position. Accepted tokens are committed to the output sequence. When a draft token is rejected, later candidates from the same proposal are no longer accepted.

If a draft token is rejected, the target model provides the next token. The remaining draft tokens are discarded, and generation continues from the updated sequence.

Conceptually, standard autoregressive decoding advances like this:

<div class="step-trace step-trace--wide" role="img" aria-label="Standard decoding advances one target model step at a time">
  <div class="step-trace__grid-line">
    <span class="step-trace__label">target model</span>
    <span class="step-trace__token step-trace__token--with-arrow"><span class="step-trace__inline-arrow">→</span>T1</span>
    <span></span>
    <span></span>
    <span></span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">target model</span>
    <span class="step-trace__token step-trace__token--with-arrow"><span class="step-trace__inline-arrow">→</span>T2</span>
    <span></span>
    <span></span>
    <span></span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">target model</span>
    <span class="step-trace__token step-trace__token--with-arrow"><span class="step-trace__inline-arrow">→</span>T3</span>
    <span></span>
    <span></span>
    <span></span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">target model</span>
    <span class="step-trace__token step-trace__token--with-arrow"><span class="step-trace__inline-arrow">→</span>T4</span>
    <span></span>
    <span></span>
    <span></span>
  </div>
</div>

Speculative decoding instead allows several candidate positions to be evaluated together:

<div class="step-trace step-trace--wide" role="img" aria-label="Speculative decoding accepts draft tokens until the first rejection, commits a target token, and discards the rest">
  <div class="step-trace__grid-line">
    <span class="step-trace__label">draft proposes</span>
    <span>T1</span>
    <span>T2</span>
    <span>T3</span>
    <span class="step-trace__muted">T4</span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">model verifies</span>
    <span class="step-trace__accept">✓</span>
    <span class="step-trace__accept">✓</span>
    <span class="step-trace__reject">✗</span>
    <span class="step-trace__muted">stop</span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">commit</span>
    <span>T1</span>
    <span>T2</span>
    <span class="step-trace__target">replacement token</span>
    <span class="step-trace__muted">-</span>
  </div>
</div>

This can reduce the number of target-model decoding rounds when multiple candidates are accepted. When the draft component produces tokens that the target model accepts, several output tokens can be committed from one target-model verification step. When a proposal is rejected, the target-side result determines how generation continues.

<p align="center">
<picture>
<img src="/assets/figures/2026-08-20-speculative-decoding-amd-gpus/figure-01.svg" width="95%">
</picture>
</p>

<p align="center"><em>Figure 1. Speculative decoding flow: a draft component proposes candidate future tokens, and the target model verifies them before output tokens are committed.</em></p>

### A simple accept/reject example

Figure 2 gives an example of one speculative decoding round. Green boxes are draft tokens that survive verification, the red box marks the first rejected draft token, and the gray box is a later draft token that is discarded. The blue token in the output comes from the target model, not from the draft proposal.

<p align="center">
<picture>
<img src="/assets/figures/2026-08-20-speculative-decoding-amd-gpus/figure-02.svg" width="95%">
</picture>
</p>

<p align="center"><em>Figure 2. Left-to-right verification of a draft proposal. The first two draft tokens are accepted, the rejected position uses a target-model token, and the remaining candidate is discarded.</em></p>

Suppose the current prompt is:

<div class="message-line" aria-label="Current prompt">
  <span class="message-token message-token--context">The weather today is</span>
</div>

The draft component proposes several future tokens:

<div class="message-line" aria-label="Draft proposal">
  <span class="message-token message-token--accept">sunny</span>
  <span class="message-token message-token--accept">and</span>
  <span class="message-token message-token--reject">warm</span>
  <span class="message-token message-token--muted">outside</span>
</div>

The target model verifies the draft tokens from left to right:

<div class="step-trace step-trace--wide" role="img" aria-label="The weather example accepts sunny and and, rejects warm, commits clear, and discards outside">
  <div class="step-trace__grid-line">
    <span class="step-trace__label">draft proposes</span>
    <span class="step-trace__accept">sunny</span>
    <span class="step-trace__accept">and</span>
    <span class="step-trace__reject">warm</span>
    <span class="step-trace__muted">outside</span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">model verifies</span>
    <span class="step-trace__accept">✓</span>
    <span class="step-trace__accept">✓</span>
    <span class="step-trace__reject">✗</span>
    <span class="step-trace__muted">stop</span>
  </div>
  <div class="step-trace__grid-line">
    <span class="step-trace__label">commit</span>
    <span class="step-trace__accept">sunny</span>
    <span class="step-trace__accept">and</span>
    <span class="step-trace__target">clear</span>
    <span class="step-trace__muted">-</span>
  </div>
</div>

The first two draft tokens, <span class="token-chip token-chip--accept">sunny</span> and <span class="token-chip token-chip--accept">and</span>, are accepted. At the third position, the draft proposes <span class="token-chip token-chip--reject">warm</span>, but the target model selects <span class="token-chip token-chip--target">clear</span>. The remaining candidate, <span class="token-chip token-chip--muted">outside</span>, is discarded because it follows the first rejected position.

The next decoding round therefore continues from:

<div class="message-line" aria-label="Next decoding round starts from the committed output">
  <span class="message-token message-token--context">The weather today is</span>
  <span class="message-token message-token--accept">sunny</span>
  <span class="message-token message-token--accept">and</span>
  <span class="message-token message-token--target">clear</span>
</div>

---

## How the drafting methods work

Although all speculative decoding methods follow the same overall draft-and-verify process, they differ in how the draft component is designed and how it works with the target model.

The main differences are:

- The type of information received from the target model.
- How this information is incorporated into the drafting process.
- Whether candidate tokens are generated sequentially or in parallel.

Based on these differences, the drafting methods discussed in this post can be grouped into three broad categories: native MTP modules, separate MTP drafters, and dedicated target-conditioned draft networks.

- **Native MTP modules:** built directly into the target-model architecture; use a model-native auxiliary prediction path; generate candidate tokens sequentially.
- **Separate MTP drafters:** use a separate checkpoint paired with a specific target model; use target-model activations and shared KV-cache information during inference; generate candidate tokens sequentially.
- **Dedicated target-conditioned draft networks:** use separate speculator models trained for a specific target model, including EAGLE-3, DFlash, and DSpark. EAGLE-3 drafts autoregressively from target-model hidden states, DFlash drafts parallel blocks from target-model hidden states, and DSpark adds lightweight causal correction and confidence-based prefix selection.

These categories describe the draft component architecture, not the target-model family. A target model may support native MTP while also having separately trained EAGLE-3, DFlash, or DSpark draft models.

The draft component does not operate entirely on its own. Depending on the method, the draft component may receive:

- A hidden representation from the target model.
- Hidden states from several selected target layers.
- The target model's KV cache.
- Features produced by combining multiple target-model representations.

The following sections explain how each method uses this information and how it generates candidate tokens.

### Native MTP

Multi-Token Prediction, or MTP, refers to a family of model-native mechanisms for predicting tokens beyond the immediate next token. In vLLM, native MTP is available when the target model includes a compatible auxiliary prediction component [[2]](#ref-2). The exact MTP architecture varies across model families, but each implementation provides an auxiliary path for proposing future tokens.

At the first speculative step, the MTP component combines a hidden representation from the target model with information from the current token to predict the first draft token. At subsequent steps, the newly drafted token and the hidden state produced by the previous MTP step are used to predict the next candidate. After the configured number of candidates has been proposed, the target model evaluates them together in one verification pass.

<div class="process-diagram process-diagram--native-mtp" role="img" aria-label="Native MTP drafts tokens sequentially before one target-model verification pass">
  <div class="process-diagram__lane">
    <div class="process-diagram__group">
      <p class="process-diagram__group-title">First draft token</p>
      <div class="process-diagram__step">Target-model hidden representation</div>
      <div class="process-diagram__arrow">↓</div>
      <div class="process-diagram__step process-diagram__step--draft">MTP component</div>
      <div class="process-diagram__arrow">↓</div>
      <div class="process-diagram__step process-diagram__step--token">Draft token 1 + updated hidden state</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__group">
      <p class="process-diagram__group-title">Subsequent draft tokens</p>
      <div class="process-diagram__step">Previous MTP hidden state + latest draft token</div>
      <div class="process-diagram__arrow">↓</div>
      <div class="process-diagram__step process-diagram__step--draft">MTP component</div>
      <div class="process-diagram__arrow">↓</div>
      <div class="process-diagram__step process-diagram__step--token">Next draft token + updated hidden state</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Configured draft sequence complete</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--verify">Target model evaluates all proposed tokens together in one verification pass</div>
  </div>
</div>

Many native MTP implementations follow a similar pattern. A hidden representation from the target model or from the previous MTP prediction is combined with the embedding of a shifted input token or the latest drafted token:

<div class="process-diagram" role="img" aria-label="Native MTP fusion path">
  <div class="process-diagram__lane">
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small">Target-model or previous MTP hidden representation</div>
      <div class="process-diagram__operator">+</div>
      <div class="process-diagram__step process-diagram__step--small">Shifted input-token or latest draft-token embedding</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Model-specific fusion or projection</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--draft">Auxiliary prediction layer</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--token">Draft-token logits</div>
  </div>
</div>

The two inputs serve different purposes: (1) the hidden representation carries information about the preceding sequence; and (2) the token embedding identifies the latest token from which drafting continues. In common implementations, they are combined along the hidden dimension and transformed before entering the auxiliary prediction layer.

The number of physical MTP layers and the configured speculative length are separate concepts. When `num_speculative_tokens` exceeds the prediction depth directly provided by the checkpoint, vLLM can reuse the MTP path through additional forward passes. A larger value therefore proposes more candidates before verification, but also introduces more sequential drafting work.

<div class="process-diagram" role="img" aria-label="Native MTP repeated drafting path">
  <div class="process-diagram__lane">
    <div class="process-diagram__step process-diagram__step--draft">Model-specific MTP path</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__tokens">
      <span class="process-diagram__token">Draft token 1</span>
      <span class="process-diagram__token">Draft token 2</span>
      <span class="process-diagram__token">Draft token 3</span>
      <span class="process-diagram__token">...</span>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Configured proposal complete</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--verify">Target model verifies the proposed sequence</div>
  </div>
</div>

Native MTP is closely tied to the target-model architecture. In many implementations, parts of the MTP path share components with the target model, which can keep the additional memory overhead relatively modest. However, generating multiple speculative tokens still requires sequential drafting before verification.

### Gemma 4 MTP

Gemma 4 uses a separately packaged MTP draft component paired with a specific target model [[3]](#ref-3). Although the draft component has its own checkpoint, it remains closely connected to the target model during inference.

<div class="process-diagram" role="img" aria-label="Gemma 4 MTP draft component inputs">
  <div class="process-diagram__lane">
    <div class="process-diagram__step process-diagram__step--verify">Gemma 4 target model</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small">Target-model activations</div>
      <div class="process-diagram__step process-diagram__step--small">Shared target KV cache</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--draft">Gemma 4 MTP draft component</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--token">Candidate tokens</div>
  </div>
</div>

The draft component uses activations produced by the target model and shares the target model's KV cache. This allows it to reuse contextual information that the target has already computed instead of processing the accepted prefix independently.

As with native MTP, the number of layers in the draft component is separate from the configured speculative length. When several candidate tokens are requested, the draft component generates them sequentially:

<div class="process-diagram" role="img" aria-label="Gemma 4 MTP sequential drafting path">
  <div class="process-diagram__lane">
    <div class="process-diagram__step process-diagram__step--draft">Gemma 4 MTP draft component</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__tokens">
      <span class="process-diagram__token">Draft token 1</span>
      <span class="process-diagram__token">Draft token 2</span>
      <span class="process-diagram__token">Draft token 3</span>
      <span class="process-diagram__token">...</span>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Configured proposal complete</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--verify">Target model verifies the proposed sequence</div>
  </div>
</div>

### EAGLE-3

EAGLE-3 uses a dedicated draft network trained for a specific target model. The draft component has its own execution path, but it remains closely conditioned on information produced by the target model [[4]](#ref-4).

During the target-model forward pass, EAGLE-3 records hidden states from three stages of the target Transformer: near the beginning, around the middle, and near the end. These are contextual representations of the same accepted sequence at different stages of target-model processing.

<div class="process-diagram" role="img" aria-label="EAGLE-3 target hidden state fusion">
  <div class="process-diagram__lane">
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small">Early-layer hidden state</div>
      <div class="process-diagram__step process-diagram__step--small">Middle-layer hidden state</div>
      <div class="process-diagram__step process-diagram__step--small">Late-layer hidden state</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Concatenate + projection</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Fused target feature</div>
  </div>
</div>

The three hidden states are concatenated and projected into a single fused target feature. This fused representation is then combined with the embedding of the sampled token before entering the EAGLE-3 draft decoder.

<div class="process-diagram" role="img" aria-label="EAGLE-3 draft decoder inputs">
  <div class="process-diagram__lane">
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small">Fused target feature</div>
      <div class="process-diagram__operator">+</div>
      <div class="process-diagram__step process-diagram__step--small">Sampled-token embedding</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Concatenate + projection</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--draft">EAGLE-3 draft decoder</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--token">Draft token</div>
  </div>
</div>

The two inputs serve different purposes:

- The fused target feature summarizes the accepted sequence using information from several stages of the target-model forward pass.
- The sampled-token embedding identifies the token from which drafting continues.

EAGLE-3 generates draft tokens autoregressively. For the first draft token, it uses the fused target feature computed from the accepted sequence together with the sampled-token embedding. After a draft token is produced, its embedding is fed into the next drafting stage.

Because the target model has not yet processed the later speculative positions, target-model hidden states for those positions are not available. EAGLE-3 therefore uses the previous draft-component output when continuing the draft sequence.

<div class="process-diagram" role="img" aria-label="EAGLE-3 autoregressive draft-token generation">
  <div class="process-diagram__lane">
    <div class="process-diagram__group">
      <p class="process-diagram__group-title">First draft token</p>
      <div class="process-diagram__row">
        <div class="process-diagram__step process-diagram__step--small">Fused target feature</div>
        <div class="process-diagram__operator">+</div>
        <div class="process-diagram__step process-diagram__step--small">Sampled-token embedding</div>
      </div>
      <div class="process-diagram__arrow">↓</div>
      <div class="process-diagram__step process-diagram__step--token">Draft token 1</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__group">
      <p class="process-diagram__group-title">Subsequent draft tokens</p>
      <div class="process-diagram__row">
        <div class="process-diagram__step process-diagram__step--small">Previous draft-component output</div>
        <div class="process-diagram__operator">+</div>
        <div class="process-diagram__step process-diagram__step--small">Newly sampled-token embedding</div>
      </div>
      <div class="process-diagram__arrow">↓</div>
      <div class="process-diagram__tokens">
        <span class="process-diagram__token">Draft token 2</span>
        <span class="process-diagram__token">Draft token 3</span>
        <span class="process-diagram__token">...</span>
      </div>
    </div>
  </div>
</div>

This sequential feedback gives later draft tokens direct dependence on earlier drafted tokens along the proposed sequence. However, generating more speculative tokens also requires more sequential drafting work before verification.

### DFlash

DFlash uses a dedicated draft network trained for a specific target model. Unlike MTP and EAGLE-3, which generate candidate tokens sequentially, DFlash predicts a whole block of future positions in parallel [[5]](#ref-5).

DFlash begins each draft block with an anchor token. The anchor is a known token produced or confirmed by the target model, so DFlash does not need to predict it. Instead, it provides a known starting point for the masked positions that follow. In later decoding rounds, this is typically the additional target token returned by the previous verification pass.

The anchor occupies the first position of the block, while the remaining positions are masked and predicted in parallel:

A draft block starts with a confirmed anchor token, followed by masked positions:

<div class="dflash-block" role="img" aria-label="DFlash input block starts with one anchor token followed by masked positions">
  <div class="dflash-block__row">
    <span class="dflash-block__label">Position</span>
    <span class="dflash-block__cell">0</span>
    <span class="dflash-block__cell">1</span>
    <span class="dflash-block__cell">2</span>
    <span class="dflash-block__cell">3</span>
    <span class="dflash-block__cell">4</span>
    <span class="dflash-block__cell">5</span>
    <span class="dflash-block__cell">6</span>
  </div>
  <div class="dflash-block__row">
    <span class="dflash-block__label">Input</span>
    <span class="dflash-block__cell dflash-block__cell--target">anchor</span>
    <span class="dflash-block__cell dflash-block__cell--muted">mask</span>
    <span class="dflash-block__cell dflash-block__cell--muted">mask</span>
    <span class="dflash-block__cell dflash-block__cell--muted">mask</span>
    <span class="dflash-block__cell dflash-block__cell--muted">mask</span>
    <span class="dflash-block__cell dflash-block__cell--muted">mask</span>
    <span class="dflash-block__cell dflash-block__cell--muted">mask</span>
  </div>
</div>

Here, `anchor` is the known target-model token, while the masked positions are predicted by DFlash.

A single DFlash forward pass predicts all masked positions together:

<div class="dflash-block" role="img" aria-label="DFlash output block keeps the anchor and predicts all draft positions together">
  <div class="dflash-block__row">
    <span class="dflash-block__label">Position</span>
    <span class="dflash-block__cell">0</span>
    <span class="dflash-block__cell">1</span>
    <span class="dflash-block__cell">2</span>
    <span class="dflash-block__cell">3</span>
    <span class="dflash-block__cell">4</span>
    <span class="dflash-block__cell">5</span>
    <span class="dflash-block__cell">6</span>
  </div>
  <div class="dflash-block__row">
    <span class="dflash-block__label">Output</span>
    <span class="dflash-block__cell dflash-block__cell--target">anchor</span>
    <span class="dflash-block__cell dflash-block__cell--draft">draft1</span>
    <span class="dflash-block__cell dflash-block__cell--draft">draft2</span>
    <span class="dflash-block__cell dflash-block__cell--draft">draft3</span>
    <span class="dflash-block__cell dflash-block__cell--draft">draft4</span>
    <span class="dflash-block__cell dflash-block__cell--draft">draft5</span>
    <span class="dflash-block__cell dflash-block__cell--draft">draft6</span>
  </div>
</div>

Like EAGLE-3, DFlash first combines hidden states from several target-model layers into a fused representation.

<div class="process-diagram" role="img" aria-label="DFlash target hidden state fusion">
  <div class="process-diagram__lane">
    <div class="process-diagram__step">Target hidden states from selected layers</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Concatenate + projection</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step">Fused target context</div>
  </div>
</div>

The main difference is how this fused representation is used. EAGLE-3 combines it with the sampled-token embedding at the input of its autoregressive draft network. DFlash instead converts the fused target context into additional Key and Value representations that are available in every layer of the draft network.

Queries from the masked draft positions can therefore attend to both:

- Key and Value representations derived from the target model.
- Key and Value representations produced from the draft block itself.

<div class="process-diagram process-diagram--dflash-kv" role="img" aria-label="DFlash makes target-derived key and value information available at every draft layer">
  <div class="process-diagram__lane">
    <div class="process-diagram__step process-diagram__step--verify">Fused target context</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__group">
      <p class="process-diagram__group-title">Available in every draft layer</p>
      <div class="process-diagram__tokens">
        <span class="process-diagram__token process-diagram__token--target">Target K/V + layer 1</span>
        <span class="process-diagram__token process-diagram__token--target">Target K/V + layer 2</span>
        <span class="process-diagram__token process-diagram__token--target">Target K/V + layer 3</span>
        <span class="process-diagram__token process-diagram__token--target">Target K/V + layer N</span>
      </div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--draft">Masked draft-position queries attend to both target-derived and draft-block K/V</div>
  </div>
</div>

The target-model context therefore remains available throughout the draft network, rather than being supplied only once at its input.

After the draft block has been generated, the target model evaluates all proposed tokens in one verification pass. The acceptance decision is then applied from left to right: accepted tokens are committed until the first rejection, and the remaining candidates are discarded.

<div class="step-trace step-trace--dflash-accept" role="img" aria-label="DFlash proposal accepts D1 and D2, rejects D3, and discards D4 and D5">
  <div class="step-trace__five-line">
    <span class="step-trace__label">draft proposal</span>
    <span class="step-trace__accept">D1</span>
    <span class="step-trace__accept">D2</span>
    <span class="step-trace__reject">D3</span>
    <span class="step-trace__muted">D4</span>
    <span class="step-trace__muted">D5</span>
  </div>
  <div class="step-trace__five-line">
    <span class="step-trace__label">acceptance result</span>
    <span class="step-trace__accept">accept</span>
    <span class="step-trace__accept">accept</span>
    <span class="step-trace__reject">reject</span>
    <span class="step-trace__muted">discard</span>
    <span class="step-trace__muted">discard</span>
  </div>
  <div class="step-trace__five-line">
    <span class="step-trace__label">committed output</span>
    <span class="step-trace__accept">D1</span>
    <span class="step-trace__accept">D2</span>
    <span class="step-trace__target">target-model token</span>
    <span class="step-trace__muted">-</span>
    <span class="step-trace__muted">-</span>
  </div>
</div>

Here, the target-model token replaces the first rejected draft token, while the remaining draft tokens are discarded.

A defining characteristic of DFlash is that all masked positions are predicted together in one draft-network forward pass. 

<div class="drafting-style drafting-style--single" role="img" aria-label="DFlash predicts draft tokens together">
  <span class="drafting-style__tokens">
    <span class="message-token message-token--accept">draft1</span>
    <span class="message-token message-token--accept">draft2</span>
    <span class="message-token message-token--accept">draft3</span>
    <span class="message-token message-token--accept">draft4</span>
    <span class="drafting-style__note">predicted together</span>
  </span>
</div>

This differs from sequential drafting:

<div class="drafting-style drafting-style--single" role="img" aria-label="Sequential drafting predicts each draft token after the previous one">
  <span class="drafting-style__tokens">
    <span class="message-token message-token--accept">draft1</span>
    <span class="drafting-style__arrow">→</span>
    <span class="message-token message-token--accept">draft2</span>
    <span class="drafting-style__arrow">→</span>
    <span class="message-token message-token--accept">draft3</span>
    <span class="drafting-style__arrow">→</span>
    <span class="message-token message-token--accept">draft4</span>
  </span>
</div>

Because all masked positions are predicted together, a later position is not conditioned on the sampled output of an earlier position during the same pass. This removes the token-by-token feedback used by autoregressive drafting. The effectiveness of later positions therefore depends on the trained checkpoint and workload, particularly when longer draft blocks are used.

### DSpark

DSpark extends parallel drafting with two additional mechanisms:

- A lightweight sequential head that introduces dependence between tokens within the draft block.
- Confidence-based selection of the prefix submitted for target-model verification.

DSpark uses a modified DFlash model as its parallel backbone [[6]](#ref-6). The backbone performs the main draft computation for all positions in one forward pass, producing a hidden state and a set of base logits for each draft position. It therefore inherits the target-context conditioning described in the DFlash section.

<div class="process-diagram" role="img" aria-label="DSpark parallel backbone outputs">
  <div class="process-diagram__lane">
    <div class="process-diagram__step">Target-derived context</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--draft">DSpark parallel backbone</div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small process-diagram__step--token">Hidden states for all draft positions</div>
      <div class="process-diagram__step process-diagram__step--small process-diagram__step--token">Base logits for all draft positions</div>
    </div>
  </div>
</div>

A fully parallel draft component predicts every position without first seeing the tokens selected at earlier positions in the same block. When several continuations are plausible, this can produce inconsistent combinations. For example, both "of course" and "no problem" may be reasonable continuations, but independent position-wise predictions could produce "of problem."

DSpark addresses this behavior by applying a lightweight sequential head after the parallel backbone. The backbone still computes the base logits for every position together. The sequential head then selects tokens from left to right, adjusting each position using information from the previously selected draft tokens.

DSpark applies a lightweight Markov head that introduces dependence between the selected draft tokens. For each position, the Markov head uses the immediately preceding selected token to produce a small bias. This bias adjusts the base logits produced by the parallel backbone:

<div class="process-diagram" role="img" aria-label="DSpark Markov head adjusts logits using previous draft token">
  <div class="process-diagram__lane">
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small process-diagram__step--token">Base logits for position k</div>
      <div class="process-diagram__operator">+</div>
      <div class="process-diagram__step process-diagram__step--small">Bias from draft token k-1</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__step process-diagram__step--token">Adjusted distribution for position k</div>
  </div>
</div>

The main draft network processes all candidate positions together in one forward pass. After that, only the lightweight Markov head runs from left to right to adjust each position using the previously selected draft token.

<div class="process-diagram" role="img" aria-label="DSpark combines parallel backbone with lightweight sequential correction">
  <div class="process-diagram__lane">
    <div class="process-diagram__row">
      <div class="process-diagram__step process-diagram__step--small process-diagram__step--draft">One parallel backbone pass</div>
      <div class="process-diagram__operator">+</div>
      <div class="process-diagram__step process-diagram__step--small process-diagram__step--draft">Lightweight sequential correction</div>
    </div>
    <div class="process-diagram__arrow">↓</div>
    <div class="process-diagram__tokens">
      <span class="process-diagram__token">Draft token 1</span>
      <span class="process-diagram__operator">→</span>
      <span class="process-diagram__token">token 2</span>
      <span class="process-diagram__operator">→</span>
      <span class="process-diagram__token">token 3</span>
      <span class="process-diagram__operator">→</span>
      <span class="process-diagram__token">...</span>
      <span class="process-diagram__operator">→</span>
      <span class="process-diagram__token">token N</span>
    </div>
  </div>
</div>

This allows later draft tokens to depend on tokens already selected within the same block without running the full draft network again for every position.

The DSpark design also includes a confidence head that can select a shorter draft prefix for target-model verification. This feature was not active in the vLLM path used for our experiments, so the benchmark results reflect only the parallel draft network and lightweight Markov correction.

The target model evaluates the proposed sequence in one verification pass, and draft tokens are committed from left to right until the first rejection.

### Summary of the drafting methods

Figure 3 gives a visual side-by-side view of the five drafting methods: what the draft component looks like, which target-model information it uses, and whether candidate tokens are generated sequentially or in parallel. The table below the figure restates the same comparison in a compact form. In all five methods, the target model still evaluates the proposed sequence in one verification pass, and the acceptance decision is applied from left to right until the first rejected draft token.

<p align="center">
<picture>
<img src="/assets/figures/2026-08-20-speculative-decoding-amd-gpus/figure-method-summary.svg" width="100%">
</picture>
</p>

<p align="center"><em>Figure 3. Draft structure and token generation patterns for the five speculative decoding methods discussed in this post.</em></p>

| Method | Draft component | Target-model information used | How draft tokens are generated |
| --- | --- | --- | --- |
| Native MTP | Model-native auxiliary MTP path | A target-model or previous MTP hidden representation combined with current draft-token information | Sequentially through repeated use of the MTP path |
| Gemma 4 MTP | Separate MTP draft component paired with the target model | Target-model activations and the shared target KV cache | Sequentially through the paired MTP component |
| EAGLE-3 | Dedicated autoregressive draft network | Hidden states captured near the beginning, around the middle, and near the end of the target-model forward pass, fused into one representation | Sequentially, with each drafted token influencing the next |
| DFlash | Dedicated parallel draft network | Fused target-model hidden states provided as additional Key and Value information in every draft layer | All candidate positions are predicted together in one parallel forward pass |
| DSpark | DFlash-style parallel draft network with a lightweight Markov head | The same target-conditioned information used by the parallel draft network | One parallel forward pass followed by lightweight sequential adjustment of token selection |

---

## How to enable speculative decoding in vLLM

In vLLM, speculative decoding is configured through `--speculative-config`. The main differences are the method name, whether a separate draft checkpoint is required, and the number of candidate tokens requested. Current vLLM supports mtp, eagle3, dflash, and dspark as method values.

<table class="config-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Separate draft checkpoint</th>
      <th>Typical configuration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Native MTP</td>
      <td>No</td>
      <td class="config-table__config">
        <code>"method": "mtp"</code><br>
        <code>"num_speculative_tokens": &lt;N&gt;</code>
      </td>
    </tr>
    <tr>
      <td>Gemma 4 MTP</td>
      <td>Yes</td>
      <td class="config-table__config">
        <code>"method": "mtp"</code><br>
        <code>"model": "&lt;matching-assistant&gt;"</code><br>
        <code>"num_speculative_tokens": &lt;N&gt;</code>
      </td>
    </tr>
    <tr>
      <td>EAGLE-3</td>
      <td>Yes</td>
      <td class="config-table__config">
        <code>"method": "eagle3"</code><br>
        <code>"model": "&lt;matching-speculator&gt;"</code><br>
        <code>"num_speculative_tokens": &lt;N&gt;</code>
      </td>
    </tr>
    <tr>
      <td>DFlash</td>
      <td>Yes</td>
      <td class="config-table__config">
        <code>"method": "dflash"</code><br>
        <code>"model": "&lt;matching-speculator&gt;"</code><br>
        <code>"num_speculative_tokens": &lt;N&gt;</code>
      </td>
    </tr>
    <tr>
      <td>DSpark</td>
      <td>Yes</td>
      <td class="config-table__config">
        <code>"method": "dspark"</code><br>
        <code>"model": "&lt;matching-speculator&gt;"</code><br>
        <code>"num_speculative_tokens": &lt;N&gt;</code>
      </td>
    </tr>
  </tbody>
</table>

For native MTP, the draft component is included with the target model, so the model field is omitted:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": <N>
  }'
```

For Gemma 4 MTP, EAGLE-3, DFlash, and DSpark, the model field normally points to a checkpoint trained for the target model:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "<method>",
    "model": "<matching-draft-checkpoint>",
    "num_speculative_tokens": <N>
  }'
```

Gemma 4 assistant checkpoints use the MTP path even though they are supplied through the model field. vLLM connects the assistant component to the target model and allows it to share the target KV cache.

Before enabling a method, check that:

- The installed vLLM version supports the method and model architecture.
- The draft checkpoint is compatible with the target model and method.
- `num_speculative_tokens` is compatible with the checkpoint.
- The model card supports the intended hardware and inference backend.

### Memory considerations

Native MTP does not load a separate draft checkpoint and may share components such as the embedding table or output head with the target model. Gemma 4 MTP, EAGLE-3, DFlash, and DSpark load additional draft weights, so sufficient GPU memory headroom should be reserved. The actual overhead depends on the draft-component size, numerical precision, tensor-parallel configuration, and runtime buffers.

---

## Where to find the pretrained draft models

Several organizations now publish pretrained draft models on Hugging Face. Google provides MTP assistants for Gemma 4, while Z-Lab maintains a collection of DFlash checkpoints. Red Hat AI offers draft models across EAGLE-3, DFlash, and DSpark, and DeepSeek's DeepSpec collection provides matched checkpoints for all three methods. LightSeek focuses on EAGLE-based draft models for Kimi, while Inferact publishes draft models for MiniMax and Kimi.

| Draft-model publisher | Methods | Representative models and targets |
| --- | --- | --- |
| Google | Gemma 4 MTP | Assistant checkpoints for Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B target models. [[7]](#ref-7) |
| LightSeek Foundation | EAGLE-3 and EAGLE-3.1 | EAGLE-based draft models for Kimi-K2.5, Kimi-K2.6, and Kimi-K2.7-Coder, including standard and MLA variants. [[8]](#ref-8) |
| Red Hat AI | EAGLE-3, DFlash, and DSpark | A collection covering target families such as Llama, Qwen, Gemma, GPT-OSS, GLM, Nemotron, and Mistral. Common suffixes include -speculator.eagle3, -speculator.dflash, and -speculator.dspark. [[9]](#ref-9) |
| Z-Lab | DFlash | DFlash checkpoints for targets including Qwen3, Qwen3.5, Qwen3.6, Gemma 4, Kimi, MiniMax, GPT-OSS, and Llama. Checkpoint names generally follow the &lt;target&gt;-DFlash pattern. [[10]](#ref-10) |
| DeepSeek AI | EAGLE-3, DFlash, and DSpark | The DeepSpec collection provides versions of all three methods for Qwen3-4B, Qwen3-8B, and Qwen3-14B, as well as Gemma 4 12B. Examples include eagle3_qwen3_8b_ttt7, dflash_qwen3_8b_block7, and dspark_qwen3_8b_block7. [[11]](#ref-11) |
| Inferact | EAGLE-3 and DSpark | Draft models including Inferact/MiniMax-M3-EAGLE3, its GQA variants, and Inferact/Kimi-K3-DSpark. [[12]](#ref-12) |

---

## Experimental setup and measurements

After enabling speculative decoding, the practical question is whether the additional drafting work improves end-to-end serving performance. Candidate tokens do not need to be correct at every position because the target model evaluates them before they are committed. Performance therefore depends on how many proposed tokens are accepted and whether the saved target-model decoding work outweighs the cost of drafting and verification.

We evaluate model quality and serving performance using task-grounded benchmarks rather than random token sequences. Acceptance behavior depends on the structure and predictability of actual model outputs, so task-based prompts provide a more representative view of practical performance.

The main performance indicators are:

- Output-token throughput and speedup over the non-speculative baseline.
- Mean accepted length and draft-token acceptance rates, where available.
- Model quality relative to the non-speculative baseline.

### Models and experiment coverage

The experiments cover five speculative-drafting approaches across several target-model families. A check mark indicates that benchmark results are available for that target-method combination; a dash indicates that the combination was not included in the current experiments.

<table class="experiment-coverage-table">
  <thead>
    <tr>
      <th>Target model</th>
      <th>Native MTP</th>
      <th>Gemma 4 MTP</th>
      <th>EAGLE-3</th>
      <th>DFlash</th>
      <th>DSpark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><span class="experiment-coverage-table__target">google/gemma-4-26B-A4B-it</span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Google</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Red Hat AI</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">google/gemma-4-31B-it</span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Google</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Red Hat AI</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Red Hat AI</span></span></td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">Qwen/Qwen3-8B</span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Red Hat AI</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>DeepSeek</span></span></td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">Qwen/Qwen3.5-27B</span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Built-in</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">Qwen/Qwen3.5-122B-A10B</span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Built-in</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">Qwen/Qwen3.6-27B</span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Built-in</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">Qwen/Qwen3.6-35B-A3B</span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Built-in</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">moonshotai/Kimi-K2.5</span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>LightSeek</span></span></td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Z-Lab</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
    <tr>
      <td><span class="experiment-coverage-table__target">MiniMaxAI/MiniMax-M3-MXFP8</span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
      <td><span class="experiment-coverage-table__entry"><span>✓</span><span>Inferact</span></span></td>
      <td class="experiment-coverage-table__dash">-</td>
      <td class="experiment-coverage-table__dash">-</td>
    </tr>
  </tbody>
</table>

The table summarizes the target-method combinations included in the experiments and shows how speculative decoding behaves across different models, workloads, and proposal lengths. Each result should be interpreted within its test configuration, since model architecture, active parameter count, draft-component size, workload, and serving conditions can all affect performance.

### Throughput measurements

For throughput, we measure generated tokens per second against a standard autoregressive baseline and sweep the number of speculative tokens to study how speculation depth affects end-to-end serving throughput.

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<div class="plotly-chart-header">
  <div class="plotly-chart-controls">
    <label class="plotly-chart-select" for="plotly-throughput-target">
      <span>Target model</span>
      <select id="plotly-throughput-target"></select>
    </label>
  </div>
</div>
<div id="plotly-throughput-summary" style="width: 100%; height: 560px;"></div>
<script src="/assets/figures/2026-08-20-speculative-decoding-amd-gpus/plotly-throughput-summary.js?v=target-model-ids"></script>

<p align="center"><em>Figure 4. Measured output throughput by method and experiment, with the non-speculative baseline included as a reference. Use the selector to switch target models; hover over bars to see speedup and selected proposal length N.</em></p>

### Main observations

The measurements varied by target model, drafting method, workload, and proposal length.

For gemma-4-26B-A4B-it, the largest measured throughput ratios within the tested sweep were 2.74× and 2.62× for Gemma 4 MTP on GSM8K and MBPP, respectively, and 2.87× and 2.79× for DFlash on MATH500 and HumanEval. The EAGLE-3 measurements ranged from 2.11× to 2.27× across the four datasets.

For gemma-4-31B-it, Gemma 4 MTP measurements reached 2.00× on GSM8K and 1.99× on MBPP, while DFlash reached 2.34× on MATH500 and 2.05× on HumanEval. The EAGLE-3 and DSpark measurements were also above baseline across the four evaluated datasets. The proposal length associated with the largest measured throughput varied by workload.

For Qwen3-8B, the DSpark measurements ranged from 1.15× on MATH500 to 1.63× on GSM8K. DFlash measurements ranged from 1.08× to 1.27×. EAGLE-3 was above baseline on GSM8K, HumanEval, and MBPP, while its largest measured MATH500 value remained below the baseline.

For Qwen3.5-27B, Qwen3.5-122B-A10B, and Qwen3.6-27B, the maximum measured native-MTP values within the tested sweeps were higher than the corresponding maximum DFlash values. The largest ratio in this group was 2.20× for Qwen3.5-122B-A10B on MATH500. The native-MTP proposal length associated with the largest measured throughput ranged from N=4 to N=7, depending on the model and dataset.

For Qwen3.6-35B-A3B, the DFlash measurements ranged from 1.77× to 2.06×, with the largest value occurring at N=7 for each of the four datasets. Native-MTP measurements ranged from 1.28× to 1.49×, with the largest values occurring at N=6. The difference from the Qwen3.6-27B measurements shows that results can vary between models in the same family.

For MiniMax-M3-MXFP8, the EAGLE-3 measurements reached 2.09× on HumanEval at N=4. For Kimi-K2.5, EAGLE-3 measurements reached up to 2.33× and DFlash measurements reached up to 2.68×. Within the tested sweeps, the largest EAGLE-3 values generally occurred at N=4, while the largest DFlash values occurred at N=7.

Across the experiments, the proposal length associated with the largest measured throughput was not constant. For the sequential methods, throughput often increased over the first few values of N before reaching a plateau. For DFlash and DSpark, N=7 was frequently among the higher-throughput settings, while larger values did not consistently increase throughput.

These observations reflect the hardware, software, target model, draft checkpoint, workload, and sweep settings used in this study.

---

## Tuning considerations

Speculative decoding should be treated as a runtime optimization rather than a fixed setting that works equally well for every workload. The value of `num_speculative_tokens` associated with the highest throughput depends on how many proposed tokens are accepted and whether the avoided target-model decode work outweighs the cost of drafting and verification.

Observability is therefore important. A model-card recommendation or example configuration provides a useful starting point, but the final setting should be selected using representative workloads and end-to-end measurements. Useful signals include throughput, mean accepted length, overall acceptance rate, and per-position acceptance rate.

A larger proposal window gives the system more opportunities to commit several tokens in one verification pass. However, acceptance may decrease at later draft positions. When this happens, the additional candidates contribute little while still adding drafting and verification work, causing throughput to flatten or regress.

### Start from a supported configuration

For native MTP, N=1 is a conservative starting point because it introduces the least additional sequential drafting work:

```json
{"method": "mtp", "num_speculative_tokens": 1}
```

After confirming correctness and stability, sweep larger values such as 2, 3, 4, 5, 6, and 7.

In our measurements, the native-MTP setting associated with the largest measured throughput varied by target model and workload. For Qwen3.5-27B, the largest measured throughput occurred at N=5 for GSM8K and MATH500, N=4 for HumanEval and MBPP, and N=3 for MT-Bench. For Qwen3.5-122B-A10B, the largest measured throughput across the four listed reasoning and code datasets occurred at N=7.

The Qwen3.6 measurements also show that this setting can change between models in the same family. For Qwen3.6-27B, the largest measured values occurred at N=4 or N=5, while throughput for the tested Qwen3.6-35B-A3B configurations increased through N=6.

For Gemma 4 MTP and EAGLE-3, increasing N also adds sequential drafting work. A short sweep is therefore useful even when the checkpoint provides a recommended configuration. In our Gemma 4 and EAGLE-3 experiments, measured throughput generally increased over the first few values of N before reaching a plateau.

For DFlash, begin with the proposal lengths recommended or supported by the draft checkpoint. Many DFlash checkpoints are trained with a fixed block size. For example, when:

```text
block_size = 16
```

the maximum proposal length is normally:

```text
num_speculative_tokens = 15
```

because the first position is the confirmed anchor token and the remaining 15 positions are draft candidates.

This is the maximum supported proposal length, not necessarily the highest-throughput setting. In practice, it is useful to test smaller values such as:

```text
N = 3, 7, 11, 15
```

Across our DFlash experiments, N=7 was frequently among the higher-throughput settings. For some workloads, the largest measured throughput occurred at N=11.

For DSpark, `num_speculative_tokens` sets the number of candidate tokens generated in each speculative round. In our vLLM experiments, the full configured proposal was submitted for target-model verification, so values such as N=3 and N=7 should be compared using end-to-end throughput.

### Monitor acceptance behavior

Relevant signals to monitor include:

| Signal | What it shows |
| --- | --- |
| Throughput | How end-to-end serving performance changes relative to the non-speculative baseline |
| Mean accepted length | How many draft tokens are committed per speculative round on average |
| Overall acceptance rate | What proportion of proposed draft tokens are accepted |
| Per-position acceptance rate | Whether later positions in the proposal remain useful |

Per-position acceptance is particularly helpful when tuning proposal length. If the first few positions are accepted frequently but later positions contribute very little, reducing `num_speculative_tokens` may improve throughput by avoiding unnecessary draft work.

Acceptance metrics should be interpreted together with throughput. A method may show higher throughput relative to baseline even with a lower acceptance rate when draft generation is inexpensive. Conversely, a high acceptance rate does not necessarily correspond to higher throughput when the draft component adds additional overhead.

### Match the sweep to the workload

Different workloads can produce different acceptance patterns.

In our GSM8K and MATH500 measurements, medium or deeper proposal lengths were often associated with higher measured throughput within the tested sweeps. For native MTP on Qwen3.5-122B-A10B, measured throughput increased through N=7. For DFlash, higher measured values frequently occurred at N=7 or N=11.

For HumanEval and MBPP, moderate proposal lengths were often among the higher-throughput settings. Code contains predictable local structure, but formatting, identifiers, and implementation choices can cause an otherwise plausible continuation to diverge.

### Example tuning workflow

1. Begin with a configuration supported or recommended for the checkpoint.

2. Benchmark using representative prompts and generation settings.

3. Record throughput, mean accepted length, and acceptance rates.

4. Sweep several smaller and larger proposal lengths.

5. Select a setting based on the metric most relevant to the intended workload. In these experiments, end-to-end serving throughput was the primary selection metric.

The selected configuration does not necessarily have the longest proposal, the highest acceptance rate, or the largest mean accepted length. Selection should consider the trade-off among drafting cost, verification cost, accepted tokens, and the metric most relevant to the intended workload.

---

## Training a speculator for a new target model

This guide does not cover speculator training in depth. The following workflow summarizes practical considerations from the referenced vLLM Speculators and DeepSpec resources [[13]](#ref-13), [[14]](#ref-14), and [[15]](#ref-15).

A typical workflow is:

1. Prepare representative prompts.
2. Generate responses with the target model.
3. Choose a hidden-state generation mode.
4. Collect the required target-model hidden states.
5. Train the speculator.
6. Test acceptance and serving throughput.

### Prepare representative prompts

Start with prompts that reflect the expected workload, such as chat, mathematics, code generation, tool use, or multilingual tasks. Keep a separate set of prompts for evaluation.

The responses used for training should be generated by the exact target model that the speculator will support. The tokenizer, chat template, thinking mode, and generation configuration should also match the intended deployment. The vLLM documentation emphasizes that applying the target model's tokenizer or chat template to existing responses does not make the data target-specific; the responses themselves must come from the target model.

### Choose how to obtain hidden states

The speculator receives internal hidden states from the target model during training. The vLLM Speculators workflow supports three ways to provide them:

| Training mode | How it works | Main consideration |
| --- | --- | --- |
| Online | Hidden states are generated by a running vLLM server when needed and discarded afterward | Avoids a large disk cache but requires resources for target inference and training at the same time |
| Offline | Hidden states are generated and stored before training begins | Frees all GPUs for training afterward but requires substantial storage |
| Hybrid | Hidden states are generated and cached during the first epoch, then reused | Pays the generation cost once without requiring a separate preprocessing stage |

The selected mode changes where the hidden states come from; the remaining training workflow is largely the same.

### Collect target-model information

A vLLM server can run the target model and expose hidden states from the layers required by the selected drafting method. When custom target layers are chosen, the same layer selections must also be used in the speculator-training configuration.

The information collected depends on the method:

- EAGLE-3 uses hidden states from selected target-model layers for autoregressive drafting. [[4]](#ref-4)
- DFlash uses target-model features to train a network that predicts a block of future positions in parallel. [[16]](#ref-16)
- DSpark adds lightweight sequential and confidence heads to a DFlash-style draft network. [[6]](#ref-6)
- MTP training fine-tunes the target model's own MTP component and therefore requires a target model that already contains compatible MTP layers. [[13]](#ref-13)

### Train and test the speculator

The speculator configuration must match the target model's hidden size, vocabulary, tokenizer, and selected target layers. Method-specific settings such as draft-network depth, block size, sequence length, and learning rate must also be selected.

After training, inspect the checkpoint and serve it together with the target model in vLLM. Training loss alone is not enough to judge the result; the important measurements are accepted length, acceptance rate, draft latency, GPU memory use, and end-to-end serving throughput. The vLLM Speculators tutorial covers the complete path from data preparation and hidden-state extraction to checkpoint testing and serving.

When acceptance is weak for a particular workload, the prompt mixture or training configuration can be adjusted and the process repeated. The main principle is to use the same target model, generation mode, and representative workload that the speculator is expected to support.

---

## Summary

This blog explored speculative decoding in vLLM as a draft-and-verify approach for LLM serving. A draft component proposes candidate future tokens, and the target model evaluates the proposal before any tokens are committed.

We examined five drafting approaches: native MTP, Gemma 4 MTP, EAGLE-3, DFlash, and DSpark. They differ mainly in how they use information from the target model and whether candidate tokens are generated sequentially, in parallel, or through a combination of parallel prediction and lightweight sequential correction.

The experiments covered selected Gemma, Qwen, MiniMax, and Kimi models on AMD Instinct™ MI300X and MI355X GPUs using the ROCm™ software platform. Measured throughput varied across target models, draft checkpoints, workloads, proposal lengths, and serving configurations.

Across the tested configurations, some settings produced smaller changes or throughput below the non-speculative baseline, while several model-workload combinations produced throughput ratios above 2×. Examples at the upper end of the observed range included 2.87× for DFlash on gemma-4-26B-A4B-it, 2.83× for Gemma 4 MTP on the same target, and 2.68× for DFlash on Kimi-K2.5.

Proposal length was also an important experimental variable. Increasing `num_speculative_tokens` sometimes increased throughput over the first few settings, while larger values could lead to a plateau or lower throughput. Checkpoint recommendations can provide starting points, but representative workload measurements and acceptance metrics are needed when selecting a deployment configuration.

## Future work

Future benchmarking could include non-learned approaches such as n-gram speculation and suffix decoding, particularly for workloads with repeated token patterns such as code editing and agentic loops.

Broader evaluation across concurrency levels, prompt and output lengths, batch sizes, and sampling settings would also help show how speculative decoding behaves under different serving conditions.

Another useful direction is to study how speculator training data affects acceptance across code, mathematics, chat, multilingual prompts, tool use, and structured output. This could provide clearer guidance when choosing or training a draft checkpoint for a specific workload.

Finally, deeper profiling of draft generation, target verification, KV-cache behavior, graph execution, and scheduling would help explain the performance differences observed across target models and workloads.

---

## References

1. <a id="ref-1"> </a> vLLM documentation, "Speculative Decoding" <a href="https://docs.vllm.ai/en/latest/features/speculative_decoding/">https://docs.vllm.ai/en/latest/features/speculative_decoding/</a>
2. <a id="ref-2"> </a> vLLM documentation, "MTP Speculative Decoding" <a href="https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/">https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/</a>
3. <a id="ref-3"> </a> Google Developers Blog, "Multi-token prediction in Gemma 4" <a href="https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/">https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/</a>
4. <a id="ref-4"> </a> EAGLE-3 paper, "Scaling up Inference Acceleration of Large Language Models via Training-Time Test" <a href="https://arxiv.org/pdf/2503.01840">https://arxiv.org/pdf/2503.01840</a>
5. <a id="ref-5"> </a> Z-Lab, "DFlash" GitHub repository <a href="https://github.com/z-lab/dflash">https://github.com/z-lab/dflash</a>
6. <a id="ref-6"> </a> DSpark paper, arXiv preprint <a href="https://arxiv.org/pdf/2607.05147">https://arxiv.org/pdf/2607.05147</a>
7. <a id="ref-7"> </a> Google, "Gemma 4" Hugging Face collection <a href="https://huggingface.co/collections/google/gemma-4">https://huggingface.co/collections/google/gemma-4</a>
8. <a id="ref-8"> </a> LightSeek Foundation model collection on Hugging Face <a href="https://huggingface.co/lightseekorg/models">https://huggingface.co/lightseekorg/models</a>
9. <a id="ref-9"> </a> Red Hat AI, "Speculator Models" Hugging Face collection <a href="https://huggingface.co/collections/RedHatAI/speculator-models">https://huggingface.co/collections/RedHatAI/speculator-models</a>
10. <a id="ref-10"> </a> Z-Lab, "DFlash" Hugging Face collection <a href="https://huggingface.co/collections/z-lab/dflash">https://huggingface.co/collections/z-lab/dflash</a>
11. <a id="ref-11"> </a> DeepSeek-AI, "DeepSpec" Hugging Face collection <a href="https://huggingface.co/collections/deepseek-ai/deepspec">https://huggingface.co/collections/deepseek-ai/deepspec</a>
12. <a id="ref-12"> </a> Inferact model collection on Hugging Face <a href="https://huggingface.co/Inferact/models">https://huggingface.co/Inferact/models</a>
13. <a id="ref-13"> </a> vLLM Speculators documentation, "Training a Speculator" <a href="https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/">https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/</a>
14. <a id="ref-14"> </a> vLLM Project, "Speculators" GitHub repository <a href="https://github.com/vllm-project/speculators">https://github.com/vllm-project/speculators</a>
15. <a id="ref-15"> </a> DeepSeek-AI, "DeepSpec" GitHub repository <a href="https://github.com/deepseek-ai/DeepSpec">https://github.com/deepseek-ai/DeepSpec</a>
16. <a id="ref-16"> </a> DFlash paper, arXiv preprint <a href="https://arxiv.org/pdf/2602.06036">https://arxiv.org/pdf/2602.06036</a>

## Appendix

The appendix focuses on acceptance behavior by draft position. Choose a target model, drafting method, and experiment to view one larger per-position acceptance heatmap. Rows are proposal lengths `N`; columns are draft positions; darker cells indicate higher acceptance. Each row also includes measured speedup and output throughput for context.

<div class="appendix-acceptance-view">
  <div class="appendix-acceptance-controls">
    <label class="plotly-chart-select" for="appendix-model-select">
      <span>Target model</span>
      <select id="appendix-model-select">
        <option value="google/gemma-4-26B-A4B-it" selected>google/gemma-4-26B-A4B-it</option>
        <option value="google/gemma-4-31B-it">google/gemma-4-31B-it</option>
        <option value="Qwen/Qwen3-8B">Qwen/Qwen3-8B</option>
        <option value="Qwen/Qwen3.5-27B">Qwen/Qwen3.5-27B</option>
        <option value="Qwen/Qwen3.5-122B-A10B">Qwen/Qwen3.5-122B-A10B</option>
        <option value="Qwen/Qwen3.6-27B">Qwen/Qwen3.6-27B</option>
        <option value="Qwen/Qwen3.6-35B-A3B">Qwen/Qwen3.6-35B-A3B</option>
        <option value="moonshotai/Kimi-K2.5">moonshotai/Kimi-K2.5</option>
        <option value="MiniMaxAI/MiniMax-M3-MXFP8">MiniMaxAI/MiniMax-M3-MXFP8</option>
      </select>
    </label>
    <label class="plotly-chart-select plotly-chart-select--metric" for="appendix-method-select">
      <span>Method</span>
      <select id="appendix-method-select"></select>
    </label>
    <label class="plotly-chart-select plotly-chart-select--metric" for="appendix-benchmark-select">
      <span>Experiment</span>
      <select id="appendix-benchmark-select"></select>
    </label>
  </div>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-gemma-4-mtp-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="Gemma 4 MTP" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-26B-A4B-it</code> / Gemma 4 MTP / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 2,344 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.73x | 4,060 tok/s</span><span>MAL 1.95 | AR 94.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=1, p1: 95%">95%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.28x | 5,334 tok/s</span><span>MAL 2.83 | AR 91.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=2, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.883" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=2, p2: 88%">88%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.54x | 5,945 tok/s</span><span>MAL 3.64 | AR 87.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.944" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.878" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=3, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.814" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=3, p3: 81%">81%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.66x | 6,230 tok/s</span><span>MAL 4.35 | AR 83.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.939" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=4, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.870" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=4, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.804" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=4, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.740" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=4, p4: 74%">74%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.74x | 6,434 tok/s</span><span>MAL 5.00 | AR 80.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.939" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=5, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.867" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=5, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.800" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=5, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.733" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=5, p4: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.663" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / GSM8K, N=5, p5: 66%">66%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-gemma-4-mtp-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="Gemma 4 MTP" data-benchmark="math500">
    <h3><code>google/gemma-4-26B-A4B-it</code> / Gemma 4 MTP / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 2,181 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.68x | 3,671 tok/s</span><span>MAL 1.95 | AR 95.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=1, p1: 95%">95%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.27x | 4,961 tok/s</span><span>MAL 2.84 | AR 91.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=2, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.888" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=2, p2: 89%">89%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.53x | 5,510 tok/s</span><span>MAL 3.64 | AR 88.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.882" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=3, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.817" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=3, p3: 82%">82%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.73x | 5,955 tok/s</span><span>MAL 4.36 | AR 84.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.942" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=4, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.875" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=4, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.807" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=4, p3: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.740" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=4, p4: 74%">74%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.83x | 6,161 tok/s</span><span>MAL 5.01 | AR 80.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.941" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=5, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.871" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=5, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.802" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=5, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.733" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=5, p4: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.664" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MATH500, N=5, p5: 66%">66%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-gemma-4-mtp-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="Gemma 4 MTP" data-benchmark="humaneval">
    <h3><code>google/gemma-4-26B-A4B-it</code> / Gemma 4 MTP / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,854 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.78x | 3,310 tok/s</span><span>MAL 1.94 | AR 93.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.938" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=1, p1: 94%">94%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.09x | 3,871 tok/s</span><span>MAL 2.79 | AR 89.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.933" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=2, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.862" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=2, p2: 86%">86%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.33x | 4,326 tok/s</span><span>MAL 3.56 | AR 85.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.928" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=3, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.854" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=3, p2: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.781" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=3, p3: 78%">78%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.50x | 4,642 tok/s</span><span>MAL 4.24 | AR 81.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.925" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=4, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.846" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=4, p2: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.772" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=4, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.702" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=4, p4: 70%">70%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.59x | 4,810 tok/s</span><span>MAL 4.81 | AR 76.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.917" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=5, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.836" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=5, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.758" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=5, p3: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.687" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=5, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.615" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / HumanEval, N=5, p5: 62%">62%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-gemma-4-mtp-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="Gemma 4 MTP" data-benchmark="mbpp">
    <h3><code>google/gemma-4-26B-A4B-it</code> / Gemma 4 MTP / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 2,163 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.73x | 3,744 tok/s</span><span>MAL 1.90 | AR 90.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=1, p1: 91%">91%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.26x | 4,882 tok/s</span><span>MAL 2.70 | AR 84.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.899" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=2, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=2, p2: 80%">80%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.50x | 5,413 tok/s</span><span>MAL 3.38 | AR 79.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.896" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.790" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=3, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.690" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=3, p3: 69%">69%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.60x | 5,628 tok/s</span><span>MAL 3.93 | AR 73.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.889" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=4, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.781" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=4, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.677" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=4, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.584" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=4, p4: 58%">58%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.62x | 5,662 tok/s</span><span>MAL 4.37 | AR 67.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.885" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=5, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.771" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=5, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.662" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=5, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.568" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=5, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.486" title="google/gemma-4-26B-A4B-it / Gemma 4 MTP / MBPP, N=5, p5: 49%">49%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-eagle-3-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="EAGLE-3" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-26B-A4B-it</code> / EAGLE-3 / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 2,344 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.55x | 3,624 tok/s</span><span>MAL 1.83 | AR 83.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.830" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=1, p1: 83%">83%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.09x | 4,888 tok/s</span><span>MAL 2.47 | AR 73.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.821" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=2, p1: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.653" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=2, p2: 65%">65%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.16x | 5,063 tok/s</span><span>MAL 2.94 | AR 64.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.811" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=3, p1: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.641" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=3, p2: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.490" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=3, p3: 49%">49%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.16x | 5,059 tok/s</span><span>MAL 3.27 | AR 56.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.803" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=4, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.632" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=4, p2: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.482" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=4, p3: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.351" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=4, p4: 35%">35%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.15x | 5,040 tok/s</span><span>MAL 3.49 | AR 49.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.798" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=5, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.625" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=5, p2: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.474" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=5, p3: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.345" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=5, p4: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.244" title="google/gemma-4-26B-A4B-it / EAGLE-3 / GSM8K, N=5, p5: 24%">24%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-eagle-3-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="EAGLE-3" data-benchmark="math500">
    <h3><code>google/gemma-4-26B-A4B-it</code> / EAGLE-3 / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 2,181 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.54x | 3,362 tok/s</span><span>MAL 1.87 | AR 87.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.872" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=1, p1: 87%">87%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.07x | 4,516 tok/s</span><span>MAL 2.57 | AR 78.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.861" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=2, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.706" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=2, p2: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.21x | 4,810 tok/s</span><span>MAL 3.09 | AR 69.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.851" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=3, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.692" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=3, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.549" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=3, p3: 55%">55%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.27x | 4,953 tok/s</span><span>MAL 3.47 | AR 61.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.845" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=4, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.683" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=4, p2: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.536" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=4, p3: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.404" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=4, p4: 40%">40%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.23x | 4,861 tok/s</span><span>MAL 3.73 | AR 54.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.842" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=5, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.675" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=5, p2: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.527" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=5, p3: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.397" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=5, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.289" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MATH500, N=5, p5: 29%">29%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-eagle-3-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="EAGLE-3" data-benchmark="humaneval">
    <h3><code>google/gemma-4-26B-A4B-it</code> / EAGLE-3 / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,854 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.51x | 2,802 tok/s</span><span>MAL 1.80 | AR 79.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.799" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=1, p1: 80%">80%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.85x | 3,438 tok/s</span><span>MAL 2.40 | AR 69.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.787" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=2, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.611" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=2, p2: 61%">61%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.92x | 3,562 tok/s</span><span>MAL 2.81 | AR 60.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.778" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=3, p1: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.595" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=3, p2: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.435" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=3, p3: 44%">44%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.16x | 3,997 tok/s</span><span>MAL 3.07 | AR 51.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.765" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=4, p1: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.585" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=4, p2: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.424" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=4, p3: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.296" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=4, p4: 30%">30%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.85x | 3,435 tok/s</span><span>MAL 3.22 | AR 44.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.761" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=5, p1: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.572" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=5, p2: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.410" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=5, p3: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.280" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=5, p4: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.197" title="google/gemma-4-26B-A4B-it / EAGLE-3 / HumanEval, N=5, p5: 20%">20%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-eagle-3-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="EAGLE-3" data-benchmark="mbpp">
    <h3><code>google/gemma-4-26B-A4B-it</code> / EAGLE-3 / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 2,163 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.54x | 3,328 tok/s</span><span>MAL 1.79 | AR 79.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.790" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=1, p1: 79%">79%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>2.08x | 4,506 tok/s</span><span>MAL 2.36 | AR 68.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.778" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=2, p1: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.584" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=2, p2: 58%">58%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.11x | 4,559 tok/s</span><span>MAL 2.75 | AR 58.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.766" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=3, p1: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.567" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=3, p2: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.416" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=3, p3: 42%">42%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.11x | 4,574 tok/s</span><span>MAL 3.00 | AR 50.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.758" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=4, p1: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.555" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=4, p2: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.401" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=4, p3: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.285" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=4, p4: 28%">28%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.05x | 4,426 tok/s</span><span>MAL 3.17 | AR 43.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.750" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=5, p1: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.548" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=5, p2: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.394" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=5, p3: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.278" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=5, p4: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.197" title="google/gemma-4-26B-A4B-it / EAGLE-3 / MBPP, N=5, p5: 20%">20%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-dflash-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-26B-A4B-it</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 2,344 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.43x | 5,697 tok/s</span><span>MAL 3.36 | AR 78.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.896" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.789" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=3, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.679" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=3, p3: 68%">68%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.70x | 6,327 tok/s</span><span>MAL 5.05 | AR 57.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.875" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.758" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.653" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.557" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.476" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p5: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.400" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p6: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.332" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=7, p7: 33%">33%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.44x | 5,724 tok/s</span><span>MAL 5.71 | AR 42.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.867" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.742" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.634" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p3: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.537" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p4: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.453" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p5: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.378" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p6: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.314" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p7: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.259" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.215" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.174" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p10: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.137" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=11, p11: 14%">14%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>2.12x | 4,973 tok/s</span><span>MAL 5.89 | AR 32.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.863" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.732" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.622" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p3: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.525" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p4: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.440" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p5: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.366" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.304" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.249" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p8: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.204" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p9: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.166" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p10: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.132" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p11: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.104" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p12: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.080" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p13: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.060" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p14: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.042" title="google/gemma-4-26B-A4B-it / DFlash / GSM8K, N=15, p15: 4%">4%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-dflash-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="DFlash" data-benchmark="math500">
    <h3><code>google/gemma-4-26B-A4B-it</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 2,181 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.49x | 5,427 tok/s</span><span>MAL 3.43 | AR 80.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.808" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.711" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.87x | 6,267 tok/s</span><span>MAL 5.26 | AR 60.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.881" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.766" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.671" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.589" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p4: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.517" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.450" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p6: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.387" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=7, p7: 39%">39%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.70x | 5,888 tok/s</span><span>MAL 6.09 | AR 46.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.868" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.745" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.645" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.558" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.486" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p5: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.422" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p6: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.365" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p7: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.316" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p8: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.270" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p9: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.228" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p10: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.190" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=11, p11: 19%">19%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>2.40x | 5,232 tok/s</span><span>MAL 6.40 | AR 36.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.862" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.735" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.633" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p3: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.546" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p4: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.474" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p5: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.408" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p6: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.352" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p7: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.302" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p8: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.257" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p9: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.217" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p10: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.182" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p11: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.150" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p12: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.121" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p13: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.095" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p14: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.072" title="google/gemma-4-26B-A4B-it / DFlash / MATH500, N=15, p15: 7%">7%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-dflash-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>google/gemma-4-26B-A4B-it</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,854 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.29x | 4,238 tok/s</span><span>MAL 3.29 | AR 76.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.875" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=3, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.758" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=3, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.655" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=3, p3: 66%">66%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.79x | 5,183 tok/s</span><span>MAL 4.90 | AR 55.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.849" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.714" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.609" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.526" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p4: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.458" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p5: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.399" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p6: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.345" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=7, p7: 35%">35%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.41x | 4,465 tok/s</span><span>MAL 5.50 | AR 40.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.830" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.685" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.573" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.485" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p4: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.417" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p5: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.359" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p6: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.310" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p7: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.263" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.227" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p9: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.193" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p10: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.160" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=11, p11: 16%">16%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>2.26x | 4,193 tok/s</span><span>MAL 5.76 | AR 31.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.824" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p1: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.679" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p2: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.567" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.477" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p4: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.406" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.347" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.298" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.256" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.217" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.184" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p10: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.152" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p11: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.125" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p12: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.099" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p13: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.077" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p14: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.058" title="google/gemma-4-26B-A4B-it / DFlash / HumanEval, N=15, p15: 6%">6%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-26b-a4b-it-dflash-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-26B-A4B-it" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>google/gemma-4-26B-A4B-it</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 2,163 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.34x | 5,065 tok/s</span><span>MAL 3.08 | AR 69.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.836" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=3, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.690" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=3, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.557" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=3, p3: 56%">56%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.41x | 5,214 tok/s</span><span>MAL 4.22 | AR 45.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.803" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.644" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p2: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.518" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p3: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.417" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p4: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.340" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p5: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.273" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p6: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.221" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=7, p7: 22%">22%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.14x | 4,621 tok/s</span><span>MAL 4.56 | AR 32.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.789" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.620" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p2: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.491" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p3: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.391" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p4: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.317" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p5: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.255" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p6: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.207" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p7: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.168" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p8: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.135" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p9: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.106" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p10: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.083" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=11, p11: 8%">8%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.86x | 4,018 tok/s</span><span>MAL 4.69 | AR 24.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.788" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.619" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p2: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.488" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p3: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.384" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p4: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.310" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p5: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.248" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p6: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.200" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p7: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.162" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p8: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.130" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p9: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.104" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p10: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.082" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p11: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.063" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p12: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.049" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p13: 5%">5%</td>
              <td class="acceptance-cell" style="--accept: 0.036" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p14: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.026" title="google/gemma-4-26B-A4B-it / DFlash / MBPP, N=15, p15: 3%">3%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-gemma-4-mtp-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="Gemma 4 MTP" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-31B-it</code> / Gemma 4 MTP / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,631 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.52x | 2,475 tok/s</span><span>MAL 1.95 | AR 95.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=1, p1: 95%">95%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.78x | 2,906 tok/s</span><span>MAL 2.85 | AR 92.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.953" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=2, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.893" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=2, p2: 89%">89%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.94x | 3,160 tok/s</span><span>MAL 3.66 | AR 88.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.950" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.888" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=3, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.824" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=3, p3: 82%">82%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.00x | 3,267 tok/s</span><span>MAL 4.40 | AR 84.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.949" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.883" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=4, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=4, p3: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.749" title="google/gemma-4-31B-it / Gemma 4 MTP / GSM8K, N=4, p4: 75%">75%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-gemma-4-mtp-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="Gemma 4 MTP" data-benchmark="math500">
    <h3><code>google/gemma-4-31B-it</code> / Gemma 4 MTP / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,365 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.54x | 2,097 tok/s</span><span>MAL 1.96 | AR 95.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.956" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=1, p1: 96%">96%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.86x | 2,542 tok/s</span><span>MAL 2.85 | AR 92.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=2, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.897" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=2, p2: 90%">90%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.09x | 2,851 tok/s</span><span>MAL 3.67 | AR 88.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.890" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=3, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.826" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=3, p3: 83%">83%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.20x | 3,006 tok/s</span><span>MAL 4.41 | AR 85.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.884" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=4, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.820" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=4, p3: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.754" title="google/gemma-4-31B-it / Gemma 4 MTP / MATH500, N=4, p4: 75%">75%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-gemma-4-mtp-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="Gemma 4 MTP" data-benchmark="humaneval">
    <h3><code>google/gemma-4-31B-it</code> / Gemma 4 MTP / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,228 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.46x | 1,793 tok/s</span><span>MAL 1.96 | AR 95.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.958" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=1, p1: 96%">96%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.76x | 2,163 tok/s</span><span>MAL 2.86 | AR 92.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=2, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.903" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=2, p2: 90%">90%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.97x | 2,419 tok/s</span><span>MAL 3.70 | AR 90.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.899" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=3, p2: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.849" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=3, p3: 85%">85%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.97x | 2,424 tok/s</span><span>MAL 4.43 | AR 85.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.884" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=4, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.827" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=4, p3: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.770" title="google/gemma-4-31B-it / Gemma 4 MTP / HumanEval, N=4, p4: 77%">77%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-gemma-4-mtp-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="Gemma 4 MTP" data-benchmark="mbpp">
    <h3><code>google/gemma-4-31B-it</code> / Gemma 4 MTP / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,519 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.55x | 2,360 tok/s</span><span>MAL 1.91 | AR 91.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.912" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=1, p1: 91%">91%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.81x | 2,743 tok/s</span><span>MAL 2.72 | AR 85.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.908" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=2, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.810" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=2, p2: 81%">81%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.97x | 2,997 tok/s</span><span>MAL 3.39 | AR 79.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.901" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.794" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=3, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.695" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=3, p3: 70%">70%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.99x | 3,020 tok/s</span><span>MAL 3.95 | AR 73.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.895" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=4, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.785" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=4, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.680" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=4, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.587" title="google/gemma-4-31B-it / Gemma 4 MTP / MBPP, N=4, p4: 59%">59%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-eagle-3-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="EAGLE-3" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-31B-it</code> / EAGLE-3 / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,631 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.48x | 2,420 tok/s</span><span>MAL 1.88 | AR 87.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.875" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=1, p1: 88%">88%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.69x | 2,756 tok/s</span><span>MAL 2.60 | AR 80.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.866" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=2, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=2, p2: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.79x | 2,915 tok/s</span><span>MAL 3.18 | AR 72.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.857" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=3, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.722" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=3, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.602" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=3, p3: 60%">60%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.77x | 2,883 tok/s</span><span>MAL 3.63 | AR 65.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.848" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=4, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.713" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=4, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.593" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=4, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.478" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=4, p4: 48%">48%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.79x | 2,913 tok/s</span><span>MAL 3.99 | AR 59.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.845" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=5, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.708" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=5, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.587" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=5, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.472" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=5, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.373" title="google/gemma-4-31B-it / EAGLE-3 / GSM8K, N=5, p5: 37%">37%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-eagle-3-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="EAGLE-3" data-benchmark="math500">
    <h3><code>google/gemma-4-31B-it</code> / EAGLE-3 / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,365 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.54x | 2,106 tok/s</span><span>MAL 1.91 | AR 90.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=1, p1: 91%">91%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.85x | 2,521 tok/s</span><span>MAL 2.69 | AR 84.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.900" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=2, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.787" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=2, p2: 79%">79%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.05x | 2,796 tok/s</span><span>MAL 3.33 | AR 77.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.894" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.776" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=3, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.664" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=3, p3: 66%">66%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.03x | 2,768 tok/s</span><span>MAL 3.84 | AR 71.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.887" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=4, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.766" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=4, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.651" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=4, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.539" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=4, p4: 54%">54%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.12x | 2,891 tok/s</span><span>MAL 4.24 | AR 64.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.883" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=5, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.759" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=5, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.641" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=5, p3: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.529" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=5, p4: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.429" title="google/gemma-4-31B-it / EAGLE-3 / MATH500, N=5, p5: 43%">43%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-eagle-3-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="EAGLE-3" data-benchmark="humaneval">
    <h3><code>google/gemma-4-31B-it</code> / EAGLE-3 / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,228 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.43x | 1,757 tok/s</span><span>MAL 1.87 | AR 87.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.874" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=1, p1: 87%">87%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.68x | 2,059 tok/s</span><span>MAL 2.60 | AR 79.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.862" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=2, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=2, p2: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.81x | 2,221 tok/s</span><span>MAL 3.19 | AR 72.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.859" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=3, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.727" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=3, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.601" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=3, p3: 60%">60%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.80x | 2,209 tok/s</span><span>MAL 3.64 | AR 66.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.853" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=4, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.720" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=4, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.594" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=4, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.475" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=4, p4: 48%">48%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.86x | 2,278 tok/s</span><span>MAL 3.97 | AR 59.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.845" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=5, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.709" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=5, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.581" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=5, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.464" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=5, p4: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.369" title="google/gemma-4-31B-it / EAGLE-3 / HumanEval, N=5, p5: 37%">37%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-eagle-3-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="EAGLE-3" data-benchmark="mbpp">
    <h3><code>google/gemma-4-31B-it</code> / EAGLE-3 / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,519 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.52x | 2,306 tok/s</span><span>MAL 1.85 | AR 84.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.847" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=1, p1: 85%">85%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.73x | 2,626 tok/s</span><span>MAL 2.52 | AR 75.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.837" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=2, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.679" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=2, p2: 68%">68%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.84x | 2,793 tok/s</span><span>MAL 3.03 | AR 67.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.827" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=3, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.667" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=3, p2: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.535" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=3, p3: 54%">54%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.80x | 2,736 tok/s</span><span>MAL 3.41 | AR 60.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.821" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=4, p1: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.658" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=4, p2: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.522" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=4, p3: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.407" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=4, p4: 41%">41%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.80x | 2,730 tok/s</span><span>MAL 3.67 | AR 53.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.814" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=5, p1: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.645" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=5, p2: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.511" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=5, p3: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.395" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=5, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.303" title="google/gemma-4-31B-it / EAGLE-3 / MBPP, N=5, p5: 30%">30%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dflash-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-31B-it</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,631 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.85x | 3,012 tok/s</span><span>MAL 3.51 | AR 83.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.931" title="google/gemma-4-31B-it / DFlash / GSM8K, N=3, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.837" title="google/gemma-4-31B-it / DFlash / GSM8K, N=3, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.743" title="google/gemma-4-31B-it / DFlash / GSM8K, N=3, p3: 74%">74%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.95x | 3,183 tok/s</span><span>MAL 5.54 | AR 64.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.921" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.723" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.635" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.554" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p5: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.480" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p6: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.411" title="google/gemma-4-31B-it / DFlash / GSM8K, N=7, p7: 41%">41%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.76x | 2,877 tok/s</span><span>MAL 6.47 | AR 49.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.913" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.802" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.704" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.612" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p4: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.529" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p5: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.456" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p6: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.388" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p7: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.333" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p8: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.284" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p9: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.241" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p10: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.204" title="google/gemma-4-31B-it / DFlash / GSM8K, N=11, p11: 20%">20%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.53x | 2,489 tok/s</span><span>MAL 6.84 | AR 38.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.910" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.796" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.696" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.602" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p4: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.518" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.443" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p6: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.374" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p7: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.317" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p8: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.270" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p9: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.228" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p10: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.193" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p11: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.162" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p12: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.134" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p13: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.109" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p14: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.087" title="google/gemma-4-31B-it / DFlash / GSM8K, N=15, p15: 9%">9%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dflash-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DFlash" data-benchmark="math500">
    <h3><code>google/gemma-4-31B-it</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,365 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.03x | 2,770 tok/s</span><span>MAL 3.56 | AR 85.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.939" title="google/gemma-4-31B-it / DFlash / MATH500, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.856" title="google/gemma-4-31B-it / DFlash / MATH500, N=3, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.769" title="google/gemma-4-31B-it / DFlash / MATH500, N=3, p3: 77%">77%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.34x | 3,197 tok/s</span><span>MAL 5.76 | AR 68.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.925" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.832" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.742" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p3: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.665" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.594" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p5: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.531" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p6: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.470" title="google/gemma-4-31B-it / DFlash / MATH500, N=7, p7: 47%">47%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.15x | 2,934 tok/s</span><span>MAL 6.88 | AR 53.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.915" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.815" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.717" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.636" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.563" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.498" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.442" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p7: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.390" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p8: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.343" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p9: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.300" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p10: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.260" title="google/gemma-4-31B-it / DFlash / MATH500, N=11, p11: 26%">26%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.91x | 2,605 tok/s</span><span>MAL 7.39 | AR 42.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.914" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.807" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.706" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p3: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.622" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.548" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p5: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.481" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p6: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.424" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p7: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.372" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p8: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.326" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p9: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.283" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p10: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.247" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p11: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.212" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p12: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.180" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p13: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.149" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p14: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.122" title="google/gemma-4-31B-it / DFlash / MATH500, N=15, p15: 12%">12%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dflash-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>google/gemma-4-31B-it</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,228 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.88x | 2,309 tok/s</span><span>MAL 3.60 | AR 86.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.943" title="google/gemma-4-31B-it / DFlash / HumanEval, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.868" title="google/gemma-4-31B-it / DFlash / HumanEval, N=3, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.792" title="google/gemma-4-31B-it / DFlash / HumanEval, N=3, p3: 79%">79%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.02x | 2,482 tok/s</span><span>MAL 5.82 | AR 68.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.923" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.834" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.750" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p3: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.674" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.605" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p5: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.545" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p6: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.488" title="google/gemma-4-31B-it / DFlash / HumanEval, N=7, p7: 49%">49%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.05x | 2,514 tok/s</span><span>MAL 7.00 | AR 54.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.918" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.718" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.638" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.568" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p5: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.511" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p6: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.456" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p7: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.406" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p8: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.364" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p9: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.321" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p10: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.282" title="google/gemma-4-31B-it / DFlash / HumanEval, N=11, p11: 28%">28%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.85x | 2,274 tok/s</span><span>MAL 7.51 | AR 43.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.909" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.802" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.703" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.619" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.548" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p5: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.490" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p6: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.436" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p7: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.387" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p8: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.347" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p9: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.301" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p10: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.264" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p11: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.227" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p12: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.191" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p13: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.159" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p14: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.127" title="google/gemma-4-31B-it / DFlash / HumanEval, N=15, p15: 13%">13%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dflash-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>google/gemma-4-31B-it</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,519 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.89x | 2,873 tok/s</span><span>MAL 3.31 | AR 77.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.897" title="google/gemma-4-31B-it / DFlash / MBPP, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.770" title="google/gemma-4-31B-it / DFlash / MBPP, N=3, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.647" title="google/gemma-4-31B-it / DFlash / MBPP, N=3, p3: 65%">65%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.92x | 2,914 tok/s</span><span>MAL 4.82 | AR 54.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.876" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.611" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.511" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.428" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p5: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.358" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p6: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.298" title="google/gemma-4-31B-it / DFlash / MBPP, N=7, p7: 30%">30%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.65x | 2,512 tok/s</span><span>MAL 5.38 | AR 39.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.870" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.718" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.586" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.482" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p4: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.400" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p5: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.332" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p6: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.277" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p7: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.232" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p8: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.192" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p9: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.160" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p10: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.132" title="google/gemma-4-31B-it / DFlash / MBPP, N=11, p11: 13%">13%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.40x | 2,127 tok/s</span><span>MAL 5.56 | AR 30.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.865" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.710" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.575" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.472" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.387" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p5: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.319" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.264" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p7: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.220" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.181" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p9: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.149" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p10: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.124" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p11: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.100" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p12: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.082" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p13: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.066" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p14: 7%">7%</td>
              <td class="acceptance-cell" style="--accept: 0.050" title="google/gemma-4-31B-it / DFlash / MBPP, N=15, p15: 5%">5%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dspark-gsm8k" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DSpark" data-benchmark="gsm8k">
    <h3><code>google/gemma-4-31B-it</code> / DSpark / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,631 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.62x | 2,635 tok/s</span><span>MAL 3.33 | AR 77.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.878" title="google/gemma-4-31B-it / DSpark / GSM8K, N=3, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.775" title="google/gemma-4-31B-it / DSpark / GSM8K, N=3, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.679" title="google/gemma-4-31B-it / DSpark / GSM8K, N=3, p3: 68%">68%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.82x | 2,971 tok/s</span><span>MAL 5.07 | AR 58.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.876" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.767" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.665" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.566" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.474" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p5: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.394" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.326" title="google/gemma-4-31B-it / DSpark / GSM8K, N=7, p7: 33%">33%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.52x | 2,484 tok/s</span><span>MAL 5.51 | AR 41.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.851" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.736" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.634" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p3: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.535" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p4: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.444" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p5: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.366" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.301" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.242" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p8: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.184" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p9: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.132" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p10: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.088" title="google/gemma-4-31B-it / DSpark / GSM8K, N=11, p11: 9%">9%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.32x | 2,155 tok/s</span><span>MAL 5.69 | AR 31.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.864" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.749" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.645" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.543" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p4: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.452" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p5: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.371" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.304" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.245" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p8: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.186" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p9: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.133" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p10: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.088" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p11: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.054" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p12: 5%">5%</td>
              <td class="acceptance-cell" style="--accept: 0.032" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p13: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.017" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p14: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.009" title="google/gemma-4-31B-it / DSpark / GSM8K, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dspark-math500" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DSpark" data-benchmark="math500">
    <h3><code>google/gemma-4-31B-it</code> / DSpark / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,365 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.98x | 2,703 tok/s</span><span>MAL 3.45 | AR 81.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.911" title="google/gemma-4-31B-it / DSpark / MATH500, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="google/gemma-4-31B-it / DSpark / MATH500, N=3, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.726" title="google/gemma-4-31B-it / DSpark / MATH500, N=3, p3: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.20x | 3,004 tok/s</span><span>MAL 5.30 | AR 61.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.892" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.784" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.686" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p3: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.596" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p4: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.519" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.443" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p6: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.376" title="google/gemma-4-31B-it / DSpark / MATH500, N=7, p7: 38%">38%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.91x | 2,612 tok/s</span><span>MAL 5.96 | AR 45.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.886" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.773" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.670" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.579" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p4: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.499" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p5: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.424" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p6: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.355" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p7: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.288" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p8: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.223" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.159" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p10: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.107" title="google/gemma-4-31B-it / DSpark / MATH500, N=11, p11: 11%">11%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.61x | 2,197 tok/s</span><span>MAL 6.05 | AR 33.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.883" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.769" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.667" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.574" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.495" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p5: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.420" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p6: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.349" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p7: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.282" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p8: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.218" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.156" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p10: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.104" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p11: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.065" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p12: 7%">7%</td>
              <td class="acceptance-cell" style="--accept: 0.038" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p13: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.021" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p14: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.011" title="google/gemma-4-31B-it / DSpark / MATH500, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dspark-humaneval" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DSpark" data-benchmark="humaneval">
    <h3><code>google/gemma-4-31B-it</code> / DSpark / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,228 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.34x | 1,648 tok/s</span><span>MAL 3.36 | AR 78.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.890" title="google/gemma-4-31B-it / DSpark / HumanEval, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.786" title="google/gemma-4-31B-it / DSpark / HumanEval, N=3, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.684" title="google/gemma-4-31B-it / DSpark / HumanEval, N=3, p3: 68%">68%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.98x | 2,425 tok/s</span><span>MAL 5.05 | AR 57.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.884" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.766" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.659" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.561" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.471" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p5: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.394" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.313" title="google/gemma-4-31B-it / DSpark / HumanEval, N=7, p7: 31%">31%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.73x | 2,121 tok/s</span><span>MAL 5.47 | AR 40.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.873" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.756" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.643" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p3: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.541" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p4: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.452" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p5: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.373" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.298" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.221" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.158" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p9: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.097" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p10: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.059" title="google/gemma-4-31B-it / DSpark / HumanEval, N=11, p11: 6%">6%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.47x | 1,811 tok/s</span><span>MAL 5.55 | AR 30.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.875" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.761" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.650" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.544" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p4: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.455" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p5: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.376" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p6: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.299" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.219" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.157" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p9: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.095" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p10: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.058" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p11: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.030" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p12: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.016" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p13: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.009" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.005" title="google/gemma-4-31B-it / DSpark / HumanEval, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-google-gemma-4-31b-it-dspark-mbpp" class="appendix-acceptance-panel" data-target="google/gemma-4-31B-it" data-method="DSpark" data-benchmark="mbpp">
    <h3><code>google/gemma-4-31B-it</code> / DSpark / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,519 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.84x | 2,797 tok/s</span><span>MAL 3.17 | AR 72.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.859" title="google/gemma-4-31B-it / DSpark / MBPP, N=3, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.719" title="google/gemma-4-31B-it / DSpark / MBPP, N=3, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.595" title="google/gemma-4-31B-it / DSpark / MBPP, N=3, p3: 60%">60%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.80x | 2,730 tok/s</span><span>MAL 4.41 | AR 48.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.840" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.686" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.553" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p3: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.450" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p4: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.363" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p5: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.291" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p6: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.228" title="google/gemma-4-31B-it / DSpark / MBPP, N=7, p7: 23%">23%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.50x | 2,272 tok/s</span><span>MAL 4.74 | AR 34.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.831" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.674" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p2: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.541" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p3: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.435" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p4: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.351" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p5: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.279" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p6: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.221" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p7: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.168" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p8: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.119" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p9: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.078" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p10: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.048" title="google/gemma-4-31B-it / DSpark / MBPP, N=11, p11: 5%">5%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.23x | 1,876 tok/s</span><span>MAL 4.77 | AR 25.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.831" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.673" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p2: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.539" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p3: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.434" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p4: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.349" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p5: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.277" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p6: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.216" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p7: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.163" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p8: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.117" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p9: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.077" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p10: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.046" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p11: 5%">5%</td>
              <td class="acceptance-cell" style="--accept: 0.027" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p12: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.014" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p13: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.007" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.004" title="google/gemma-4-31B-it / DSpark / MBPP, N=15, p15: 0%">0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-eagle-3-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="EAGLE-3" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3-8B</code> / EAGLE-3 / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 3,698 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.71x | 2,634 tok/s</span><span>MAL 1.86 | AR 86.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.863" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=1, p1: 86%">86%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>0.91x | 3,349 tok/s</span><span>MAL 2.57 | AR 78.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.856" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=2, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.709" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=2, p2: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>0.99x | 3,645 tok/s</span><span>MAL 3.12 | AR 70.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.851" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=3, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.697" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=3, p2: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.569" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=3, p3: 57%">57%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.10x | 4,079 tok/s</span><span>MAL 3.54 | AR 63.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.843" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=4, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.687" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=4, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.561" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=4, p3: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.449" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=4, p4: 45%">45%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.18x | 4,347 tok/s</span><span>MAL 3.86 | AR 57.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.840" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=5, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.682" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=5, p2: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.555" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=5, p3: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.441" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=5, p4: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.346" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=5, p5: 35%">35%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.17x | 4,322 tok/s</span><span>MAL 4.09 | AR 51.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.836" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=6, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.676" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=6, p2: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.547" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=6, p3: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.434" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=6, p4: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.339" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=6, p5: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.259" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=6, p6: 26%">26%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.17x | 4,327 tok/s</span><span>MAL 4.25 | AR 46.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.833" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.670" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p2: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.542" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p3: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.429" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p4: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.335" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p5: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.255" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p6: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.192" title="Qwen/Qwen3-8B / EAGLE-3 / GSM8K, N=7, p7: 19%">19%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-eagle-3-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="EAGLE-3" data-benchmark="math500">
    <h3><code>Qwen/Qwen3-8B</code> / EAGLE-3 / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 3,530 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.44x | 1,563 tok/s</span><span>MAL 1.89 | AR 89.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.890" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=1, p1: 89%">89%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>0.61x | 2,141 tok/s</span><span>MAL 2.64 | AR 82.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.882" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=2, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.762" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=2, p2: 76%">76%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>0.72x | 2,527 tok/s</span><span>MAL 3.27 | AR 75.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.877" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=3, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.753" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=3, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.638" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=3, p3: 64%">64%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>0.78x | 2,753 tok/s</span><span>MAL 3.75 | AR 68.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.869" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=4, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.739" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=4, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.622" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=4, p3: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.516" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=4, p4: 52%">52%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>0.83x | 2,935 tok/s</span><span>MAL 4.14 | AR 62.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.865" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=5, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=5, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.614" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=5, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.508" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=5, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.418" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=5, p5: 42%">42%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>0.85x | 3,010 tok/s</span><span>MAL 4.43 | AR 57.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.861" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=6, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.726" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=6, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.606" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=6, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.498" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=6, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.409" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=6, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.331" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=6, p6: 33%">33%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>0.88x | 3,105 tok/s</span><span>MAL 4.68 | AR 52.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.858" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.723" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.601" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p3: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.496" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.406" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.328" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p6: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.265" title="Qwen/Qwen3-8B / EAGLE-3 / MATH500, N=7, p7: 27%">27%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-eagle-3-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="EAGLE-3" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3-8B</code> / EAGLE-3 / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 3,226 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.61x | 1,955 tok/s</span><span>MAL 1.84 | AR 83.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.836" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=1, p1: 84%">84%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>0.86x | 2,776 tok/s</span><span>MAL 2.50 | AR 74.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.828" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=2, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.668" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=2, p2: 67%">67%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.00x | 3,238 tok/s</span><span>MAL 2.97 | AR 65.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.814" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=3, p1: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.649" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=3, p2: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.511" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=3, p3: 51%">51%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.04x | 3,346 tok/s</span><span>MAL 3.36 | AR 58.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.813" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=4, p1: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.647" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=4, p2: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.504" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=4, p3: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.393" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=4, p4: 39%">39%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.05x | 3,376 tok/s</span><span>MAL 3.59 | AR 51.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.804" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=5, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.635" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=5, p2: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.490" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=5, p3: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.377" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=5, p4: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.288" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=5, p5: 29%">29%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.04x | 3,369 tok/s</span><span>MAL 3.80 | AR 46.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.804" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=6, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.632" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=6, p2: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.485" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=6, p3: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.376" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=6, p4: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.289" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=6, p5: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.218" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=6, p6: 22%">22%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.03x | 3,337 tok/s</span><span>MAL 3.96 | AR 42.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.801" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.630" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p2: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.483" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p3: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.374" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p4: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.284" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p5: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.218" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p6: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.167" title="Qwen/Qwen3-8B / EAGLE-3 / HumanEval, N=7, p7: 17%">17%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-eagle-3-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="EAGLE-3" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3-8B</code> / EAGLE-3 / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 3,268 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.80x | 2,621 tok/s</span><span>MAL 1.81 | AR 81.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.813" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=1, p1: 81%">81%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>0.91x | 2,985 tok/s</span><span>MAL 2.43 | AR 71.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.808" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=2, p1: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.625" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=2, p2: 63%">63%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.00x | 3,254 tok/s</span><span>MAL 2.89 | AR 63.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.800" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=3, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.619" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=3, p2: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.474" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=3, p3: 47%">47%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.11x | 3,631 tok/s</span><span>MAL 3.23 | AR 55.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.794" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=4, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.615" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=4, p2: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.468" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=4, p3: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.350" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=4, p4: 35%">35%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.16x | 3,798 tok/s</span><span>MAL 3.42 | AR 48.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.789" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=5, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.604" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=5, p2: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.453" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=5, p3: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.337" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=5, p4: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.242" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=5, p5: 24%">24%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.07x | 3,513 tok/s</span><span>MAL 3.64 | AR 43.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.793" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=6, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.606" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=6, p2: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.462" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=6, p3: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.344" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=6, p4: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.251" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=6, p5: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.181" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=6, p6: 18%">18%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.06x | 3,475 tok/s</span><span>MAL 3.68 | AR 38.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.783" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p1: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.596" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p2: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.446" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p3: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.333" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p4: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.239" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p5: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.168" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p6: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.119" title="Qwen/Qwen3-8B / EAGLE-3 / MBPP, N=7, p7: 12%">12%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dflash-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3-8B</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 3,698 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.23x | 4,535 tok/s</span><span>MAL 3.23 | AR 74.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.865" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=3, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.739" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=3, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.623" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=3, p3: 62%">62%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.25x | 4,608 tok/s</span><span>MAL 4.84 | AR 54.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.860" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.725" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.608" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.514" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.439" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p5: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.375" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p6: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.321" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=7, p7: 32%">32%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.27x | 4,678 tok/s</span><span>MAL 5.51 | AR 41.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.853" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.710" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.583" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.486" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p4: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.409" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.346" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.297" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.256" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.219" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.188" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p10: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.160" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=11, p11: 16%">16%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.20x | 4,442 tok/s</span><span>MAL 6.04 | AR 33.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.730" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.598" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p3: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.499" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.418" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p5: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.352" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.300" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.258" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.221" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.191" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p10: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.164" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p11: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.140" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p12: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.118" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p13: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.100" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p14: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.083" title="Qwen/Qwen3-8B / DFlash / GSM8K, N=15, p15: 8%">8%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dflash-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DFlash" data-benchmark="math500">
    <h3><code>Qwen/Qwen3-8B</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 3,530 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>0.99x | 3,487 tok/s</span><span>MAL 3.41 | AR 80.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.899" title="Qwen/Qwen3-8B / DFlash / MATH500, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.801" title="Qwen/Qwen3-8B / DFlash / MATH500, N=3, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.710" title="Qwen/Qwen3-8B / DFlash / MATH500, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.07x | 3,794 tok/s</span><span>MAL 5.53 | AR 64.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.894" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.790" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.698" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.624" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.562" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.505" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p6: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.454" title="Qwen/Qwen3-8B / DFlash / MATH500, N=7, p7: 45%">45%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.08x | 3,828 tok/s</span><span>MAL 6.69 | AR 51.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.886" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.773" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.673" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.594" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p4: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.531" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p5: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.475" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p6: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.428" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p7: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.387" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p8: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.349" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p9: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.315" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p10: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.282" title="Qwen/Qwen3-8B / DFlash / MATH500, N=11, p11: 28%">28%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.10x | 3,868 tok/s</span><span>MAL 7.52 | AR 43.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.898" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.676" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.594" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p4: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.528" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p5: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.471" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p6: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.423" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p7: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.380" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p8: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.343" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p9: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.309" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p10: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.277" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p11: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.250" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p12: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.223" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p13: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.198" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p14: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.173" title="Qwen/Qwen3-8B / DFlash / MATH500, N=15, p15: 17%">17%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dflash-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3-8B</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 3,226 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.20x | 3,866 tok/s</span><span>MAL 3.45 | AR 81.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.814" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.727" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=3, p3: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.27x | 4,103 tok/s</span><span>MAL 5.27 | AR 61.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.883" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.765" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.664" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.581" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p4: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.515" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.458" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p6: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.409" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=7, p7: 41%">41%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.27x | 4,081 tok/s</span><span>MAL 5.68 | AR 42.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.854" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.710" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.589" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.495" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.420" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p5: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.364" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p6: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.320" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p7: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.279" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p8: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.244" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p9: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.215" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p10: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.190" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=11, p11: 19%">19%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.20x | 3,877 tok/s</span><span>MAL 6.15 | AR 34.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.866" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.719" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.592" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.487" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p4: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.412" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.353" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.307" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p7: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.268" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p8: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.236" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p9: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.205" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p10: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.181" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p11: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.159" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p12: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.140" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p13: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.123" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p14: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.104" title="Qwen/Qwen3-8B / DFlash / HumanEval, N=15, p15: 10%">10%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dflash-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3-8B</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 3,268 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.21x | 3,952 tok/s</span><span>MAL 3.44 | AR 81.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="Qwen/Qwen3-8B / DFlash / MBPP, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.812" title="Qwen/Qwen3-8B / DFlash / MBPP, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.727" title="Qwen/Qwen3-8B / DFlash / MBPP, N=3, p3: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.22x | 3,982 tok/s</span><span>MAL 4.79 | AR 54.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.856" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.713" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.597" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p3: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.502" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.429" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p5: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.372" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.322" title="Qwen/Qwen3-8B / DFlash / MBPP, N=7, p7: 32%">32%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.22x | 3,974 tok/s</span><span>MAL 5.23 | AR 38.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.840" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.692" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.561" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p3: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.458" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p4: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.379" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p5: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.320" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.267" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p7: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.225" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p8: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.190" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p9: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.162" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p10: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.137" title="Qwen/Qwen3-8B / DFlash / MBPP, N=11, p11: 14%">14%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.13x | 3,695 tok/s</span><span>MAL 5.59 | AR 30.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.862" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.707" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.572" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.465" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.379" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p5: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.315" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.259" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p7: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.216" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.183" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p9: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.154" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p10: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.131" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p11: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.110" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p12: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.093" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p13: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.078" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p14: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.064" title="Qwen/Qwen3-8B / DFlash / MBPP, N=15, p15: 6%">6%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dspark-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DSpark" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3-8B</code> / DSpark / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 3,698 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.08x | 4,001 tok/s</span><span>MAL 3.68 | AR 89.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.949" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.894" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=3, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.837" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=3, p3: 84%">84%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.63x | 6,032 tok/s</span><span>MAL 6.49 | AR 78.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.891" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.833" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p3: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.779" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p4: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.729" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p5: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.679" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p6: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.631" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=7, p7: 63%">63%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.58x | 5,841 tok/s</span><span>MAL 7.63 | AR 60.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.939" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.801" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.733" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p4: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.671" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p5: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.614" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p6: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.555" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p7: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.483" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p8: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.403" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p9: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.319" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p10: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.238" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=11, p11: 24%">24%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.31x | 4,857 tok/s</span><span>MAL 7.17 | AR 41.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.937" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.865" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.785" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.702" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p4: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.621" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p5: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.543" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p6: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.463" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p7: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.380" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p8: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.297" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p9: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.218" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p10: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.150" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p11: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.097" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p12: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.060" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p13: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.035" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p14: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.020" title="Qwen/Qwen3-8B / DSpark / GSM8K, N=15, p15: 2%">2%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dspark-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DSpark" data-benchmark="math500">
    <h3><code>Qwen/Qwen3-8B</code> / DSpark / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 3,530 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>0.73x | 2,589 tok/s</span><span>MAL 3.67 | AR 88.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="Qwen/Qwen3-8B / DSpark / MATH500, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.890" title="Qwen/Qwen3-8B / DSpark / MATH500, N=3, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.830" title="Qwen/Qwen3-8B / DSpark / MATH500, N=3, p3: 83%">83%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.15x | 4,048 tok/s</span><span>MAL 6.39 | AR 77.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.944" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.882" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.821" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p3: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.765" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p4: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.713" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p5: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.660" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p6: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.610" title="Qwen/Qwen3-8B / DSpark / MATH500, N=7, p7: 61%">61%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.12x | 3,937 tok/s</span><span>MAL 7.18 | AR 56.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.930" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.857" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.781" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.707" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p4: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.640" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p5: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.572" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p6: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.500" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p7: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.419" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p8: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.335" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p9: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.254" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p10: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.184" title="Qwen/Qwen3-8B / DSpark / MATH500, N=11, p11: 18%">18%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>0.96x | 3,376 tok/s</span><span>MAL 6.83 | AR 38.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.932" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.858" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.773" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.687" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.605" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p5: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.515" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p6: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.423" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p7: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.332" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p8: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.247" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p9: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.173" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p10: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.117" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p11: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.075" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p12: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.046" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p13: 5%">5%</td>
              <td class="acceptance-cell" style="--accept: 0.027" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p14: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.015" title="Qwen/Qwen3-8B / DSpark / MATH500, N=15, p15: 2%">2%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dspark-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DSpark" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3-8B</code> / DSpark / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 3,226 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>0.96x | 3,090 tok/s</span><span>MAL 3.53 | AR 84.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.924" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.846" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=3, p2: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.762" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=3, p3: 76%">76%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.48x | 4,769 tok/s</span><span>MAL 5.87 | AR 69.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.921" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.839" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.755" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p3: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.685" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.618" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p5: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.556" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p6: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.500" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=7, p7: 50%">50%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.32x | 4,271 tok/s</span><span>MAL 6.28 | AR 48.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.912" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.815" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.711" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p3: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.622" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.542" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p5: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.460" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p6: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.390" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p7: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.311" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p8: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.239" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p9: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.167" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p10: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.111" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=11, p11: 11%">11%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.04x | 3,357 tok/s</span><span>MAL 5.81 | AR 32.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.911" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.703" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.597" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p4: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.496" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p5: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.394" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.302" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.216" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.147" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p9: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.095" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p10: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.060" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p11: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.034" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p12: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.019" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p13: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.010" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.006" title="Qwen/Qwen3-8B / DSpark / HumanEval, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-8b-dspark-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3-8B" data-method="DSpark" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3-8B</code> / DSpark / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 3,268 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.20x | 3,919 tok/s</span><span>MAL 3.42 | AR 80.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="Qwen/Qwen3-8B / DSpark / MBPP, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.805" title="Qwen/Qwen3-8B / DSpark / MBPP, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.708" title="Qwen/Qwen3-8B / DSpark / MBPP, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.51x | 4,936 tok/s</span><span>MAL 5.56 | AR 65.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.912" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.815" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.718" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.634" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p4: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.560" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.492" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p6: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.430" title="Qwen/Qwen3-8B / DSpark / MBPP, N=7, p7: 43%">43%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.39x | 4,536 tok/s</span><span>MAL 5.90 | AR 44.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.896" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.791" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.678" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.582" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p4: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.495" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p5: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.417" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p6: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.347" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p7: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.270" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p8: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.201" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p9: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.135" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p10: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.091" title="Qwen/Qwen3-8B / DSpark / MBPP, N=11, p11: 9%">9%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.16x | 3,779 tok/s</span><span>MAL 5.40 | AR 29.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.895" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.782" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.663" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.549" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p4: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.444" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p5: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.350" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.260" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p7: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.182" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p8: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.118" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p9: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.072" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p10: 7%">7%</td>
              <td class="acceptance-cell" style="--accept: 0.041" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p11: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.022" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p12: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.012" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p13: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.006" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.003" title="Qwen/Qwen3-8B / DSpark / MBPP, N=15, p15: 0%">0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-native-mtp-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="Native MTP" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.5-27B</code> / Native MTP / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,555 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.11x | 1,724 tok/s</span><span>MAL 1.97 | AR 96.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.965" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=1, p1: 97%">97%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.37x | 2,133 tok/s</span><span>MAL 2.86 | AR 92.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.962" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.893" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=2, p2: 89%">89%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.50x | 2,337 tok/s</span><span>MAL 3.65 | AR 88.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.958" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=3, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.887" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=3, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.801" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=3, p3: 80%">80%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.62x | 2,522 tok/s</span><span>MAL 4.32 | AR 83.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.955" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=4, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.879" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=4, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.790" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=4, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.698" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=4, p4: 70%">70%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.63x | 2,537 tok/s</span><span>MAL 4.89 | AR 77.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=5, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=5, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.781" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=5, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.689" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=5, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.600" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=5, p5: 60%">60%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.66x | 2,575 tok/s</span><span>MAL 5.37 | AR 72.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=6, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.865" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=6, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.773" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=6, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.680" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=6, p4: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.593" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=6, p5: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.511" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=6, p6: 51%">51%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.56x | 2,423 tok/s</span><span>MAL 5.77 | AR 68.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.862" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.768" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.676" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p4: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.588" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p5: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.504" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.431" title="Qwen/Qwen3.5-27B / Native MTP / GSM8K, N=7, p7: 43%">43%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-native-mtp-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="Native MTP" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.5-27B</code> / Native MTP / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,500 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.10x | 1,644 tok/s</span><span>MAL 1.97 | AR 96.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.965" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=1, p1: 97%">97%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.39x | 2,085 tok/s</span><span>MAL 2.86 | AR 92.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.962" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.894" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=2, p2: 89%">89%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.56x | 2,345 tok/s</span><span>MAL 3.65 | AR 88.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.959" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=3, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.887" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=3, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.799" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=3, p3: 80%">80%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.66x | 2,489 tok/s</span><span>MAL 4.32 | AR 83.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.955" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=4, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.880" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=4, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.790" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=4, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.696" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=4, p4: 70%">70%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.71x | 2,564 tok/s</span><span>MAL 4.89 | AR 77.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.953" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=5, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.874" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=5, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.782" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=5, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.686" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=5, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.595" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=5, p5: 60%">60%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.70x | 2,549 tok/s</span><span>MAL 5.35 | AR 72.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.950" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=6, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.867" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=6, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.772" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=6, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.674" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=6, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.584" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=6, p5: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.499" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=6, p6: 50%">50%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.55x | 2,325 tok/s</span><span>MAL 5.73 | AR 67.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.947" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.862" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.765" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.668" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.577" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p5: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.492" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p6: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.416" title="Qwen/Qwen3.5-27B / Native MTP / MATH500, N=7, p7: 42%">42%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-native-mtp-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="Native MTP" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.5-27B</code> / Native MTP / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,256 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.15x | 1,439 tok/s</span><span>MAL 1.97 | AR 96.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.965" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=1, p1: 97%">97%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.20x | 1,507 tok/s</span><span>MAL 2.86 | AR 92.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.959" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.891" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=2, p2: 89%">89%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.53x | 1,917 tok/s</span><span>MAL 3.63 | AR 87.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.882" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=3, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.799" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=3, p3: 80%">80%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.63x | 2,044 tok/s</span><span>MAL 4.31 | AR 82.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.874" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=4, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.786" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=4, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.698" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=4, p4: 70%">70%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.55x | 1,953 tok/s</span><span>MAL 4.89 | AR 77.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=5, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.868" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=5, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.779" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=5, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.689" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=5, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.609" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=5, p5: 61%">61%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.46x | 1,836 tok/s</span><span>MAL 5.39 | AR 73.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=6, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.862" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=6, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.771" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=6, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.685" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=6, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.603" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=6, p5: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.520" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=6, p6: 52%">52%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.41x | 1,766 tok/s</span><span>MAL 5.73 | AR 67.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.941" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.852" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p2: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.757" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p3: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.668" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.582" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p5: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.503" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.431" title="Qwen/Qwen3.5-27B / Native MTP / HumanEval, N=7, p7: 43%">43%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-native-mtp-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="Native MTP" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.5-27B</code> / Native MTP / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,418 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.10x | 1,562 tok/s</span><span>MAL 1.94 | AR 94.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.944" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=1, p1: 94%">94%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.39x | 1,974 tok/s</span><span>MAL 2.76 | AR 88.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.934" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=2, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.830" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=2, p2: 83%">83%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.49x | 2,117 tok/s</span><span>MAL 3.46 | AR 81.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.929" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=3, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.820" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=3, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.706" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.60x | 2,268 tok/s</span><span>MAL 4.03 | AR 75.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.927" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=4, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.812" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=4, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.696" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=4, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.591" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=4, p4: 59%">59%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.59x | 2,254 tok/s</span><span>MAL 4.42 | AR 68.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.917" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=5, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.793" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=5, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.672" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=5, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.562" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=5, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.472" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=5, p5: 47%">47%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.57x | 2,233 tok/s</span><span>MAL 4.77 | AR 62.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.915" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=6, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.786" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=6, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.665" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=6, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.556" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=6, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.466" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=6, p5: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.384" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=6, p6: 38%">38%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.41x | 1,995 tok/s</span><span>MAL 5.02 | AR 57.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.911" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.779" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.654" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.546" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p4: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.456" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p5: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.375" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p6: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.304" title="Qwen/Qwen3.5-27B / Native MTP / MBPP, N=7, p7: 30%">30%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-dflash-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.5-27B</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,555 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.45x | 2,247 tok/s</span><span>MAL 3.57 | AR 85.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.945" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.860" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=3, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.763" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=3, p3: 76%">76%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.54x | 2,397 tok/s</span><span>MAL 5.64 | AR 66.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.928" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.830" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.735" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p3: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.648" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p4: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.570" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p5: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.500" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.432" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=7, p7: 43%">43%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.50x | 2,335 tok/s</span><span>MAL 6.63 | AR 51.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.921" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.811" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.708" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p3: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.616" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.537" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p5: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.468" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p6: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.408" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p7: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.356" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p8: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.310" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p9: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.269" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p10: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.229" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=11, p11: 23%">23%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.32x | 2,054 tok/s</span><span>MAL 7.11 | AR 40.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.920" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.807" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.701" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.606" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p4: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.524" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.453" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p6: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.393" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p7: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.341" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p8: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.296" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p9: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.256" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p10: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.220" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p11: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.187" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p12: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.159" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p13: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.134" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p14: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.110" title="Qwen/Qwen3.5-27B / DFlash / GSM8K, N=15, p15: 11%">11%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-dflash-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="DFlash" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.5-27B</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,500 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.51x | 2,259 tok/s</span><span>MAL 3.60 | AR 86.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.949" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=3, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.782" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=3, p3: 78%">78%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.61x | 2,421 tok/s</span><span>MAL 5.80 | AR 68.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.933" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.841" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.752" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p3: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.671" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.600" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p5: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.534" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p6: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.471" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=7, p7: 47%">47%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.65x | 2,482 tok/s</span><span>MAL 6.98 | AR 54.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.925" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.824" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.729" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p3: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.644" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.570" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p5: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.504" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.448" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p7: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.398" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p8: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.354" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p9: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.310" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p10: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.271" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=11, p11: 27%">27%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.47x | 2,208 tok/s</span><span>MAL 7.56 | AR 43.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.926" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.819" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.720" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.632" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p4: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.555" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.489" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p6: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.432" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p7: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.381" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p8: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.336" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p9: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.296" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p10: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.258" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p11: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.224" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p12: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.193" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p13: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.164" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p14: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.135" title="Qwen/Qwen3.5-27B / DFlash / MATH500, N=15, p15: 14%">14%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-dflash-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.5-27B</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,256 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.46x | 1,829 tok/s</span><span>MAL 3.61 | AR 87.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.947" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.874" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=3, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.789" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=3, p3: 79%">79%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.22x | 1,535 tok/s</span><span>MAL 5.82 | AR 68.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.927" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.836" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.746" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p3: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.671" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.606" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p5: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.547" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p6: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.488" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=7, p7: 49%">49%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.40x | 1,757 tok/s</span><span>MAL 6.88 | AR 53.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.700" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.620" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.556" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.497" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.447" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p7: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.399" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p8: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.359" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p9: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.319" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p10: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.280" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=11, p11: 28%">28%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.40x | 1,761 tok/s</span><span>MAL 7.57 | AR 43.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.915" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.805" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.704" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.620" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.550" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p5: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.487" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p6: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.432" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p7: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.384" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p8: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.344" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p9: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.305" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p10: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.268" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p11: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.235" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p12: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.205" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p13: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.175" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p14: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.144" title="Qwen/Qwen3.5-27B / DFlash / HumanEval, N=15, p15: 14%">14%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-27b-dflash-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-27B" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.5-27B</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,418 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.44x | 2,042 tok/s</span><span>MAL 3.37 | AR 79.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.791" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=3, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.672" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=3, p3: 67%">67%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.38x | 1,963 tok/s</span><span>MAL 4.91 | AR 55.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.880" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.735" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.611" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.518" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p4: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.447" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p5: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.386" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.332" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=7, p7: 33%">33%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.25x | 1,770 tok/s</span><span>MAL 5.51 | AR 41.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.867" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.702" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p2: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.574" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.476" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p4: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.406" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.349" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.299" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.259" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.225" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p9: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.194" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p10: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.164" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=11, p11: 16%">16%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.06x | 1,504 tok/s</span><span>MAL 5.99 | AR 33.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.716" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.583" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.484" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p4: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.410" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.352" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.300" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.259" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.224" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.193" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p10: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.164" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p11: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.140" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p12: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.118" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p13: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.098" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p14: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.079" title="Qwen/Qwen3.5-27B / DFlash / MBPP, N=15, p15: 8%">8%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-native-mtp-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="Native MTP" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / Native MTP / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,494 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.02x | 1,528 tok/s</span><span>MAL 1.96 | AR 96.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.960" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=1, p1: 96%">96%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.47x | 2,202 tok/s</span><span>MAL 2.85 | AR 92.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.960" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.893" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=2, p2: 89%">89%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.64x | 2,445 tok/s</span><span>MAL 3.64 | AR 87.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.883" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=3, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.802" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=3, p3: 80%">80%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.81x | 2,697 tok/s</span><span>MAL 4.31 | AR 82.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=4, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.790" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=4, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.705" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=4, p4: 71%">71%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.98x | 2,958 tok/s</span><span>MAL 4.93 | AR 78.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.950" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=5, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=5, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.787" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=5, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.703" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=5, p4: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.620" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=5, p5: 62%">62%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.98x | 2,953 tok/s</span><span>MAL 5.42 | AR 73.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.947" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=6, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.866" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=6, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=6, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.691" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=6, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.605" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=6, p5: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.529" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=6, p6: 53%">53%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.08x | 3,107 tok/s</span><span>MAL 5.85 | AR 69.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.945" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.863" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.774" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.686" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p4: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.603" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p5: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.526" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p6: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.457" title="Qwen/Qwen3.5-122B-A10B / Native MTP / GSM8K, N=7, p7: 46%">46%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-native-mtp-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="Native MTP" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / Native MTP / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,446 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.06x | 1,529 tok/s</span><span>MAL 1.97 | AR 96.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.965" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=1, p1: 97%">97%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.58x | 2,280 tok/s</span><span>MAL 2.86 | AR 93.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.962" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.899" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=2, p2: 90%">90%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.82x | 2,625 tok/s</span><span>MAL 3.67 | AR 89.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.960" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=3, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.895" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=3, p2: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=3, p3: 82%">82%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.97x | 2,843 tok/s</span><span>MAL 4.37 | AR 84.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.956" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=4, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.886" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=4, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.806" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=4, p3: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.723" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=4, p4: 72%">72%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>2.14x | 3,088 tok/s</span><span>MAL 4.98 | AR 79.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=5, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.882" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=5, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.799" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=5, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.715" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=5, p4: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.632" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=5, p5: 63%">63%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>2.13x | 3,078 tok/s</span><span>MAL 5.49 | AR 74.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=6, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.876" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=6, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.791" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=6, p3: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.706" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=6, p4: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.623" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=6, p5: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.545" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=6, p6: 55%">55%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.20x | 3,183 tok/s</span><span>MAL 5.91 | AR 70.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.948" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.869" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.782" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.695" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p4: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.612" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p5: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.535" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p6: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.464" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MATH500, N=7, p7: 46%">46%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-native-mtp-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="Native MTP" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / Native MTP / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,105 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.02x | 1,131 tok/s</span><span>MAL 1.97 | AR 96.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.966" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=1, p1: 97%">97%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.46x | 1,610 tok/s</span><span>MAL 2.86 | AR 93.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.963" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.900" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=2, p2: 90%">90%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.69x | 1,868 tok/s</span><span>MAL 3.68 | AR 89.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.959" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=3, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.897" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=3, p2: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.824" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=3, p3: 82%">82%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.69x | 1,869 tok/s</span><span>MAL 4.38 | AR 84.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.886" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=4, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.811" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=4, p3: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=4, p4: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.83x | 2,017 tok/s</span><span>MAL 5.03 | AR 80.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.950" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=5, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.880" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=5, p2: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.803" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=5, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.731" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=5, p4: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.664" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=5, p5: 66%">66%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.83x | 2,021 tok/s</span><span>MAL 5.65 | AR 77.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.954" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=6, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.887" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=6, p2: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.814" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=6, p3: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.739" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=6, p4: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.665" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=6, p5: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.595" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=6, p6: 60%">60%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.85x | 2,044 tok/s</span><span>MAL 6.07 | AR 72.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.945" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.872" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p3: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.722" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p4: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.648" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p5: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.575" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p6: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.511" title="Qwen/Qwen3.5-122B-A10B / Native MTP / HumanEval, N=7, p7: 51%">51%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-native-mtp-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="Native MTP" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / Native MTP / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,459 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.99x | 1,447 tok/s</span><span>MAL 1.95 | AR 95.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.950" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=1, p1: 95%">95%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.43x | 2,092 tok/s</span><span>MAL 2.86 | AR 92.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.958" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.898" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=2, p2: 90%">90%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.60x | 2,336 tok/s</span><span>MAL 3.56 | AR 85.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.942" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.857" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=3, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.765" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=3, p3: 77%">77%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.66x | 2,422 tok/s</span><span>MAL 4.19 | AR 79.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.934" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=4, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.846" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=4, p2: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.748" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=4, p3: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.661" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=4, p4: 66%">66%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.75x | 2,558 tok/s</span><span>MAL 4.68 | AR 73.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.926" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=5, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.829" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=5, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.728" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=5, p3: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.636" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=5, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.558" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=5, p5: 56%">56%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.84x | 2,678 tok/s</span><span>MAL 5.16 | AR 69.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.926" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=6, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.829" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=6, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.729" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=6, p3: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.639" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=6, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.557" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=6, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.485" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=6, p6: 49%">49%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.88x | 2,747 tok/s</span><span>MAL 5.56 | AR 65.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.925" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.829" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.727" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p3: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.635" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.554" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p5: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.478" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p6: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.411" title="Qwen/Qwen3.5-122B-A10B / Native MTP / MBPP, N=7, p7: 41%">41%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-dflash-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,494 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.41x | 2,111 tok/s</span><span>MAL 3.26 | AR 75.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.875" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=3, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.753" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=3, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.634" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=3, p3: 63%">63%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.58x | 2,356 tok/s</span><span>MAL 4.19 | AR 45.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.812" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p1: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.651" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p2: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.518" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p3: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.414" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p4: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.330" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p5: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.262" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p6: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.206" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=7, p7: 21%">21%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.38x | 2,066 tok/s</span><span>MAL 4.17 | AR 28.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.784" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p1: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.611" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p2: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.469" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p3: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.359" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p4: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.273" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p5: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.206" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p6: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.153" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p7: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.114" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p8: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.086" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p9: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.064" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p10: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.047" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=11, p11: 5%">5%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.01x | 1,508 tok/s</span><span>MAL 3.81 | AR 18.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.770" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p1: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.578" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p2: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.427" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p3: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.312" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p4: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.226" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p5: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.161" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p6: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.111" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p7: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.077" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p8: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.053" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p9: 5%">5%</td>
              <td class="acceptance-cell" style="--accept: 0.035" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p10: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.023" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p11: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.015" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p12: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.009" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p13: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.006" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.004" title="Qwen/Qwen3.5-122B-A10B / DFlash / GSM8K, N=15, p15: 0%">0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-dflash-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="DFlash" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,446 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.62x | 2,336 tok/s</span><span>MAL 3.34 | AR 78.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.890" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=3, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.670" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=3, p3: 67%">67%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.78x | 2,572 tok/s</span><span>MAL 4.45 | AR 49.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.838" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.684" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p2: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.556" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p3: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.453" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p4: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.369" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p5: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.302" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p6: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.246" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=7, p7: 25%">25%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.64x | 2,367 tok/s</span><span>MAL 4.50 | AR 31.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.820" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p1: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.650" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p2: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.509" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p3: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.397" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.308" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p5: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.240" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p6: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.186" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p7: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.143" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p8: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.108" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p9: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.081" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p10: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.060" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=11, p11: 6%">6%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.25x | 1,805 tok/s</span><span>MAL 4.01 | AR 20.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.795" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.604" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p2: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.450" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p3: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.334" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p4: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.245" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p5: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.179" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p6: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.129" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p7: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.091" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p8: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.064" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p9: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.043" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p10: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.029" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p11: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.019" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p12: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.012" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p13: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.007" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.005" title="Qwen/Qwen3.5-122B-A10B / DFlash / MATH500, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-dflash-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,105 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.40x | 1,551 tok/s</span><span>MAL 3.40 | AR 79.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.901" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=3, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.699" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=3, p3: 70%">70%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.66x | 1,838 tok/s</span><span>MAL 4.53 | AR 50.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.842" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.692" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.566" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.467" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.384" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p5: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.320" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.263" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=7, p7: 26%">26%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.20x | 1,331 tok/s</span><span>MAL 4.56 | AR 32.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.820" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p1: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.641" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p2: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.505" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p3: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.396" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.310" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p5: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.248" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p6: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.198" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p7: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.155" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p8: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.122" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p9: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.094" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p10: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.072" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=11, p11: 7%">7%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>0.94x | 1,042 tok/s</span><span>MAL 4.05 | AR 20.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.790" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p1: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.596" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p2: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.450" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p3: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.338" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p4: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.250" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p5: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.187" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p6: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.137" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p7: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.099" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p8: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.070" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p9: 7%">7%</td>
              <td class="acceptance-cell" style="--accept: 0.049" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p10: 5%">5%</td>
              <td class="acceptance-cell" style="--accept: 0.033" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p11: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.023" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p12: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.015" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p13: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.009" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.005" title="Qwen/Qwen3.5-122B-A10B / DFlash / HumanEval, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-5-122b-a10b-dflash-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.5-122B-A10B" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.5-122B-A10B</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,459 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.38x | 2,019 tok/s</span><span>MAL 3.29 | AR 76.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.881" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=3, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.757" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=3, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.649" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=3, p3: 65%">65%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.05x | 1,529 tok/s</span><span>MAL 4.12 | AR 44.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.797" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.618" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p2: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.490" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p3: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.398" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.328" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p5: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.271" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p6: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.221" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=7, p7: 22%">22%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.34x | 1,958 tok/s</span><span>MAL 4.21 | AR 29.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.795" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p1: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.592" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p2: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.452" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p3: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.347" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p4: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.270" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p5: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.212" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p6: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.166" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p7: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.129" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p8: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.101" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p9: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.080" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p10: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.060" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=11, p11: 6%">6%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>0.95x | 1,386 tok/s</span><span>MAL 3.68 | AR 17.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.750" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p1: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.527" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p2: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.383" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p3: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.281" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p4: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.206" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p5: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.154" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p6: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.113" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p7: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.085" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p8: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.060" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p9: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.043" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p10: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.029" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p11: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.020" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p12: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.014" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p13: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.009" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.005" title="Qwen/Qwen3.5-122B-A10B / DFlash / MBPP, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-native-mtp-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="Native MTP" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.6-27B</code> / Native MTP / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,521 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.20x | 1,830 tok/s</span><span>MAL 1.95 | AR 94.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.945" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=1, p1: 95%">95%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.45x | 2,212 tok/s</span><span>MAL 2.79 | AR 89.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.941" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=2, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.853" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=2, p2: 85%">85%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.61x | 2,441 tok/s</span><span>MAL 3.53 | AR 84.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.935" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.843" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=3, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.748" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=3, p3: 75%">75%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.69x | 2,570 tok/s</span><span>MAL 4.15 | AR 78.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.929" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=4, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.834" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=4, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.738" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=4, p3: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.646" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=4, p4: 65%">65%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.72x | 2,609 tok/s</span><span>MAL 4.66 | AR 73.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.925" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=5, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.824" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=5, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.727" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=5, p3: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.637" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=5, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.553" title="Qwen/Qwen3.6-27B / Native MTP / GSM8K, N=5, p5: 55%">55%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-native-mtp-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="Native MTP" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.6-27B</code> / Native MTP / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,514 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.20x | 1,820 tok/s</span><span>MAL 1.96 | AR 95.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.959" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=1, p1: 96%">96%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.48x | 2,235 tok/s</span><span>MAL 2.84 | AR 91.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.955" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=2, p1: 96%">96%</td>
              <td class="acceptance-cell" style="--accept: 0.883" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=2, p2: 88%">88%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.64x | 2,488 tok/s</span><span>MAL 3.61 | AR 87.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.873" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=3, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.789" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=3, p3: 79%">79%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.75x | 2,647 tok/s</span><span>MAL 4.28 | AR 82.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.946" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=4, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.866" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=4, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.779" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=4, p3: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.692" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=4, p4: 69%">69%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.78x | 2,701 tok/s</span><span>MAL 4.83 | AR 76.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.942" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=5, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.856" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=5, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.766" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=5, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.678" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=5, p4: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.593" title="Qwen/Qwen3.6-27B / Native MTP / MATH500, N=5, p5: 59%">59%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-native-mtp-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="Native MTP" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.6-27B</code> / Native MTP / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,481 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.19x | 1,756 tok/s</span><span>MAL 1.93 | AR 92.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.929" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=1, p1: 93%">93%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.42x | 2,101 tok/s</span><span>MAL 2.73 | AR 86.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.922" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=2, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.810" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=2, p2: 81%">81%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.53x | 2,270 tok/s</span><span>MAL 3.40 | AR 80.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.916" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.802" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=3, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.687" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=3, p3: 69%">69%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.60x | 2,373 tok/s</span><span>MAL 3.94 | AR 73.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.910" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=4, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.789" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=4, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.672" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=4, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.571" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=4, p4: 57%">57%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.60x | 2,365 tok/s</span><span>MAL 4.36 | AR 67.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.904" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=5, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=5, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.656" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=5, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.555" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=5, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.469" title="Qwen/Qwen3.6-27B / Native MTP / HumanEval, N=5, p5: 47%">47%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-native-mtp-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="Native MTP" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.6-27B</code> / Native MTP / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,495 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.22x | 1,827 tok/s</span><span>MAL 1.92 | AR 91.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.917" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=1, p1: 92%">92%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.44x | 2,156 tok/s</span><span>MAL 2.69 | AR 84.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.910" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=2, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.782" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=2, p2: 78%">78%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.57x | 2,341 tok/s</span><span>MAL 3.31 | AR 77.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.902" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.770" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=3, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.643" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=3, p3: 64%">64%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.61x | 2,411 tok/s</span><span>MAL 3.80 | AR 70.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.894" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=4, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.759" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=4, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.630" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=4, p3: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.518" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=4, p4: 52%">52%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.60x | 2,389 tok/s</span><span>MAL 4.16 | AR 63.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.887" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=5, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.744" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=5, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.615" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=5, p3: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.504" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=5, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.413" title="Qwen/Qwen3.6-27B / Native MTP / MBPP, N=5, p5: 41%">41%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-dflash-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.6-27B</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 1,521 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.39x | 2,112 tok/s</span><span>MAL 3.48 | AR 82.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.929" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=3, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.828" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=3, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.720" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=3, p3: 72%">72%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.43x | 2,176 tok/s</span><span>MAL 5.34 | AR 62.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.911" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.691" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p3: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.599" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p4: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.517" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.444" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p6: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.379" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=7, p7: 38%">38%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.42x | 2,160 tok/s</span><span>MAL 6.18 | AR 47.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.903" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.781" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.668" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.574" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.493" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p5: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.421" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p6: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.360" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p7: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.308" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p8: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.263" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p9: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.224" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p10: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.189" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=11, p11: 19%">19%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.24x | 1,883 tok/s</span><span>MAL 6.52 | AR 36.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.899" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.772" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.657" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.561" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.478" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p5: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.405" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p6: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.344" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p7: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.293" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p8: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.249" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p9: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.211" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p10: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.179" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p11: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.151" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p12: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.127" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p13: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.105" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p14: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.087" title="Qwen/Qwen3.6-27B / DFlash / GSM8K, N=15, p15: 9%">9%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-dflash-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="DFlash" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.6-27B</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 1,514 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.44x | 2,185 tok/s</span><span>MAL 3.58 | AR 85.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.943" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.862" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=3, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.773" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=3, p3: 77%">77%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.54x | 2,339 tok/s</span><span>MAL 5.73 | AR 67.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.927" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.830" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.740" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p3: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.660" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p4: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.589" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p5: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.525" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p6: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.463" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=7, p7: 46%">46%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.59x | 2,411 tok/s</span><span>MAL 6.86 | AR 53.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.919" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.813" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.716" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.632" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p4: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.558" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p5: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.494" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p6: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.436" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p7: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.386" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p8: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.341" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p9: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.300" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p10: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.263" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=11, p11: 26%">26%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.41x | 2,136 tok/s</span><span>MAL 7.37 | AR 42.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.913" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.800" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.699" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.614" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p4: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.538" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p5: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.474" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p6: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.417" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p7: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.367" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p8: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.322" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p9: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.283" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p10: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.248" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p11: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.216" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p12: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.187" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p13: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.161" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p14: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.137" title="Qwen/Qwen3.6-27B / DFlash / MATH500, N=15, p15: 14%">14%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-dflash-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.6-27B</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 1,481 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.40x | 2,067 tok/s</span><span>MAL 3.44 | AR 81.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.923" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=3, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.703" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=3, p3: 70%">70%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.40x | 2,070 tok/s</span><span>MAL 5.19 | AR 59.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.780" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.666" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.568" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.488" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p5: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.420" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p6: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.363" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=7, p7: 36%">36%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.39x | 2,061 tok/s</span><span>MAL 5.96 | AR 45.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.896" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.762" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.641" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p3: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.537" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p4: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.453" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p5: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.386" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.332" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p7: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.287" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p8: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.250" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p9: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.220" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p10: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.195" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=11, p11: 20%">20%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.21x | 1,793 tok/s</span><span>MAL 6.27 | AR 35.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.889" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.750" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.624" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p3: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.520" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p4: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.434" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p5: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.364" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p6: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.310" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p7: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.266" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p8: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.230" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p9: 23%">23%</td>
              <td class="acceptance-cell" style="--accept: 0.200" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p10: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.175" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p11: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.154" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p12: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.136" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p13: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.119" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p14: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.104" title="Qwen/Qwen3.6-27B / DFlash / HumanEval, N=15, p15: 10%">10%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-27b-dflash-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-27B" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.6-27B</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 1,495 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.38x | 2,069 tok/s</span><span>MAL 3.33 | AR 77.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.906" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.777" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=3, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.649" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=3, p3: 65%">65%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.37x | 2,047 tok/s</span><span>MAL 4.81 | AR 54.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.886" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.740" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.612" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.508" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.423" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p5: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.351" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.291" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=7, p7: 29%">29%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.29x | 1,925 tok/s</span><span>MAL 5.37 | AR 39.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.879" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.726" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.593" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.484" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p4: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.396" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p5: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.324" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.269" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p7: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.224" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.187" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p9: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.157" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p10: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.131" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=11, p11: 13%">13%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.11x | 1,658 tok/s</span><span>MAL 5.57 | AR 30.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.874" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.716" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.580" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.471" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.384" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p5: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.313" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p6: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.257" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p7: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.211" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p8: 21%">21%</td>
              <td class="acceptance-cell" style="--accept: 0.176" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p9: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.147" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p10: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.122" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p11: 12%">12%</td>
              <td class="acceptance-cell" style="--accept: 0.102" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p12: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.086" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p13: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.071" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p14: 7%">7%</td>
              <td class="acceptance-cell" style="--accept: 0.058" title="Qwen/Qwen3.6-27B / DFlash / MBPP, N=15, p15: 6%">6%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-native-mtp-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="Native MTP" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / Native MTP / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 2,275 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.89x | 2,023 tok/s</span><span>MAL 1.94 | AR 93.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.937" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=1, p1: 94%">94%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.12x | 2,544 tok/s</span><span>MAL 2.77 | AR 88.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.933" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=2, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.837" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=2, p2: 84%">84%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.27x | 2,894 tok/s</span><span>MAL 3.49 | AR 82.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.926" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=3, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.828" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=3, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.731" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=3, p3: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.25x | 2,854 tok/s</span><span>MAL 4.07 | AR 76.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.919" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=4, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=4, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.715" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=4, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.624" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=4, p4: 62%">62%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.31x | 2,976 tok/s</span><span>MAL 4.57 | AR 71.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.915" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=5, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.807" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=5, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.705" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=5, p3: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.613" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=5, p4: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.529" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=5, p5: 53%">53%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.43x | 3,253 tok/s</span><span>MAL 4.97 | AR 66.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.909" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=6, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=6, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.696" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=6, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.603" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=6, p4: 60%">60%</td>
              <td class="acceptance-cell" style="--accept: 0.518" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=6, p5: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.443" title="Qwen/Qwen3.6-35B-A3B / Native MTP / GSM8K, N=6, p6: 44%">44%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-native-mtp-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="Native MTP" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / Native MTP / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 2,235 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.88x | 1,973 tok/s</span><span>MAL 1.95 | AR 95.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.955" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=1, p1: 96%">96%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.13x | 2,515 tok/s</span><span>MAL 2.83 | AR 91.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.951" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=2, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.875" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=2, p2: 88%">88%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.29x | 2,889 tok/s</span><span>MAL 3.59 | AR 86.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.945" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=3, p1: 95%">95%</td>
              <td class="acceptance-cell" style="--accept: 0.866" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=3, p2: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=3, p3: 78%">78%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.28x | 2,871 tok/s</span><span>MAL 4.24 | AR 81.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.941" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=4, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.856" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=4, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.766" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=4, p3: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.678" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=4, p4: 68%">68%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.35x | 3,020 tok/s</span><span>MAL 4.79 | AR 75.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.936" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=5, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.848" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=5, p2: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.755" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=5, p3: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.666" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=5, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.582" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=5, p5: 58%">58%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.49x | 3,334 tok/s</span><span>MAL 5.25 | AR 70.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.933" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=6, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.841" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=6, p2: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.747" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=6, p3: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.656" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=6, p4: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.572" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=6, p5: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.496" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MATH500, N=6, p6: 50%">50%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-native-mtp-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="Native MTP" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / Native MTP / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 2,193 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.87x | 1,900 tok/s</span><span>MAL 1.92 | AR 91.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.916" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=1, p1: 92%">92%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.07x | 2,346 tok/s</span><span>MAL 2.70 | AR 84.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.910" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=2, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.787" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=2, p2: 79%">79%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.20x | 2,640 tok/s</span><span>MAL 3.33 | AR 77.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.902" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.774" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=3, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.655" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=3, p3: 66%">66%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.17x | 2,559 tok/s</span><span>MAL 3.85 | AR 71.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.897" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=4, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.768" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=4, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.646" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=4, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.541" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=4, p4: 54%">54%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.18x | 2,587 tok/s</span><span>MAL 4.21 | AR 64.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.887" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=5, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.749" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=5, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.623" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=5, p3: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.517" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=5, p4: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.433" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=5, p5: 43%">43%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.28x | 2,811 tok/s</span><span>MAL 4.51 | AR 58.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.882" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=6, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.739" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=6, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.611" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=6, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.502" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=6, p4: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.420" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=6, p5: 42%">42%</td>
              <td class="acceptance-cell" style="--accept: 0.353" title="Qwen/Qwen3.6-35B-A3B / Native MTP / HumanEval, N=6, p6: 35%">35%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-native-mtp-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="Native MTP" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / Native MTP / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 2,258 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>0.89x | 2,005 tok/s</span><span>MAL 1.90 | AR 90.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=1, p1: 91%">91%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.10x | 2,480 tok/s</span><span>MAL 2.66 | AR 82.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.896" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=2, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.762" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=2, p2: 76%">76%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.23x | 2,773 tok/s</span><span>MAL 3.26 | AR 75.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.889" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.749" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=3, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.619" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=3, p3: 62%">62%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.19x | 2,676 tok/s</span><span>MAL 3.72 | AR 67.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.881" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=4, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.736" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=4, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.605" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=4, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.496" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=4, p4: 50%">50%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.22x | 2,747 tok/s</span><span>MAL 4.08 | AR 61.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.874" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=5, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.726" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=5, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.594" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=5, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.485" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=5, p4: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.398" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=5, p5: 40%">40%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=6</strong>
                <small><span>1.29x | 2,903 tok/s</span><span>MAL 4.34 | AR 55.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.867" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=6, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.716" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=6, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.582" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=6, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.474" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=6, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.387" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=6, p5: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.313" title="Qwen/Qwen3.6-35B-A3B / Native MTP / MBPP, N=6, p6: 31%">31%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-dflash-gsm8k" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 2,275 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.54x | 3,510 tok/s</span><span>MAL 3.47 | AR 82.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.924" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.823" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=3, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.725" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=3, p3: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.88x | 4,276 tok/s</span><span>MAL 5.42 | AR 63.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.903" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.786" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.689" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p3: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.609" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p4: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.539" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p5: 54%">54%</td>
              <td class="acceptance-cell" style="--accept: 0.477" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p6: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.417" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=7, p7: 42%">42%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.70x | 3,871 tok/s</span><span>MAL 6.40 | AR 49.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.893" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.766" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.664" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.579" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p4: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.509" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p5: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.445" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p6: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.391" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p7: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.346" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p8: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.304" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p9: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.269" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p10: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.234" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=11, p11: 23%">23%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.49x | 3,394 tok/s</span><span>MAL 6.88 | AR 39.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.886" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.752" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.646" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.561" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.490" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p5: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.427" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p6: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.373" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p7: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.327" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p8: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.288" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p9: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.253" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p10: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.223" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p11: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.197" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p12: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.173" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p13: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.153" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p14: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.134" title="Qwen/Qwen3.6-35B-A3B / DFlash / GSM8K, N=15, p15: 13%">13%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-dflash-math500" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="DFlash" data-benchmark="math500">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 2,235 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.64x | 3,655 tok/s</span><span>MAL 3.58 | AR 86.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.943" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=3, p1: 94%">94%</td>
              <td class="acceptance-cell" style="--accept: 0.862" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=3, p2: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=3, p3: 78%">78%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.06x | 4,600 tok/s</span><span>MAL 5.82 | AR 68.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.925" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.827" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.742" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p3: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.670" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p4: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.607" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p5: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.551" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p6: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.495" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=7, p7: 50%">50%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.97x | 4,404 tok/s</span><span>MAL 7.13 | AR 55.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.914" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.804" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.714" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p3: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.640" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p4: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.577" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p5: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.522" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p6: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.473" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p7: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.428" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p8: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.388" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p9: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.350" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p10: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.314" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=11, p11: 31%">31%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.76x | 3,938 tok/s</span><span>MAL 7.80 | AR 45.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.792" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p2: 79%">79%</td>
              <td class="acceptance-cell" style="--accept: 0.697" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.618" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p4: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.553" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p5: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.495" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p6: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.445" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p7: 45%">45%</td>
              <td class="acceptance-cell" style="--accept: 0.399" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p8: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.360" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p9: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.324" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p10: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.293" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p11: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.265" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p12: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.239" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p13: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.216" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p14: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.193" title="Qwen/Qwen3.6-35B-A3B / DFlash / MATH500, N=15, p15: 19%">19%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-dflash-humaneval" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 2,193 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.58x | 3,476 tok/s</span><span>MAL 3.42 | AR 80.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.919" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.804" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=3, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.696" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=3, p3: 70%">70%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.84x | 4,036 tok/s</span><span>MAL 5.22 | AR 60.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.901" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.774" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.663" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.570" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.497" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p5: 50%">50%</td>
              <td class="acceptance-cell" style="--accept: 0.435" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p6: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.380" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=7, p7: 38%">38%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.63x | 3,584 tok/s</span><span>MAL 6.01 | AR 45.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.888" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.745" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.627" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p3: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.529" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p4: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.455" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p5: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.393" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.343" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p7: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.304" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p8: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.270" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p9: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.242" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p10: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.215" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=11, p11: 22%">22%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.52x | 3,334 tok/s</span><span>MAL 6.39 | AR 35.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.879" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.732" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.610" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.509" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.430" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p5: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.368" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.318" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p7: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.278" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p8: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.245" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p9: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.217" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p10: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.195" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p11: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.175" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p12: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.158" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p13: 16%">16%</td>
              <td class="acceptance-cell" style="--accept: 0.143" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p14: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.130" title="Qwen/Qwen3.6-35B-A3B / DFlash / HumanEval, N=15, p15: 13%">13%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-qwen-qwen3-6-35b-a3b-dflash-mbpp" class="appendix-acceptance-panel" data-target="Qwen/Qwen3.6-35B-A3B" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>Qwen/Qwen3.6-35B-A3B</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 2,258 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.53x | 3,462 tok/s</span><span>MAL 3.34 | AR 77.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.777" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=3, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.655" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=3, p3: 66%">66%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>1.77x | 3,990 tok/s</span><span>MAL 4.87 | AR 55.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.882" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.735" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.612" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.517" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p4: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.437" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p5: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.373" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p6: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.317" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=7, p7: 32%">32%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>1.53x | 3,444 tok/s</span><span>MAL 5.52 | AR 41.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.870" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.715" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.587" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p3: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.488" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p4: 49%">49%</td>
              <td class="acceptance-cell" style="--accept: 0.409" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p5: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.346" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p6: 35%">35%</td>
              <td class="acceptance-cell" style="--accept: 0.295" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p7: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.252" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p8: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.216" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.187" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p10: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.159" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=11, p11: 16%">16%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.38x | 3,127 tok/s</span><span>MAL 5.81 | AR 32.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.867" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.705" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.573" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.472" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.393" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p5: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.329" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p6: 33%">33%</td>
              <td class="acceptance-cell" style="--accept: 0.279" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p7: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.237" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p8: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.201" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p9: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.173" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p10: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.149" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p11: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.130" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p12: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.114" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p13: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.100" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p14: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.086" title="Qwen/Qwen3.6-35B-A3B / DFlash / MBPP, N=15, p15: 9%">9%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-eagle-3-gsm8k" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="EAGLE-3" data-benchmark="gsm8k">
    <h3><code>moonshotai/Kimi-K2.5</code> / EAGLE-3 / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 324 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.54x | 499 tok/s</span><span>MAL 1.92 | AR 91.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.916" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=1, p1: 92%">92%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.85x | 600 tok/s</span><span>MAL 2.72 | AR 85.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.908" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=2, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.807" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=2, p2: 81%">81%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.09x | 677 tok/s</span><span>MAL 3.40 | AR 80.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.902" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.798" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=3, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.700" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=3, p3: 70%">70%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.24x | 728 tok/s</span><span>MAL 3.96 | AR 73.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.894" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=4, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.783" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=4, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.683" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=4, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.596" title="moonshotai/Kimi-K2.5 / EAGLE-3 / GSM8K, N=4, p4: 60%">60%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-eagle-3-math500" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="EAGLE-3" data-benchmark="math500">
    <h3><code>moonshotai/Kimi-K2.5</code> / EAGLE-3 / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 310 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.54x | 480 tok/s</span><span>MAL 1.94 | AR 93.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.936" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=1, p1: 94%">94%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.88x | 584 tok/s</span><span>MAL 2.77 | AR 88.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.929" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=2, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.843" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=2, p2: 84%">84%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.14x | 664 tok/s</span><span>MAL 3.48 | AR 82.7%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.920" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.827" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=3, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=3, p3: 73%">73%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.33x | 722 tok/s</span><span>MAL 4.09 | AR 77.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.915" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=4, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.818" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=4, p2: 82%">82%</td>
              <td class="acceptance-cell" style="--accept: 0.722" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=4, p3: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.633" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MATH500, N=4, p4: 63%">63%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-eagle-3-humaneval" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="EAGLE-3" data-benchmark="humaneval">
    <h3><code>moonshotai/Kimi-K2.5</code> / EAGLE-3 / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 301 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.51x | 456 tok/s</span><span>MAL 1.90 | AR 90.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.903" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=1, p1: 90%">90%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.81x | 546 tok/s</span><span>MAL 2.67 | AR 83.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.894" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=2, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.777" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=2, p2: 78%">78%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.03x | 610 tok/s</span><span>MAL 3.30 | AR 76.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.885" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.763" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=3, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.655" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=3, p3: 66%">66%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.16x | 649 tok/s</span><span>MAL 3.79 | AR 69.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.873" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=4, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.744" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=4, p2: 74%">74%</td>
              <td class="acceptance-cell" style="--accept: 0.634" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=4, p3: 63%">63%</td>
              <td class="acceptance-cell" style="--accept: 0.540" title="moonshotai/Kimi-K2.5 / EAGLE-3 / HumanEval, N=4, p4: 54%">54%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-eagle-3-mbpp" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="EAGLE-3" data-benchmark="mbpp">
    <h3><code>moonshotai/Kimi-K2.5</code> / EAGLE-3 / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 311 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.52x | 472 tok/s</span><span>MAL 1.88 | AR 88.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.881" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=1, p1: 88%">88%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.78x | 553 tok/s</span><span>MAL 2.59 | AR 79.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.868" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=2, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.724" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=2, p2: 72%">72%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.95x | 608 tok/s</span><span>MAL 3.14 | AR 71.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.857" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=3, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.709" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=3, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.578" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=3, p3: 58%">58%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.99x | 619 tok/s</span><span>MAL 3.53 | AR 63.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.842" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=4, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.688" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=4, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.556" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=4, p3: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.448" title="moonshotai/Kimi-K2.5 / EAGLE-3 / MBPP, N=4, p4: 45%">45%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-dflash-gsm8k" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="DFlash" data-benchmark="gsm8k">
    <h3><code>moonshotai/Kimi-K2.5</code> / DFlash / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 324 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.01x | 651 tok/s</span><span>MAL 3.30 | AR 76.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.891" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.765" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=3, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.641" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=3, p3: 64%">64%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.37x | 768 tok/s</span><span>MAL 4.80 | AR 54.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.871" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.612" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.509" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.426" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p5: 43%">43%</td>
              <td class="acceptance-cell" style="--accept: 0.355" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p6: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.292" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=7, p7: 29%">29%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.23x | 723 tok/s</span><span>MAL 5.06 | AR 36.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.858" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.706" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.576" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.469" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.381" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p5: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.307" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p6: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.245" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p7: 25%">25%</td>
              <td class="acceptance-cell" style="--accept: 0.192" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p8: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.146" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p9: 15%">15%</td>
              <td class="acceptance-cell" style="--accept: 0.106" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p10: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.073" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=11, p11: 7%">7%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>2.05x | 665 tok/s</span><span>MAL 5.02 | AR 26.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.848" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.696" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p2: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.565" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p3: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.458" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p4: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.371" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p5: 37%">37%</td>
              <td class="acceptance-cell" style="--accept: 0.297" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p6: 30%">30%</td>
              <td class="acceptance-cell" style="--accept: 0.236" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p7: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.182" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p8: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.135" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p9: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.095" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p10: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.062" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p11: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.038" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p12: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.022" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p13: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.011" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.006" title="moonshotai/Kimi-K2.5 / DFlash / GSM8K, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-dflash-math500" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="DFlash" data-benchmark="math500">
    <h3><code>moonshotai/Kimi-K2.5</code> / DFlash / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 310 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.12x | 659 tok/s</span><span>MAL 3.49 | AR 83.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.927" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=3, p1: 93%">93%</td>
              <td class="acceptance-cell" style="--accept: 0.833" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=3, p2: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.734" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=3, p3: 73%">73%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.68x | 832 tok/s</span><span>MAL 5.38 | AR 62.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.906" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.797" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.695" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.605" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p4: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.528" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p5: 53%">53%</td>
              <td class="acceptance-cell" style="--accept: 0.459" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p6: 46%">46%</td>
              <td class="acceptance-cell" style="--accept: 0.394" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=7, p7: 39%">39%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.64x | 818 tok/s</span><span>MAL 5.90 | AR 44.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.896" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.770" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.659" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p3: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.564" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p4: 56%">56%</td>
              <td class="acceptance-cell" style="--accept: 0.481" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p5: 48%">48%</td>
              <td class="acceptance-cell" style="--accept: 0.407" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p6: 41%">41%</td>
              <td class="acceptance-cell" style="--accept: 0.338" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p7: 34%">34%</td>
              <td class="acceptance-cell" style="--accept: 0.276" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p8: 28%">28%</td>
              <td class="acceptance-cell" style="--accept: 0.219" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p9: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.168" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p10: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.122" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=11, p11: 12%">12%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>2.42x | 750 tok/s</span><span>MAL 5.86 | AR 32.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.889" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.761" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.649" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p3: 65%">65%</td>
              <td class="acceptance-cell" style="--accept: 0.551" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p4: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.467" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p5: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.390" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p6: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.321" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p7: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.256" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p8: 26%">26%</td>
              <td class="acceptance-cell" style="--accept: 0.197" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p9: 20%">20%</td>
              <td class="acceptance-cell" style="--accept: 0.144" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p10: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.099" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p11: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.064" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p12: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.038" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p13: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.021" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p14: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.011" title="moonshotai/Kimi-K2.5 / DFlash / MATH500, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-dflash-humaneval" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="DFlash" data-benchmark="humaneval">
    <h3><code>moonshotai/Kimi-K2.5</code> / DFlash / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 301 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>2.02x | 609 tok/s</span><span>MAL 3.32 | AR 77.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.895" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=3, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.771" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=3, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.656" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=3, p3: 66%">66%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.42x | 727 tok/s</span><span>MAL 4.87 | AR 55.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.872" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.733" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p2: 73%">73%</td>
              <td class="acceptance-cell" style="--accept: 0.613" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p3: 61%">61%</td>
              <td class="acceptance-cell" style="--accept: 0.515" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p4: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.437" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p5: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.376" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p6: 38%">38%</td>
              <td class="acceptance-cell" style="--accept: 0.323" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=7, p7: 32%">32%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.32x | 699 tok/s</span><span>MAL 5.24 | AR 38.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.861" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.705" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.572" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p3: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.468" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.388" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p5: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.322" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.269" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p7: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.222" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.182" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p9: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.144" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p10: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.109" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=11, p11: 11%">11%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>2.20x | 661 tok/s</span><span>MAL 5.34 | AR 28.9%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.861" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p1: 86%">86%</td>
              <td class="acceptance-cell" style="--accept: 0.707" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p2: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.577" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p3: 58%">58%</td>
              <td class="acceptance-cell" style="--accept: 0.471" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p4: 47%">47%</td>
              <td class="acceptance-cell" style="--accept: 0.386" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p5: 39%">39%</td>
              <td class="acceptance-cell" style="--accept: 0.319" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p6: 32%">32%</td>
              <td class="acceptance-cell" style="--accept: 0.265" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p7: 27%">27%</td>
              <td class="acceptance-cell" style="--accept: 0.217" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p8: 22%">22%</td>
              <td class="acceptance-cell" style="--accept: 0.172" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p9: 17%">17%</td>
              <td class="acceptance-cell" style="--accept: 0.130" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p10: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.093" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p11: 9%">9%</td>
              <td class="acceptance-cell" style="--accept: 0.063" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p12: 6%">6%</td>
              <td class="acceptance-cell" style="--accept: 0.040" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p13: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.024" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p14: 2%">2%</td>
              <td class="acceptance-cell" style="--accept: 0.013" title="moonshotai/Kimi-K2.5 / DFlash / HumanEval, N=15, p15: 1%">1%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-kimi-k2-5-dflash-mbpp" class="appendix-acceptance-panel" data-target="moonshotai/Kimi-K2.5" data-method="DFlash" data-benchmark="mbpp">
    <h3><code>moonshotai/Kimi-K2.5</code> / DFlash / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 311 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
              <th>p6</th>
              <th>p7</th>
              <th>p8</th>
              <th>p9</th>
              <th>p10</th>
              <th>p11</th>
              <th>p12</th>
              <th>p13</th>
              <th>p14</th>
              <th>p15</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.96x | 609 tok/s</span><span>MAL 3.17 | AR 72.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.869" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=3, p1: 87%">87%</td>
              <td class="acceptance-cell" style="--accept: 0.721" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=3, p2: 72%">72%</td>
              <td class="acceptance-cell" style="--accept: 0.582" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=3, p3: 58%">58%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=7</strong>
                <small><span>2.21x | 687 tok/s</span><span>MAL 4.41 | AR 48.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.851" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p1: 85%">85%</td>
              <td class="acceptance-cell" style="--accept: 0.689" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p2: 69%">69%</td>
              <td class="acceptance-cell" style="--accept: 0.551" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p3: 55%">55%</td>
              <td class="acceptance-cell" style="--accept: 0.443" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p4: 44%">44%</td>
              <td class="acceptance-cell" style="--accept: 0.358" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p5: 36%">36%</td>
              <td class="acceptance-cell" style="--accept: 0.289" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p6: 29%">29%</td>
              <td class="acceptance-cell" style="--accept: 0.232" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=7, p7: 23%">23%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=11</strong>
                <small><span>2.04x | 636 tok/s</span><span>MAL 4.53 | AR 32.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.838" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p1: 84%">84%</td>
              <td class="acceptance-cell" style="--accept: 0.663" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p2: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.516" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p3: 52%">52%</td>
              <td class="acceptance-cell" style="--accept: 0.399" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.312" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p5: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.242" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p6: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.186" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p7: 19%">19%</td>
              <td class="acceptance-cell" style="--accept: 0.141" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p8: 14%">14%</td>
              <td class="acceptance-cell" style="--accept: 0.105" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p9: 11%">11%</td>
              <td class="acceptance-cell" style="--accept: 0.076" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p10: 8%">8%</td>
              <td class="acceptance-cell" style="--accept: 0.051" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=11, p11: 5%">5%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=15</strong>
                <small><span>1.88x | 586 tok/s</span><span>MAL 4.50 | AR 23.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.831" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p1: 83%">83%</td>
              <td class="acceptance-cell" style="--accept: 0.655" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p2: 66%">66%</td>
              <td class="acceptance-cell" style="--accept: 0.509" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p3: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.395" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p4: 40%">40%</td>
              <td class="acceptance-cell" style="--accept: 0.306" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p5: 31%">31%</td>
              <td class="acceptance-cell" style="--accept: 0.235" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p6: 24%">24%</td>
              <td class="acceptance-cell" style="--accept: 0.179" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p7: 18%">18%</td>
              <td class="acceptance-cell" style="--accept: 0.134" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p8: 13%">13%</td>
              <td class="acceptance-cell" style="--accept: 0.096" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p9: 10%">10%</td>
              <td class="acceptance-cell" style="--accept: 0.066" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p10: 7%">7%</td>
              <td class="acceptance-cell" style="--accept: 0.043" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p11: 4%">4%</td>
              <td class="acceptance-cell" style="--accept: 0.026" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p12: 3%">3%</td>
              <td class="acceptance-cell" style="--accept: 0.014" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p13: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.008" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p14: 1%">1%</td>
              <td class="acceptance-cell" style="--accept: 0.003" title="moonshotai/Kimi-K2.5 / DFlash / MBPP, N=15, p15: 0%">0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-minimax-m3-mxfp8-eagle-3-gsm8k" class="appendix-acceptance-panel" data-target="MiniMaxAI/MiniMax-M3-MXFP8" data-method="EAGLE-3" data-benchmark="gsm8k">
    <h3><code>MiniMaxAI/MiniMax-M3-MXFP8</code> / EAGLE-3 / GSM8K</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>GSM8K <span>baseline 2,086 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.31x | 2,743 tok/s</span><span>MAL 1.92 | AR 92.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.922" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=1, p1: 92%">92%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.56x | 3,249 tok/s</span><span>MAL 2.73 | AR 86.4%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.912" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=2, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.816" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=2, p2: 82%">82%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.65x | 3,434 tok/s</span><span>MAL 3.42 | AR 80.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.905" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.806" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.711" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.82x | 3,807 tok/s</span><span>MAL 4.01 | AR 75.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.898" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=4, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.796" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=4, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.702" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=4, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.616" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=4, p4: 62%">62%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.82x | 3,787 tok/s</span><span>MAL 4.45 | AR 69.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.890" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=5, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.778" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=5, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.677" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=5, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.590" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=5, p4: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.515" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / GSM8K, N=5, p5: 52%">52%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-minimax-m3-mxfp8-eagle-3-math500" class="appendix-acceptance-panel" data-target="MiniMaxAI/MiniMax-M3-MXFP8" data-method="EAGLE-3" data-benchmark="math500">
    <h3><code>MiniMaxAI/MiniMax-M3-MXFP8</code> / EAGLE-3 / MATH500</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MATH500 <span>baseline 2,468 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.35x | 3,338 tok/s</span><span>MAL 1.93 | AR 93.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.930" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=1, p1: 93%">93%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.64x | 4,047 tok/s</span><span>MAL 2.74 | AR 87.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.920" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=2, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.822" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=2, p2: 82%">82%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.84x | 4,551 tok/s</span><span>MAL 3.44 | AR 81.3%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.915" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=3, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.813" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.712" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.93x | 4,772 tok/s</span><span>MAL 4.01 | AR 75.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.907" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=4, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.801" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=4, p2: 80%">80%</td>
              <td class="acceptance-cell" style="--accept: 0.698" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=4, p3: 70%">70%</td>
              <td class="acceptance-cell" style="--accept: 0.602" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=4, p4: 60%">60%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.90x | 4,677 tok/s</span><span>MAL 4.39 | AR 67.8%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.895" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=5, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.777" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=5, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.665" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=5, p3: 67%">67%</td>
              <td class="acceptance-cell" style="--accept: 0.567" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=5, p4: 57%">57%</td>
              <td class="acceptance-cell" style="--accept: 0.485" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MATH500, N=5, p5: 49%">49%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-minimax-m3-mxfp8-eagle-3-humaneval" class="appendix-acceptance-panel" data-target="MiniMaxAI/MiniMax-M3-MXFP8" data-method="EAGLE-3" data-benchmark="humaneval">
    <h3><code>MiniMaxAI/MiniMax-M3-MXFP8</code> / EAGLE-3 / HumanEval</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>HumanEval <span>baseline 2,317 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.39x | 3,224 tok/s</span><span>MAL 1.93 | AR 93.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.931" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=1, p1: 93%">93%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.70x | 3,931 tok/s</span><span>MAL 2.74 | AR 87.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.919" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=2, p1: 92%">92%</td>
              <td class="acceptance-cell" style="--accept: 0.823" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=2, p2: 82%">82%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.82x | 4,208 tok/s</span><span>MAL 3.43 | AR 81.0%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.911" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=3, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.808" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=3, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.712" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=3, p3: 71%">71%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>2.09x | 4,835 tok/s</span><span>MAL 4.05 | AR 76.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.910" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=4, p1: 91%">91%</td>
              <td class="acceptance-cell" style="--accept: 0.807" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=4, p2: 81%">81%</td>
              <td class="acceptance-cell" style="--accept: 0.710" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=4, p3: 71%">71%</td>
              <td class="acceptance-cell" style="--accept: 0.623" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=4, p4: 62%">62%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.95x | 4,529 tok/s</span><span>MAL 4.46 | AR 69.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.897" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=5, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.783" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=5, p2: 78%">78%</td>
              <td class="acceptance-cell" style="--accept: 0.679" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=5, p3: 68%">68%</td>
              <td class="acceptance-cell" style="--accept: 0.588" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=5, p4: 59%">59%</td>
              <td class="acceptance-cell" style="--accept: 0.513" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / HumanEval, N=5, p5: 51%">51%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="appendix-panel-minimax-m3-mxfp8-eagle-3-mbpp" class="appendix-acceptance-panel" data-target="MiniMaxAI/MiniMax-M3-MXFP8" data-method="EAGLE-3" data-benchmark="mbpp">
    <h3><code>MiniMaxAI/MiniMax-M3-MXFP8</code> / EAGLE-3 / MBPP</h3>
    <div class="acceptance-card acceptance-card--single">
      <h4>MBPP <span>baseline 2,277 tok/s</span></h4>
      <div class="acceptance-table">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>p1</th>
              <th>p2</th>
              <th>p3</th>
              <th>p4</th>
              <th>p5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>N=1</strong>
                <small><span>1.36x | 3,095 tok/s</span><span>MAL 1.91 | AR 90.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.906" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=1, p1: 91%">91%</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=2</strong>
                <small><span>1.68x | 3,825 tok/s</span><span>MAL 2.68 | AR 84.2%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.900" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=2, p1: 90%">90%</td>
              <td class="acceptance-cell" style="--accept: 0.784" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=2, p2: 78%">78%</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=3</strong>
                <small><span>1.89x | 4,298 tok/s</span><span>MAL 3.31 | AR 77.1%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.891" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=3, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.771" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=3, p2: 77%">77%</td>
              <td class="acceptance-cell" style="--accept: 0.650" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=3, p3: 65%">65%</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=4</strong>
                <small><span>1.97x | 4,487 tok/s</span><span>MAL 3.82 | AR 70.5%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.886" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=4, p1: 89%">89%</td>
              <td class="acceptance-cell" style="--accept: 0.761" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=4, p2: 76%">76%</td>
              <td class="acceptance-cell" style="--accept: 0.639" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=4, p3: 64%">64%</td>
              <td class="acceptance-cell" style="--accept: 0.533" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=4, p4: 53%">53%</td>
              <td></td>
            </tr>
            <tr>
              <td>
                <strong>N=5</strong>
                <small><span>1.93x | 4,392 tok/s</span><span>MAL 4.18 | AR 63.6%</span></small>
              </td>
              <td class="acceptance-cell" style="--accept: 0.875" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=5, p1: 88%">88%</td>
              <td class="acceptance-cell" style="--accept: 0.746" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=5, p2: 75%">75%</td>
              <td class="acceptance-cell" style="--accept: 0.619" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=5, p3: 62%">62%</td>
              <td class="acceptance-cell" style="--accept: 0.513" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=5, p4: 51%">51%</td>
              <td class="acceptance-cell" style="--accept: 0.427" title="MiniMaxAI/MiniMax-M3-MXFP8 / EAGLE-3 / MBPP, N=5, p5: 43%">43%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

</div>

<script>
(function() {
  const methodsByModel = {
    "google/gemma-4-26B-A4B-it": ["Gemma 4 MTP", "EAGLE-3", "DFlash"],
    "google/gemma-4-31B-it": ["Gemma 4 MTP", "EAGLE-3", "DFlash", "DSpark"],
    "Qwen/Qwen3-8B": ["EAGLE-3", "DFlash", "DSpark"],
    "Qwen/Qwen3.5-27B": ["Native MTP", "DFlash"],
    "Qwen/Qwen3.5-122B-A10B": ["Native MTP", "DFlash"],
    "Qwen/Qwen3.6-27B": ["Native MTP", "DFlash"],
    "Qwen/Qwen3.6-35B-A3B": ["Native MTP", "DFlash"],
    "moonshotai/Kimi-K2.5": ["EAGLE-3", "DFlash"],
    "MiniMaxAI/MiniMax-M3-MXFP8": ["EAGLE-3"]
  };
  const benchmarksByModelMethod = {
    "google/gemma-4-26B-A4B-it||Gemma 4 MTP": ["gsm8k", "math500", "humaneval", "mbpp"],
    "google/gemma-4-26B-A4B-it||EAGLE-3": ["gsm8k", "math500", "humaneval", "mbpp"],
    "google/gemma-4-26B-A4B-it||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "google/gemma-4-31B-it||Gemma 4 MTP": ["gsm8k", "math500", "humaneval", "mbpp"],
    "google/gemma-4-31B-it||EAGLE-3": ["gsm8k", "math500", "humaneval", "mbpp"],
    "google/gemma-4-31B-it||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "google/gemma-4-31B-it||DSpark": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3-8B||EAGLE-3": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3-8B||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3-8B||DSpark": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.5-27B||Native MTP": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.5-27B||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.5-122B-A10B||Native MTP": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.5-122B-A10B||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.6-27B||Native MTP": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.6-27B||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.6-35B-A3B||Native MTP": ["gsm8k", "math500", "humaneval", "mbpp"],
    "Qwen/Qwen3.6-35B-A3B||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "moonshotai/Kimi-K2.5||EAGLE-3": ["gsm8k", "math500", "humaneval", "mbpp"],
    "moonshotai/Kimi-K2.5||DFlash": ["gsm8k", "math500", "humaneval", "mbpp"],
    "MiniMaxAI/MiniMax-M3-MXFP8||EAGLE-3": ["gsm8k", "math500", "humaneval", "mbpp"]
  };
  const modelSelect = document.getElementById("appendix-model-select");
  const methodSelect = document.getElementById("appendix-method-select");
  const benchmarkSelect = document.getElementById("appendix-benchmark-select");
  if (!modelSelect || !methodSelect || !benchmarkSelect) return;
  const panels = Array.from(document.querySelectorAll(".appendix-acceptance-panel"));
  const fillSelect = (select, values, previous, labelFor = (value) => value) => {
    select.innerHTML = "";
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labelFor(value);
      select.appendChild(option);
    });
    select.value = values.includes(previous) ? previous : values[0];
  };
  const fillMethods = () => {
    fillSelect(methodSelect, methodsByModel[modelSelect.value] || [], methodSelect.value);
  };
  const fillBenchmarks = () => {
    const key = modelSelect.value + "||" + methodSelect.value;
    fillSelect(benchmarkSelect, benchmarksByModelMethod[key] || [], benchmarkSelect.value, (value) => value.toUpperCase());
  };
  const showPanel = () => {
    panels.forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.target === modelSelect.value && panel.dataset.method === methodSelect.value && panel.dataset.benchmark === benchmarkSelect.value);
    });
  };
  modelSelect.addEventListener("change", () => {
    fillMethods();
    fillBenchmarks();
    showPanel();
  });
  methodSelect.addEventListener("change", () => {
    fillBenchmarks();
    showPanel();
  });
  benchmarkSelect.addEventListener("change", showPanel);
  fillMethods();
  fillBenchmarks();
  showPanel();
}());
</script>

<p class="appendix-metric-note"><strong>MAL</strong> means mean accepted length. <strong>AR</strong> means acceptance rate.</p>

<details markdown="1">
<summary>Example vLLM serve commands used in the experiments</summary>

### `google/gemma-4-26B-A4B-it`

Baseline:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768
```

Gemma 4 MTP:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

EAGLE-3:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --speculative-config '{"model":"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3","num_speculative_tokens":1,"method":"eagle3"}'
```

DFlash:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend triton_attn \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --speculative-config '{"method":"dflash","model":"z-lab/gemma-4-26B-A4B-it-DFlash","num_speculative_tokens":15,"attention_backend":"triton_attn"}'
```

### `google/gemma-4-31B-it`

Baseline:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768
```

Gemma 4 MTP:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}'
```

EAGLE-3:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3"}'
```

DFlash:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend triton_attn \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dflash","model":"z-lab/gemma-4-31B-it-DFlash","num_speculative_tokens":15,"attention_backend":"triton_attn"}'
```

DSpark:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend triton_attn \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.dspark","num_speculative_tokens":7,"method":"dspark","attention_backend":"triton_attn"}'
```

### `Qwen/Qwen3-8B`

Baseline:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

EAGLE-3:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"RedHatAI/Qwen3-8B-Thinking-speculator.eagle3","num_speculative_tokens":5,"method":"eagle3"}'
```

DFlash:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"z-lab/Qwen3-8B-DFlash-b16","method":"dflash","num_speculative_tokens":7}'
```

DSpark:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"deepseek-ai/dspark_qwen3_8b_block7","method":"dspark","num_speculative_tokens":11}'
```

### `Qwen/Qwen3.5-27B`

Baseline:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768
```

Native MTP:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

DFlash:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-27B-DFlash","num_speculative_tokens":15}'
```

### `Qwen/Qwen3.5-122B-A10B`

Baseline:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768
```

Native MTP:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}'
```

DFlash:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-122B-A10B-DFlash","num_speculative_tokens":15}'
```

### `Qwen/Qwen3.6-27B`

Baseline:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768
```

Native MTP:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

DFlash:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'
```

### `Qwen/Qwen3.6-35B-A3B`

Baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384
```

Native MTP:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

DFlash:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":15}'
```

### `moonshotai/Kimi-K2.5`

Baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --language-model-only \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2
```

EAGLE-3:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --language-model-only \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --speculative-config '{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3}'
```

DFlash:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --language-model-only \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --speculative-config '{"model":"z-lab/Kimi-K2.5-DFlash","method":"dflash","num_speculative_tokens":7}'
```

### `MiniMaxAI/MiniMax-M3-MXFP8`

Baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --attention_config.indexer_kv_dtype fp8 \
  --linear-backend emulation \
  --attention-backend TRITON_ATTN \
  --language-model-only \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3
```

EAGLE-3:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --attention_config.indexer_kv_dtype fp8 \
  --linear-backend emulation \
  --attention-backend TRITON_ATTN \
  --language-model-only \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

</details>

## Acknowledgements

We would like to thank everyone who contributed to this collaboration, including Hongxia Yang and Peng Sun from AMD, and Pin Siang Tan, Jun Kang Chow, and Ye Hur Cheong from Embedded LLM.

---

## Disclaimer

Measurements were run on AMD Instinct™ MI300X and MI355X platforms using the configurations below.

**Hardware Configuration**

- Hardware 1: 8× AMD Instinct™ MI300X GPUs (gfx942) with 2× AMD EPYC™ 9654 96-Core Processor.
- Hardware 2: 8× AMD Instinct™ MI355X GPUs (gfx950) with 2× AMD EPYC™ 9575F 64-Core processors. This platform was used for the MiniMax-M3-MXFP8 experiment.

**Software Configuration**

Ubuntu 22.04.5 LTS, ROCm/HIP runtime 7.2.53211, vLLM 0.23.1rc1.dev1120+g0f0f28b53, PyTorch 2.11.0+gitd0c8b1f, Transformers 5.13.1, Python 3.12.13.

Server manufacturers may vary configurations, yielding different results. Performance may vary based on configuration, software, vLLM version, and the use of the latest drivers and optimizations.

---
