---
layout: post
title: "Adaptive Verification Without a Confidence Head"
author: "Giancarlo Delfin (Inferact)"
summary: "An online-fitted acceptance estimator that lets every draft-model speculator use Adaptive Verification, by predicting the target's acceptance probability from the drafter's own logits and refitting continuously during serving."
image: /assets/figures/2026-10-09-adaptive-verification-online-estimator/cover.png
social_image: /assets/figures/2026-10-09-adaptive-verification-online-estimator/cover.png
math: true
tags:
  - speculative-decoding
  - performance
---

## TL;DR

Confidence scoring is now supported for all draft-model-based speculator types. In the absence of a confidence head, an **online-fitted estimator** derives a feature from the drafter's logits to predict the target model's probability of accepting it, which is used directly as a confidence score for Adaptive Verification. The estimator's coefficients are continuously fitted, adapting to the incoming data traffic. This feature landed in PR [#52228](https://github.com/vllm-project/vllm/pull/52228), and is enabled with `enable_adaptive_verification` for draft models without a confidence head.

## Background

vLLM's [Adaptive Verification](https://github.com/vllm-project/vllm/pull/47808) (AV) enables verification of a subset of the drafted tokens during speculative decoding. Draft tokens with low probability of acceptance are trimmed from the batch before running the target model's forward pass, saving on compute. This optimization is particularly beneficial while serving large batch sizes, because in that regime the GPU becomes compute-bound. See [this vLLM blog post](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) for more details.

AV needs a per-token estimate of acceptance probability, which is why, until now, it only supported DSpark drafters. DSpark models often ship with a confidence head: a small linear layer over the drafter's hidden state for each position, trained as a binary classifier for "will the target accept this token?"

## Problem

Draft models that lack a confidence head, such as EAGLE, MTP, DFlash, and some DSpark checkpoints, miss out on AV entirely. To use it, these drafters need a per-token acceptance estimate, and adding a confidence head means retraining the drafter, which isn't an option for the many existing checkpoints that people already serve.

## Prior Work

Using the drafter's own confidence as a proxy for acceptance is well established. [EAGLE-2](https://arxiv.org/abs/2406.16858) showed that EAGLE's draft probabilities closely track acceptance rates, and used them to grow and rerank a dynamic draft tree. [Draft & Verify](https://arxiv.org/abs/2309.08168) and several follow-ups stop drafting once the drafter's probability falls below a fixed threshold. [SpecDec++](https://arxiv.org/abs/2405.19715) instead trains a dedicated acceptance-prediction head, similar in spirit to DSpark's confidence head. In each case, the mapping from draft confidence to acceptance is either used as-is or learned offline.

AV needs more than a good ranking. Its budget decision sums predicted acceptance probabilities and weighs them against measured costs, so the predictions must be calibrated for the specific model pair and workload being served, which is what the online-fitted estimator in the next section provides.

## Solution: Online-Fitted Acceptance Estimator

We added a lightweight logistic regression that estimates each draft token's acceptance probability from a single scalar derived from the draft distribution, the log-odds of the drafter's most likely token. It has one shared slope and one intercept per draft position, so a *K*-token drafter needs a total of *K* + 1 parameters. Those parameters are fitted online from the accept/reject decisions the target model already produces during normal verification. Any existing draft checkpoint gets AV for free.

The rest of this section covers the two halves of the design: what quantity the estimator scores, and how its coefficients are fitted during serving.

### Scoring the Distribution

The confidence score for a given draft token is estimated using a per-position logistic regression:

$$
p = \sigma\big(\omega \cdot \text{logit}(q_{\max}) + \beta_i\big)
$$

where ω is the fitted slope shared across all draft positions, β<sub>i</sub> is the fitted per-position intercept, and *q* is the draft distribution. The alternate log-odds form makes the relationship clear:

$$
\text{logit}(p) = \omega \cdot \text{logit}(q_{\max}) + \beta_i
$$

logit(*q*<sub>max</sub>) is the estimator's feature, and is the log-odds of the draft's most likely token. Intuitively, it represents how confident the draft model is in its predictions. It is transformed linearly by the coefficients ω and β<sub>i</sub> to predict the log-odds of the draft's acceptance by the target model. The sigmoid of that is *p*, the estimated acceptance rate, which is used as the confidence score for AV.

#### Other Considered Features

Several alternative features were considered and rejected for one reason or another. The four main axes that we cared most about when evaluating a feature were:

1. **Losslessness** — Whether the feature biases the output distribution during speculative decoding.
2. **Area Under the Curve (AUC)** — The probability that a randomly chosen accepted draft scores higher than a randomly selected rejected one. This measures **discrimination** between drafts that should be accepted vs rejected. A higher value indicates a better ability to discriminate, and 0.5 is random chance.
3. **Log-loss** — The mean negative log-likelihood of the predicted acceptance probability, which penalizes confident wrong answers and rewards calibration. A lower value indicates a better ability to predict the true acceptance probability of a given draft.
4. **Expected Calibration Error (ECE)** — Bin each draft based on its predicted acceptance probability, then measure the error between the mean prediction vs the mean observed frequency within each bin, weighted by the bin occupancy. This measures **calibration**, and a lower value indicates a better ability to assign a probability to a draft that matches the true frequency among drafts scored similarly.

Losslessness is a non-negotiable property for the features to have. Otherwise, we break the fundamental guarantee of speculative decoding to output the target model's exact distribution. AUC and ECE both matter for AV's trimming decision, but in different ways. AUC is important for the **local ordering** of survival probabilities:

```python
survival = confidence_probs[idx_mapping].cumprod(dim=1)
...
winners = flat.topk(draft_budget).indices
```

AUC ensures that a more likely to be accepted draft token is admitted over a less likely one. On the other hand, ECE is important for the **global** budget decision:

```python
num_tokens_to_estimated_accepted_tokens = np.concatenate(
    ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
)
costs = draft_cost_ms[...] + verify_cost_ms[...]
...
draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
```

The probabilities are summed and then divided by the measured cost table. ECE ensures that the estimated number of accepted tokens matches reality.

Below is a table of the various experimented features, along with their scores for the four categories. They were measured over ~4.5M labeled drafts collected over the course of serving DeepSeek-V4-Flash + DSpark (temp=1, top-p=0.95, conc=64), without AV, across five datasets: SPEED-Bench 2k low/high-entropy, GSM8K, HumanEval, and MT-Bench. The last two rows are references rather than fitted features: the drafter's *q*<sub>max</sub> used directly as the acceptance probability with no fit, and the trained confidence head's scores.

| feature | lossless | AUC | log-loss | ECE |
| :-- | :-: | :-: | :-: | :-: |
| log(*q*<sub>max</sub>) | yes | 0.8069 | 0.4813 | 0.0592 |
| log(*q*<sub>max</sub>) − log(*q*<sub>2nd</sub>) (top-2 margin) | yes | 0.8156 | 0.4437 | 0.0105 |
| *H*(*q*) (entropy) | yes | 0.8321 | 0.4577 | 0.0638 |
| −log(−log(*q*<sub>max</sub>)) | yes | 0.8336 | 0.4289 | 0.0103 |
| **logit(*q*<sub>max</sub>) (shipped)** | **yes** | **0.8336** | **0.4284** | **0.0100** |
| logit(*q*<sub>sampled</sub>) | no | 0.8553 | 0.4040 | 0.0152 |
| −log(−log(*q*<sub>sampled</sub>)) | no | 0.8554 | 0.4075 | 0.0193 |
| *q*<sub>max</sub> (no fit, used directly as the probability) | yes | 0.8259 | 0.4880 | 0.0736 |
| **trained confidence head** | **yes** | **0.8191** | **0.4449** | **0.0052** |

logit(*q*<sub>sampled</sub>) and −log(−log(*q*<sub>sampled</sub>)) clearly outperform the chosen feature in the table in terms of AUC, but were not used because trimming based on them results in a biased output distribution. Both read *q*<sub>sampled</sub>, the probability of the token the drafter actually drew. Trimming a draft on a function of this value makes the decision to verify at position *i* dependent on *x*<sub>i</sub> itself, so the drafts that survive trimming are no longer distributed as *q*. Standard rejection sampling accepts a token with probability min(1, *p*/*q*), expecting *q* to be the distribution the draft token was sampled from. But trimming based on *q*<sub>sampled</sub> biases the emitted distribution towards the drafter's higher confidence tokens.[^1]

The *q*<sub>max</sub> "no fit" row shows why we fit at all, instead of using the drafter's own probability directly, as EAGLE-2 does when building its draft tree. Used directly, the drafter's top-token probability actually ranks drafts better than the trained confidence head (0.8259 vs 0.8191 AUC). However, it is not well calibrated. Its mean prediction over the labeled drafts is 0.784 against the observed acceptance rate of 0.723, and its ECE is 7x worse than the shipped logit(*q*<sub>max</sub>) feature (0.0736 vs 0.0100). The drafter's top-token probability reflects only how confident the drafter is in its own prediction, not whether the target model will agree. This agreement is further influenced by the sampling setup and the workload. An online fit learns this mapping directly from the target's accept/reject decisions, yielding better calibration, which is what the budget decision depends on.

We ultimately went with the logit(*q*<sub>max</sub>) feature, which has a good balance of AUC and ECE at a reasonable cost. It was also a natural choice, since it shares both the log-odds scale and the resolution of the quantity we were trying to estimate.

#### Versus a Trained Confidence Head

As seen in the table above, for DeepSeek-V4-Flash + DSpark, the estimator outperforms the confidence head in both AUC and log-likelihood loss, but underperforms in ECE. The table below summarizes the difference:

| | AUC ↑ | log-loss ↓ | ECE ↓ |
| :-- | :-: | :-: | :-: |
| estimator logit(*q*<sub>max</sub>) | **0.8336** | **0.4284** | 0.0100 |
| trained confidence head | 0.8191 | 0.4449 | **0.0052** |
| **Δ (estimator − head)** | **+0.0146** | **−0.0166** | +0.0048 |

However, once we exclude the first draft from the averages, the picture changes substantially:

| | AUC ↑ | log-loss ↓ | ECE ↓ |
| :-- | :-: | :-: | :-: |
| estimator logit(*q*<sub>max</sub>) | **0.8352** | **0.4420** | 0.0103 |
| trained confidence head | 0.8075 | 0.4743 | **0.0090** |
| **Δ (estimator − head)** | **+0.0278** | **−0.0323** | +0.0013 |

The head and estimator are now virtually tied on calibration. The per-position AUC breakdown helps explain why:

![Per-position AUC for the online estimator and the trained confidence head on DeepSeek-V4-Flash + DSpark at 7 speculative tokens, over 4.47M labeled drafts. The head leads at the first position; the estimator leads from the second onwards, and both fall off past the fifth position.](/assets/figures/2026-10-09-adaptive-verification-online-estimator/auc-by-draft-position.png)

The confidence head is the better predictor on the first draft position, but the estimator takes the lead from the second position onwards. The gap widens with depth, with the estimator leading by 0.08 AUC by the last position. Both predictors also fall off after the fifth position. This is likely because DSV4-Flash's DSpark declares a block size of 5 and the drafter was never trained past it, so the drafts themselves get worse there.

Since survival is a running product, and the first draft position always carries a request's highest score, it is less often the token that the budget cuts. The estimator's advantage sits right where the trimming happens, and it explains why the estimator can match the confidence head in end-to-end performance.

### Fitting the Coefficients

The coefficients ω and β<sub>i</sub> are continuously refitted during live serving of requests. Draft tokens proposed during the previous decode step are verified by the target model in the next decode step. Only draft tokens up until the first rejection (if any) are eligible for training on, because the target model has explicitly labeled them (accept or reject).

We use Iteratively Reweighted Least Squares (IRLS) to minimize the log-likelihood loss of the predictions vs the labels. This minimization is performed over a second-order Taylor approximation of the log-likelihood loss. The optimization step for ω and β<sub>i</sub> is obtained by solving the system of equations

$$
\mathbf{H} \Delta\boldsymbol{\theta} = -\mathbf{g}
$$

where Δ**θ** is the delta for the shared slope ω and the per-position intercepts β<sub>i</sub>, **H** is the Hessian of the loss function with a shape of (*K* + 1) × (*K* + 1), and **g** is the gradient of the loss function, with a size of *K* + 1.

A refit happens after every fixed round of *N* decode steps. During each decode step in that interval, we accumulate the Hessian and gradient values, which are derived from the coefficients, features, and labels collected during the round. Doing this upper bounds memory to *O*(*K*²) across a round, in contrast with *O*(*N* × *B* × *K*) if we instead collected the raw labels. On the final step of the round, the system is solved, and the coefficients are updated.

As one might expect, deeper drafts are accepted less often than shallower drafts. Consequently, deeper draft positions can be starved of labels during a round. Two design decisions were made to mitigate the impact of this on the refitting process:

1. **Shared Slope**: The slope is shared across all draft positions to prevent noisy updates caused by data thinning at depth. Predictions are much more sensitive to slope errors than intercept errors, so sharing makes the Newton step well-determined. Furthermore, empirical measurements show sharing the slope costs virtually nothing in terms of accuracy (only 0.0002 AUC and 0.0003 log-loss).
2. **Step Damping**: Each coefficient's update is scaled by the amount of supporting evidence using the formula *n* / (*n* + constant), where *n* is the count of graded drafts. This ensures that deeper positions with fewer labels move more conservatively, preventing updates based on noise.

#### Slope vs Intercept Roles

The fitted coefficients were tracked across several model/dataset combinations, and the two kinds of parameters behave very differently.

**Hold the model pair fixed and vary the workload**: Fitting DSV4-Flash + DSpark independently on each of five workloads, the slope barely moves (0.51 to 0.55), uncorrelated with how hard the workload is (r = −0.08 against mean draft entropy). The intercepts do the opposite: the first draft position intercept correlates with draft entropy at r = −0.89, and runs from ~0.5–0.6 on the more predictable workloads down to below 0 on the least.

| Dataset | Acceptance Rate | Mean Draft Entropy | Slope ω | Intercept β<sub>0</sub> |
| :-- | :-: | :-: | :-: | :-: |
| gsm8k | 0.781 | 0.562 | 0.5365 | +0.478 |
| humaneval | 0.751 | 0.669 | 0.5135 | +0.486 |
| speed-bench 2k low-entropy | 0.755 | 0.703 | 0.5087 | +0.610 |
| mt-bench | 0.714 | 0.800 | 0.5495 | +0.181 |
| speed-bench 2k high-entropy | 0.642 | 1.077 | 0.5197 | −0.089 |

**Now vary the model pair:** The slope ranges from 0.45 to 0.59 across pairs, while staying tight within each one. The intercepts also differ between pairs, so unlike the slope they are not a property of one axis alone.

| Model | nspec | Slope ω | ω sd | Intercept β<sub>0</sub> |
| :-- | :-: | :-: | :-: | :-: |
| DeepSeek-V4-Flash + DSpark | 5 | 0.5013 | 0.0250 | +0.279 |
| MiMo-V2.5-Pro-FP4 + DFlash | 8 | 0.5197 | 0.0415 | −1.000 |
| Inkling-NVFP4 + MTP | 8 | 0.5859 | 0.0349 | −0.572 |
| Muse-Glimmer-30B + DFlash2 | 7 | 0.4450 | 0.0323 | +0.167 |

Read together, the two tables suggest a division of labor. The slope measures how well the draft model knows when it's right. This is a property of the target/draft pair and the draft sampling method, which is why it barely moves when only the workload changes. The intercepts absorb the rest: the base acceptance rate at each draft position, which falls with depth and drops even further on less predictable content. That base acceptance rate depends on the model pair as well as the workload, which is why the intercepts move along both axes while the slope moves along just one.

## Benchmarks

We benchmarked three checkpoints (MiMo-V2.5-Pro-FP4 + DFlash, Inkling-NVFP4 + MTP, and Muse-Glimmer-30B + DFlash2) that previously could not use AV at all, along with DSV4-Flash + DSpark + the estimator, and DSV4-Flash + DSpark + the confidence head, as a reference. The gains below were captured with AV off vs on, temperature 1.0, top-p 0.95, and concurrency 64, over four datasets.

![Output throughput gain from Adaptive Verification, AV off vs on, for MiMo-V2.5-Pro, Inkling-NVFP4, Muse-Glimmer-30B, and DeepSeek-V4-Flash + DSpark with both the online estimator and the confidence head, across four datasets at concurrency 64](/assets/figures/2026-10-09-adaptive-verification-online-estimator/gain-by-dataset-with-dsv4.png)

Compared to fixed verification, the online estimator at the same speculative length is a win across all benchmarked datasets. However, that gain alone doesn't tell the whole story. Oftentimes, a smaller fixed length can be found to outperform AV at the max speculative length at specific concurrencies. The Pareto curve below was captured with DSV4 + DSpark across a sweep of fixed speculative lengths and AV at the max speculative length:

![Throughput versus interactivity for DeepSeek-V4-Flash + DSpark across fixed speculative lengths and Adaptive Verification with the online estimator and the confidence head](/assets/figures/2026-10-09-adaptive-verification-online-estimator/dsv4-frontier-throughput-vs-interactivity.png)

The online estimator and the confidence head are roughly on par in terms of performance, across all measured concurrencies. However, the online estimator is not meant to be a replacement for the confidence head, but rather a drop-in alternative in the absence of one. Let's see how it performs on a model pair that currently lacks a confidence head. Below is the Pareto curve for MiMo-V2.5-Pro-FP4 + DFlash across a sweep of fixed speculative lengths and AV:

![Throughput versus interactivity for MiMo-V2.5-Pro-FP4 + DFlash across fixed speculative lengths and Adaptive Verification with the online estimator](/assets/figures/2026-10-09-adaptive-verification-online-estimator/mimo-frontier-throughput-vs-interactivity.png)

AV sits on the throughput-interactivity frontier at 5 of 7 concurrencies.

## Limitations

- The target model's attention backend must support variable-length verification. This is the same prerequisite that DSpark AV already has.
- MTP and EAGLE benefit less from AV than DFlash and DSpark do. This is because those auto-regressive speculators pay a proportionally larger cost for drafting tokens that may get trimmed before verification.

## vLLM Usage

AV is enabled through the speculative config. On a draft model that (1) has no confidence head and (2) supports variable length attention for all of its backends, the online estimator is selected automatically:

```bash
vllm serve <target_model> \
  ...
  --speculative-config '{
    "model": "<draft_model>",
    "method": "<method>",
    "num_speculative_tokens": <num_speculative_tokens>,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

## Acknowledgements

This work was done by Giancarlo Delfin (Inferact), and builds directly on Adaptive Verification by Lucas Wilkinson (Red Hat) and Benjamin Chislett (NVIDIA).

[^1]: An example with a simple two-token vocabulary of [*a*, *b*] can be used to illustrate this bias. Assume *p* = [0.5, 0.5], *q* = [0.2, 0.8], and that a draft token is trimmed from verification if *q*(*x*) < 0.5. Drawing *a* results in trimming, and the target samples a token from its own distribution, *p*(*x*). However, drawing *b* results in verification: the token is accepted with probability min(1, *p*(*b*)/*q*(*b*)) = 0.625, and otherwise rejected and resampled from the residual norm(max(0, *p* − *q*)) = [1, 0], yielding *a*. During rejection sampling we get *p*<sub>out</sub>(*a*) = *q*(*a*)·*p*(*a*) + *q*(*b*)·(1 − 0.625) = 0.4 and *p*<sub>out</sub>(*b*) = *q*(*a*)·*p*(*b*) + *q*(*b*)·0.625 = 0.6. This differs from the target's distribution of [0.5, 0.5] and violates speculative decoding's lossless guarantee.
