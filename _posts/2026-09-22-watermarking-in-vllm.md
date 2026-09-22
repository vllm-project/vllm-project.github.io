---
layout: post
math: true
title: "Watermarking in vLLM"
author: "Raphaël Rialland (Mistral AI), Simon Veitner (Red Hat), and Tomas Ruiz (Red Hat)"
summary: "How vLLM implements distribution-preserving Gumbel-max text watermarking with efficient GPU kernels, statistical detection, speculative decoding, and repeated-context safeguards."
image: /assets/figures/2026-09-22-watermarking-in-vllm/gumbel-max-sampling.png
social_image: /assets/figures/2026-09-22-watermarking-in-vllm/gumbel-max-sampling.png
tags:
  - watermarking
  - sampling
  - speculative-decoding
  - performance
---

<!-- Draft note: confirm the final author list and any equal-contribution attribution before publication. -->

## What is text provenance, and why is it challenging?

Establishing the provenance of digital content is essential to build trust and accountability in the information we share. Watermarking offers a solution: it alters a work to embed *a message about it* [[1]](#ref-1), for instance its provenance. Images and audio can embed watermarks as an imperceptible perturbation, but these techniques are not applicable to text due to its discrete nature. Instead of trying to perturb the text itself, we influence the generation process in a way that is imperceptible and detectable.

Watermarking text is both a scientific and engineering challenge, and there are several requirements that we must aim to balance to have a practical solution:

### Non-distortion

Adding a watermark should not modify the expected output distribution of a model [[2]](#ref-2): it should not systematically favour certain words, writing styles, or solutions to problems.

### Robustness

Generated text can be altered in many ways: shortened, edited, mixed with other corpora, or even rephrased by another LLM. A good watermarking system aims to retain its provenance signal even under such modification.

### Speed

Watermarking should add minimal latency and memory overhead to generation and detection. This is particularly important in a high-throughput serving engine such as vLLM. Slowdowns arise both when serving small models, sensitive to changes in scheduling latency, and models with large vocabulary sizes.

### Minimal detection dependencies

Detecting watermarking on text should require as little information as possible: users seeking to determine the source of a piece of writing may not necessarily have additional information on its origin, such as its date of creation or what language model may have been used to generate it.

These desired qualities pull in different directions: a strong signal that is easily detectable may require us to significantly change a model’s outputs, and avoiding reliance on additional information limits the algorithms available to us.

For example:

- Metadata attached to a file can be stripped off, and they do not persist when copying the data into a new file.
- Red-green lists [[3]](#ref-3) and direct data watermarking can lead to a distortion of the output distribution.
- Unicode substitution is easily removed and tampered with.
- Text modeling techniques (e.g. looking at sentence length, unusual words and expressions) are unreliable.
- Tournament sampling [[2]](#ref-2) requires multiple rounds of processing that can be hard to implement efficiently without impacting inference speed

To satisfy these criteria, we exploit a natural property of language model generations: randomness.

## Exploiting randomness in LLMs

At each generation step, a language model assigns a probability to every possible next token. Decoding may first reshape or filter this distribution, for example with temperature or min-p, before randomly sampling one of the remaining candidates. Repeating this process traces a path through a tree of possible continuations: despite some of these paths being more likely than others, many of them produce plausible final outputs of the model.

Ordinary stochastic sampling draws fresh random values for each sampling step, and then discards them. The key insight to watermarking is that watermarking instead derives a reproducible pseudorandom value from a known key and input (the recent context, and each candidate token).

During generation, we can use these keyed values to influence which tokens vLLM selects, and during detection we can re-derive these same values using the generated tokens. Unwatermarked text should show no systematic alignment with these values, but watermarked text should favor tokens that receive higher pseudorandom values, creating a statistical signal we can detect over a long enough sequence.

The next section discusses how we can implement this process while still preserving the initial sampling distribution of the model, effectively allowing us to *reproducibly* choose one of the possible final outputs of the model.

## Gumbel-max Trick

The Gumbel-max trick is a technique for sampling from a categorical distribution. Suppose the model assigns a probability $p_v$ to each candidate token $v$: for every token, we draw independent uniform random variables $U_v \sim \mathcal{U}(0, 1)$ and transform them into Gumbel random variables: 

$$
G_v = -\log\ (-\log\ U_v) \quad \quad G_v \sim \text{Gumbel(0,1)}
$$

The sampled token is then:

$$
v^\star = \argmax_{v \in \mathcal{V}}\ (\log\ p_v + G_v)
$$

A remarkable result of this transformation is that:

$$
\mathbb{P}(v^\star = v) = p_v
$$

In other words, adding Gumbel noise to the token log-probabilities and taking their argmax produces the exact same categorical distribution as ordinary random sampling: in fact, [Model Runner v2’s standard sampling path](https://github.com/vllm-project/vllm/blob/f92b78f6ef9c9b28f60668da77af5b65645b1a45/vllm/v1/worker/gpu/sample/sampler.py#L341-L352) already uses this process.

![Gumbel-max sampling example: model probabilities for table, mat, and chair are combined with Gumbel noise. “mat” wins one draw, while 20,000 repetitions come close to recovering the original 18%, 57%, and 25% distribution.](/assets/figures/2026-09-22-watermarking-in-vllm/gumbel-max-sampling.png)

To make these random draws reproducible, we use a pseudorandom function (PRF) [[4]](#ref-4). A PRF deterministically maps a key and an input to a value that behaves like a random uniform draw: since the same inputs always produce the same output, a detector can reconstruct the values used during generation. We replace each independent uniform draw $U_v$ with:

$$
U_v = \operatorname{PRF}(K; \mathbf{w}, v)
$$

where $K$ is the secret watermark key, and $\mathbf{w}$ is the recent watermarking context (the last $k$ tokens).

Including the token ID is important as it allows every token to receive different noise, the recent context (by default, we use the last 4 tokens) means that the random noise we apply is different at *almost* every generation step, and the secret key makes this pattern unpredictable for anyone who does not have it.

The generator computes these keyed values for each candidate token, and selects the token with the largest noised log-probability. Given the same key and context this choice can be reproduced, but without the key it looks like an ordinary random sample.

![Bar chart comparing Qwen3.5-27B quality with and without dual-key watermarking: GSM8K 93.0% versus 94.2%, MBPP 79.2% versus 77.2%, and IFEval 90.7% versus 91.9%, with overlapping error bars.](/assets/figures/2026-09-22-watermarking-in-vllm/quality.svg)

This gives the method a precise guarantee of non-distortion: in expectation over keys, the probability of selecting token $v$ remains exactly $p_v$. However, this guarantee does not automatically extend to complete sequences: repeated contexts reuse the same pseudorandom values and can therefore correlate later token choices. We discuss this limitation and its solutions further in “Maintaining output diversity”.

## Detecting a watermark

At detection time, we reverse the process used during generation. It does not require access to the model’s weights or logits, but it does require knowledge about the secret key, and the tokenizer.

A user will provide a sequence of text, with unknown origin, to a detector. This detector will convert the sequence of text back into token IDs. For each token $x_t$, the detector will use the preceding context and the secret key to recompute its pseudorandom value:

$$
U_{x_t} = \operatorname{PRF}(K; \mathbf{w}_t, x_t)
$$

For unwatermarked text, the observed tokens are independent of their keyed values, so $U_{x_t}$ behaves like a uniform random variable. Watermarked generations will instead favour tokens with larger values. We transform each value into a token-level score:

$$
s_t = -\log(1-U_{x_t})
$$

Because repeated contexts reproduce the same keyed values, the detector scores each context only once (*context deduplication*). If the text is not watermarked, the scores follow an exponential distribution with mean $1$. Their sum,

$$
S = \sum_{t=1}^{n} s_t
$$

therefore follows a Gamma distribution with shape $n$ and scale $1$. This lets us compute a one-sided p-value: the probability that unwatermarked text would produce a score at least as large as $S$. A small p-value is evidence that the text is consistent with the watermark.

![Animation of watermark detection: each token's key and preceding context reconstruct keyed noise, which becomes a token score. Scores are summed and compared with the unwatermarked Gamma distribution to obtain a p-value.](/assets/figures/2026-09-22-watermarking-in-vllm/watermark-detection-animation.gif)

Choosing a threshold trades false positives against false negatives. For a calibrated test, $p \leq 0.01$ will flag approximately 1% of unwatermarked sequences as positives. In practice, we might also test against several keys, tokenizers, or watermarking configurations. This increases the chance of finding a large score by accident, requiring us to apply a multiple-testing correction. As the number of candidates grows, the threshold becomes smaller and our detection power falls.

![Watermark detection power at 1% FPR versus distinct scored tokens for 1, 10, and 100 candidate tests. Creative writing approaches 100% TPR by about 100 tokens; MBPP rises more slowly and reaches about 69%, 49%, and 43% at 400 tokens.](/assets/figures/2026-09-22-watermarking-in-vllm/detection-power.svg)

Detection becomes stronger as independent evidence accumulates: long outputs provide more tokens to score, and high entropy steps give the watermark more opportunities to influence token selection. Short or predictable outputs therefore provide less signal.

Since the signal is encoded into the token choices themselves, copying and pasting a sufficiently long passage will preserve its evidence. Modifying the text will disrupt scores around the modified tokens, but the signal remains intact once these tokens are no longer in the recent context.

## How this was implemented in vLLM

As described in the [watermarking RFC](https://github.com/vllm-project/vllm/issues/53916), we first considered an out-of-tree implementation using `CustomLogitsProcessor`, but the approach was deemed insufficient: watermarking must run after all logits transformations, it performs sampling itself, and it must be integrated with speculative decoding. As such, we chose to integrate watermarking directly into vLLM’s final token selection step.

The [initial implementation (PR #54053)](https://github.com/vllm-project/vllm/pull/54053) connects watermarking to the Model Runner v2 sampling pipeline through `GPUWatermarkSampler`, which delegates algorithm-specific work to a `Watermarker`. This allows different schemes to share the same request and batch handling.

A naïve implementation would materialize one pseudorandom value per request and vocabulary token before computing the argmax. The resulting `[batch, vocabulary]` tensor would create substantial temporary storage and memory traffic. Mixed batches where some requests require ordinary sampling and others require watermarking would only make this worse, as we would have to materialize two such tensors and mask between them.

vLLM instead fuses the PRF, Gumbel transformation, mixed-batch handling and blockwise reduction into a single GPU kernel which never materializes the full noise tensors. We also take advantage of the specific PRF used (Philox) which generates four pseudorandom values at each invocation, which the kernel applies to four consecutive token IDs. Each GPU program reduces a block of 1,024 tokens to a single max candidate, leaving only a small final reduction.

![Decode throughput for Qwen3.5-27B across batch sizes 1–256, comparing watermarked and unwatermarked decoding with MTP on and off. Watermarking closely tracks the corresponding baseline.](/assets/figures/2026-09-22-watermarking-in-vllm/decode-throughput.svg)

## Challenges

There are some major challenges with watermarking that required particularly interesting solutions. We will discuss two of the most important below: interaction with speculative decoding, and reduction in output diversity.

### Speculative decoding

Speculative decoding proposes tokens from a draft model and accepts them against the target distribution. A natural approach would be to watermark both models with the same keyed perturbation. However, this can reduce the acceptance rate: while watermarking preserves the distributions individually when averaging over keys, it does not preserve their overlap (which is what acceptance rate depends on).

Following the speculative-decoding construction in the SynthID-Text supplementary information [[2]](#ref-2), vLLM’s [dual-key implementation (PR #56122)](https://github.com/vllm-project/vllm/pull/56122) uses two independent keys to retain the ordinary acceptance rate: $K_D$ watermarks the draft model’s accepted tokens, and $K_T$ watermarks the target model’s residual and bonus tokens. Rejection sampling still uses the unwatermarked $p$ and $q$ distributions. We must use two independent keys to avoid correlating rejected proposals with residual sampling, which would break the lossless guarantee of speculative decoding.

The final text will contain tokens watermarked by one key or the other, so we must use both keys during scoring and combine their scores. Unfortunately, this dilutes the signal from both keys, and leads to a weakening of the detection signal.

$$
s_i = (1 - \alpha) \cdot s_{K_D} + \alpha \cdot s_{K_T}
$$

This weighting can be calibrated to better match the fraction of tokens expected to come from the target, and also the amount of signal expected from each key (accepted draft tokens may be lower entropy on average and thus contain less signal).

### Maintaining output diversity

Single-token non-distortion does not prevent correlations across a sequence: with a fixed key, a repeated context receives the same pseudorandom values and may favour the same continuation again. For example, imagine a fixed key that makes a model generate the following sequence:

```text
1 + 1 +       -> 1   context: (1, +, 1, +)
1 + 1 + 1     -> +   context: (+, 1, +, 1)
1 + 1 + 1 +   -> 1   context: (1, +, 1, +)
```

After generating `1` again, the context `(1, +, 1, +)` has appeared again. The fixed key therefore produces the same pseudorandom values that previously favoured the generation of `1`. If the model’s probabilities are also similar, or the bias is strong enough, it is likely to select `1` again: in this case, the generation could enter an infinite repetition loop.

```text
1 + 1 + 1 + 1 + 1 + (...)
```

This is a worst-case scenario, but the impact of this correlation across contexts may generally lead to lower diversity within a sequence, increased rates of degenerate behaviour, longer generations, and reduced output quality.

![Output-length survival on Qwen3.5-2B math prompts: plain watermarking produces many capped long outputs, dual-key reduces the excess, and context deduplication matches unwatermarked decoding.](/assets/figures/2026-09-22-watermarking-in-vllm/output-length-survival.svg)

vLLM avoids this problem through [generation-time context deduplication (PR #56233)](https://github.com/vllm-project/vllm/pull/56233), skipping watermarking when a context has already appeared within the current generation: this provides sequence-level non-distortion. A fused GPU kernel identifies duplicate contexts and masks those positions before watermark sampling. The detector also scores each context only once, so skipped repetitions do not weaken the watermarking signal.

We can also take advantage of the previously discussed dual-key watermarking scheme, since the randomness from whether a token will come from the target or draft distribution mitigates the impact of this bad behaviour, and improves diversity *across* sequences. More generally, we can use dual-key routing without speculative decoding [[5]](#ref-5) by randomly selecting which key to use during generation.

## Conclusion

vLLM now supports distribution-preserving Gumbel-max text watermarking with efficient GPU generation, statistical detection, speculative-decoding support, and safeguards for repeated contexts. Our evaluations show minimal performance overhead while preserving output quality and diversity.

## Acknowledgements

We thank Wassim Bouaziz, Andy Lo, Nicolò Lucchesi, Victor Paltz, Leila Saidi, and Mickaël Seznec for their thoughtful review and feedback on the article, and Lucas Wilkinson for his contributions to the implementation and review.

## Appendix

<a id="ref-1"></a>1. Ingemar J. Cox, Matthew L. Miller, Jeffrey A. Bloom, Jessica Fridrich, and Ton Kalker. “[Digital Watermarking and Steganography](https://www.sciencedirect.com/book/9780123725851/digital-watermarking-and-steganography).” 2nd ed., Morgan Kaufmann (2008).

<a id="ref-2"></a>2. Sumanth Dathathri et al. “[Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4).” *Nature* 634, 818–823 (2024). [Supplementary information, §G.3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf).

<a id="ref-3"></a>3. John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. “[A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226).” *Proceedings of the 40th International Conference on Machine Learning* (2023).

<a id="ref-4"></a>4. Scott Aaronson and Hendrik Kirchner. “[Watermarking GPT outputs](https://scottaaronson.blog/?m=202302),” 2023.

<a id="ref-5"></a>5. Tom Sander et al. “[TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection](https://arxiv.org/abs/2605.12456).” arXiv:2605.12456 (2026).
