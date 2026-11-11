---
layout: post
math: true
title: "Watermarking in vLLM"
author: "Raphaël Rialland (Mistral), Simon Veitner (Red Hat), and Tomas Ruiz (Red Hat)"
summary: "How vLLM implements distribution-preserving Gumbel-max text watermarking with efficient GPU kernels, statistical detection, speculative decoding, and repeated-context safeguards."
image: /assets/figures/2026-09-24-watermarking-in-vllm/gumbel-max-sampling.png
social_image: /assets/figures/2026-09-24-watermarking-in-vllm/gumbel-max-sampling.png
tags:
  - watermarking
  - sampling
  - speculative-decoding
  - performance
---

## What is text provenance, and why is it challenging?

Establishing the provenance of digital content is essential to build trust and accountability in the information we share. Watermarking offers a solution: it alters a work to embed *a message about it* [[1]](#ref-1), for instance its provenance. Images and audio can embed watermarks as a subtle perturbation, but these techniques are not applicable to text due to its discrete nature. Instead of trying to perturb the text itself, we influence the generation process in a way that is detectable but does not modify the expected output distribution.

Watermarking text is both a scientific and engineering challenge, and there are several requirements that we must aim to balance to have a practical solution:

### Non-distortion

Adding a watermark should not modify the expected output distribution of a model [[2]](#ref-2): it should not systematically favour certain words, writing styles, or solutions to problems.

### Robustness

Generated text can be altered in many ways: shortened, edited, mixed with other corpora, or even rephrased by another LLM. A good watermarking system aims to preserve its provenance signal even under such modification.

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

At each generation step, a language model assigns a probability to every possible next token, then samples one. This process traces a path through a tree of possible continuations: some of these paths are more likely than others, but many are valid final outputs.

Ordinary stochastic sampling draws and discards fresh random values at every step. Watermarking instead derives reproducible keyed values from the recent context and each candidate token. Because the context and selected token remain in the output, a detector with the key can reconstruct these values after generation. Watermarked text will systematically align with these values, creating a signal that accumulates over a long enough sequence.

The next section discusses how we can implement this process while still preserving the initial sampling distribution of the model, effectively allowing us to *reproducibly* choose one of the possible final outputs of the model.

## Gumbel-max Trick

The Gumbel-max trick is a technique for sampling from a categorical distribution. Suppose the model assigns a probability $p_v$ to each candidate token $v$: for every token, we draw independent uniform random variables $U_v \sim \mathcal{U}(0, 1)$ and transform them into Gumbel random variables:

$$
G_v = -\log\ (-\log\ U_v) \quad \quad G_v \sim \text{Gumbel(0,1)}
$$

The sampled token is then:

$$
v^\star = \operatorname*{argmax}_{v \in \mathcal{V}}\ (\log\ p_v + G_v)
$$

A remarkable result of this transformation is that:

$$
\mathbb{P}(v^\star = v) = p_v
$$

In other words, adding Gumbel noise to the token log-probabilities and taking their argmax produces the exact same categorical distribution as ordinary random sampling: in fact, [Model Runner v2’s standard sampling path](https://github.com/vllm-project/vllm/blob/f92b78f6ef9c9b28f60668da77af5b65645b1a45/vllm/v1/worker/gpu/sample/sampler.py#L341-L352) already uses this process. In practice, we can apply Gumbel noise directly to the logits, avoiding a softmax and making the computation more parallelizable.

![Gumbel-max sampling example: model probabilities for table, mat, and chair are combined with Gumbel noise. “mat” wins one draw, while 20,000 repetitions come close to recovering the original 18%, 57%, and 25% distribution.](/assets/figures/2026-09-24-watermarking-in-vllm/gumbel-max-sampling.png)

To make these random draws reproducible, we use a pseudorandom function (PRF) [[4]](#ref-4). A PRF deterministically maps a key and an input to a value that behaves like a random uniform draw: since the same inputs always produce the same output, a detector can reconstruct the values used during generation. We replace each independent uniform draw $U_v$ with:

$$
U_v = \operatorname{PRF}(K; \mathbf{w}, v)
$$

where $K$ is the secret watermark key, and $\mathbf{w}$ is the recent watermarking context (the last $k$ tokens).

Including the token ID is important as it allows every token to receive different noise, the recent context (by default, we use the last 4 tokens) means that the random noise we apply is different at *almost* every generation step, and the secret key makes this pattern unpredictable for anyone who does not have it.

The generator computes these keyed values for each candidate token, and selects the token with the largest noised log-probability. Given the same key and context this choice can be reproduced, but without the key it looks like an ordinary random sample.

![Bar chart comparing Qwen3.5-27B quality with and without dual-key watermarking: GSM8K 93.0% versus 94.2%, MBPP 79.2% versus 77.2%, and IFEval 90.7% versus 91.9%, with overlapping error bars.](/assets/figures/2026-09-24-watermarking-in-vllm/quality.svg)

This gives the method a precise guarantee of non-distortion: in expectation over keys, the probability of selecting token $v$ remains exactly $p_v$. However, this alone does not guarantee that complete sequences will be distortion-free. We discuss this limitation and its solutions further in “Maintaining output diversity”.

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

Because repeated contexts reproduce the same keyed values, the detector scores each context only once. If the text is not watermarked, the scores follow an exponential distribution with mean $1$. Their sum,

$$
S = \sum_{t=1}^{n} s_t
$$

therefore follows a Gamma distribution with shape $n$ and scale $1$. This lets us compute a one-sided p-value: the probability that unwatermarked text would produce a score at least as large as $S$. A small p-value is evidence that the text is consistent with the watermark.

![Animation of watermark detection: each token's key and preceding context reconstruct keyed noise, which becomes a token score. Scores are summed and compared with the unwatermarked Gamma distribution to obtain a p-value.](/assets/figures/2026-09-24-watermarking-in-vllm/watermark-detection-animation.gif)

Choosing a threshold trades false positives against false negatives. For a calibrated test, $p \leq 0.01$ will flag approximately 1% of unwatermarked sequences as positives. In practice, we might also test against several keys, tokenizers, or watermarking configurations. This increases the chance of finding a large score by accident, requiring us to apply a multiple-testing correction. As the number of candidates grows, the threshold becomes smaller and our detection power falls.

![Watermark detection power at 1% FPR versus distinct scored tokens for 1, 10, and 100 candidate tests. Creative writing approaches 100% TPR by about 100 tokens; MBPP rises more slowly and reaches about 69%, 49%, and 43% at 400 tokens.](/assets/figures/2026-09-24-watermarking-in-vllm/detection-power.svg)

Detection becomes stronger as independent evidence accumulates: long outputs provide more tokens to score, and high entropy steps give the watermark more opportunities to influence token selection. Short or predictable outputs therefore provide less signal.

Since the signal is encoded into the token choices themselves, copying and pasting a sufficiently long passage will preserve its evidence. Modifying the text will disrupt scores around the modified tokens, but the signal remains intact once these tokens are no longer in the recent context of following tokens.

## How this was implemented in vLLM

The implementation was initially presented in the [watermarking RFC](https://github.com/vllm-project/vllm/issues/53916), which highlighted that watermarking must integrate with final token sampling and speculative decoding. We therefore integrated it directly into Model Runner v2’s sampling pipeline. The [initial implementation (PR #54053)](https://github.com/vllm-project/vllm/pull/54053) connects `GPUWatermarkSampler` to an algorithm-specific `Watermarker`, allowing different schemes to share the same request and batch handling.

A naïve implementation would materialize a `[batch, vocabulary]` noise tensor, creating substantial temporary storage and memory traffic. vLLM instead fuses pseudorandom-number generation, the Gumbel transformation, and argmax reduction into a single GPU kernel. Philox generates four values per invocation, allowing the kernel to process four consecutive token IDs together.

A per-row mask lets the same fused sampler use keyed noise for watermarked requests and ordinary randomness for unwatermarked requests or repeated contexts.

![Decode throughput for Qwen3.5-27B with MTP-3 across batch sizes 1–256. The watermarked and unwatermarked curves closely overlap throughout, showing no consistent throughput change.](/assets/figures/2026-09-24-watermarking-in-vllm/decode-throughput.svg)

Across eight keys, mean matched throughput changes ranged from −1.1% to +2.0% across batch sizes, with no significant slowdown.

## Challenges

There are some major challenges with watermarking that required particularly interesting solutions. We will discuss two of the most important below: interaction with speculative decoding, and reduction in output diversity.

### Speculative decoding

Speculative decoding proposes tokens from a draft distribution and accepts them against a target distribution. Applying the same watermark to both distributions preserves them individually, but can reduce their overlap and lower the acceptance rate. vLLM instead uses a [dual-key implementation (PR #56122)](https://github.com/vllm-project/vllm/pull/56122): $K_D$ is used for accepted draft tokens, while $K_T$ is used for target residual and bonus tokens. Separating the keys preserves the ordinary acceptance rate and keeps target sampling independent of rejected draft proposals.

The final text will contain tokens watermarked by one key or the other, so we must use both keys during scoring and combine their scores. Unfortunately, this dilutes the signal from both keys, and leads to a weakening of the detection signal.

$$
s_i = (1 - \alpha) \cdot s_{K_D} + \alpha \cdot s_{K_T}
$$

This weighting can be calibrated to reflect both the expected source of each token and how much signal it carries. For example, draft tokens may be accepted more often when they are “easy” and therefore lower entropy, meaning they carry less watermarking signal.

### Maintaining output diversity

Single-token non-distortion only guarantees the distribution at each generation step, not over a complete sequence. If a context repeats, so does its keyed noise, correlating token choices that ordinary sampling would make independently (see example below). Single-sequence non-distortion is a stronger guarantee that the distribution over complete sequences is also unchanged.

For example, imagine a fixed key that makes a model generate the following sequence:

```text
1 + 1 +       -> 1   context: (1, +, 1, +)
1 + 1 + 1     -> +   context: (+, 1, +, 1)
1 + 1 + 1 +   -> 1   context: (1, +, 1, +)
```

After generating `1` again, the context `(1, +, 1, +)` has repeated. The fixed key therefore produces the same pseudorandom values that previously favoured the generation of `1`. If the model’s probabilities are also similar, or the bias is strong enough, it is likely to select `1` again: in this case, the generation could enter an infinite repetition loop.

```text
1 + 1 + 1 + 1 + 1 + (...)
```

This is a worst-case scenario, but the impact of this correlation across contexts may generally lead to lower diversity within a sequence, increased rates of degenerate behaviour, longer generations, and reduced output quality.

![Output-length survival on Qwen3.5-2B math prompts: plain watermarking produces many capped long outputs, dual-key reduces the excess, and context deduplication matches unwatermarked decoding.](/assets/figures/2026-09-24-watermarking-in-vllm/output-length-survival.svg)

vLLM addresses this with [generation-time context deduplication (PR #56233)](https://github.com/vllm-project/vllm/pull/56233), which skips watermarking whenever a context repeats and preserves single-sequence non-distortion. Despite checking the preceding generation for repeated contexts at every step, it reduced end-to-end throughput by at most 0.19% on Qwen3.5-27B.

We can also take advantage of the previously discussed dual-key watermarking scheme, since the randomness from whether a token will come from the target or draft distribution mitigates the impact of this bad behaviour, and improves diversity *across* sequences. More generally, we can use dual-key routing without speculative decoding [[5]](#ref-5) by randomly selecting which key to use during generation.

## Getting started

You can enable Gumbel-max watermarking when starting a vLLM server:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --watermark-config '{"algorithm":"gumbel","key":42}'
```

vLLM also provides a [minimal HTTP detection server example](https://github.com/vllm-project/vllm/blob/main/examples/basic/online_serving/watermark_detection_server.py) for checking text using the matching tokenizer and watermark key. See the [watermarking documentation](https://docs.vllm.ai/en/latest/features/watermarking/) for configuration and detection examples.

## Conclusion

vLLM now supports distortion-free watermarking using the Gumbel-max algorithm, with efficient GPU generation and speculative decoding support. Our evaluations show minimal performance overhead while preserving output quality and diversity.

## Acknowledgements

We thank Wassim Bouaziz, Andy Lo, Nicolò Lucchesi, Victor Paltz, Leila Saidi, and Mickaël Seznec for their thoughtful review and feedback on the article, and Lucas Wilkinson for his contributions to the implementation and review.

## Appendix

### Throughput measurements

The measurements use Qwen3.5-27B on one H100 with MTP-3 and 512 output tokens. Throughput values are medians across eight watermarked keys, and four unwatermarked controls.

| Batch size | Unwatermarked median (tok/s) | Watermarked median (tok/s) | Mean matched change | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 113.9 | 113.9 | −0.23% | [−1.00%, +0.53%] |
| 2 | 218.2 | 218.5 | −0.30% | [−2.87%, +1.08%] |
| 4 | 420.5 | 413.7 | −1.11% | [−2.36%, +0.14%] |
| 8 | 753.9 | 767.3 | +0.77% | [−0.99%, +2.72%] |
| 16 | 1364.8 | 1358.7 | +0.02% | [−1.06%, +1.61%] |
| 32 | 1378.8 | 1407.9 | +2.03% | [+0.59%, +3.46%] |
| 64 | 1439.9 | 1427.3 | −1.10% | [−2.59%, +0.77%] |
| 128 | 1568.5 | 1574.5 | +0.36% | [−1.14%, +2.04%] |
| 256 | 1590.7 | 1586.5 | −0.58% | [−1.92%, +0.77%] |

### References

<a id="ref-1"></a>1. Ingemar J. Cox, Matthew L. Miller, Jeffrey A. Bloom, Jessica Fridrich, and Ton Kalker. “[Digital Watermarking and Steganography](https://www.sciencedirect.com/book/9780123725851/digital-watermarking-and-steganography).” 2nd ed., Morgan Kaufmann (2008).

<a id="ref-2"></a>2. Sumanth Dathathri et al. “[Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4).” *Nature* 634, 818–823 (2024). [Supplementary information, §G.3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf).

<a id="ref-3"></a>3. John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. “[A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226).” *Proceedings of the 40th International Conference on Machine Learning* (2023).

<a id="ref-4"></a>4. Scott Aaronson and Hendrik Kirchner. “[Watermarking GPT outputs](https://scottaaronson.blog/?m=202302),” 2023.

<a id="ref-5"></a>5. Tom Sander et al. “[TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection](https://arxiv.org/abs/2605.12456).” arXiv:2605.12456 (2026).
