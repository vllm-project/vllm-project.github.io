# Bitwise-exact Batch Invariant On-Policy Reinforcement Learning with vLLM and TorchTitan

In the septillions of flops used to pre-train models, this mismatch between values has largely been avoidable.  Pre-training typically runs at a fixed batch size which induces the same reduction kernels to be run - often side-stepping the issue entirely.

Reinforcement learning, on the other hand, seems to almost exclusively run different reduction algorithms due to its inference-heavy (and thus largely latency and memory-bound) nature.  Kernels optimized for low-batch size inference typically run reductions all at once, whereas kernels for training models parallelize heavily to reuse data and amp up compute utilization.  That means the generators and the trainers are typically running completely different kernels!

So intuitively, why might this be an issue?  A rudimentary explanation is that the training becomes implicitly “off-policy” because the outputs from the generator do not match the outputs a trainer might produce given the same inputs.

Discussion on this can be found on ThinkingMachine’s post Defeating Nondeterminism in LLM Inference ([He et al.](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)) and the post Your Efficient RL Framework Secretly Brings You Off-Policy RL Training ([Yao, Liu et al.](https://fengyao.notion.site/off-policy-rl)).

## Background

Floating point numbers are effectively a binary scientific notation.  They utilize three components: a sign bit (s), a mantissa (M) and an exponent (e).
<p align="center">
  <img width="340" height="130" alt="Screenshot 2025-11-10 at 5 12 41 PM" src="https://github.com/user-attachments/assets/24275084-1b8c-45fd-b40c-6169ed04c837" />
</p>

Each of these components are represented as integers and suffer from the exact same rounding errors you might expect.  In bf16, the most commonly used representation for machine learning, 7 bits are dedicated to the mantissa.  This is not very many bits!  The value 3.0 can be represented exactly, but a value like 3.6 cannot…

<p align="center">
<img width="480" height="355" alt="Screenshot 2025-11-10 at 5 13 24 PM" src="https://github.com/user-attachments/assets/1a51da11-b0b4-45fb-853d-bc19a23c1300" />
</p>

When you want a new value in bf16 you end up rounding it to the nearest available value. What’s of particular interest today is the implication of this rounding process happening at different points in a sequence of additions.

<img width="944" height="414" alt="Screenshot 2025-11-10 at 5 13 56 PM" src="https://github.com/user-attachments/assets/aa334e61-778a-4a18-ab11-e88bd202d7d2" />

These rounding steps can cause two of the exact same inputs to generate *different* outputs!  That means the same framework on the same hardware with the same inputs and the same weights can produce distinct outputs if *any* of the logic *anywhere* in the execution dispatches a different (but still correct) kernel.

## Demonstration

Reinforcement learning has been shown to amplify tiny numerical perturbations, leading to non-deterministic and unstable training behavior.  By combining the [recent work](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/deterministic_vllm_rl) of vLLM with TorchTitan we were able to demonstrate the stabilized training dynamics of reinforcement learning with exact bitwise parity between generator and trainer.  This has been landed as a script in TorchTitan [here](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/deterministic_vllm_rl/simple_rl.py).

<img width="1051" height="430" alt="Screenshot 2025-11-10 at 5 14 45 PM" src="https://github.com/user-attachments/assets/6cb38cab-89d4-409f-8abf-db1aeb1e24f2" />

The script will download and run an RL fine-tune of Qwen3 1.7B locally and plot the reward and entropy in tensorboard.

<img width="1365" height="668" alt="Screenshot 2025-11-10 at 5 16 45 PM" src="https://github.com/user-attachments/assets/86ae5415-8429-403d-8473-180c7d1cfe0b" />

Running the demonstration associated with this blog post we see exactly the issue described below.  Running the generator with different kernels than the trainer (batch_inv_OFF) shows a reduced reward over 100 steps.  Enabling bitwise exact training, we see the model not only train in fewer steps, but reach a higher total reward!

<img width="1319" height="473" alt="Screenshot 2025-11-10 at 5 17 16 PM" src="https://github.com/user-attachments/assets/f2c9d6aa-68c2-4064-b4ab-de425f2b78a7" />


## How It’s Done & What’s Next

We tackled not only invariance in the same framework, but across two different frameworks.  This was a challenging task as it required effectively auditing every single invocation of every kernel.  We heavily leveraged the forward pass kernels from vLLM’s [recent batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) work and wrote simple backward passes for these.

Then, we wrote a generic reinforcement learning script using GSM8K and a correctness reward.  We run everything synchronously, alternating between trainer and generator on a single host.  This is demonstrative of exactly on-policy execution, but is not very common in large scale runs.

While building this, testing was straightforward as we are able to use exact bitwise checks to ensure the forward logprobs and the perplexity generated by the trainer are identical.  We will continue to improve the performance of vLLM and simplify the integration to support all TorchTitan models.  To follow this work, please see the linked RFC: [#28326](https://github.com/vllm-project/vllm/issues/28326).

Acknowledgements
Bram Wasti, Teja Rao, Paul Zhang, Tianyu Liu, Zhuohan Li, Natalia Gimelshein
