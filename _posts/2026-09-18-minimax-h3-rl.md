---
layout: post
title: "Audio-Video Reinforcement Learning Post-Training for MiniMax-H3 with VeRL-Omni and vLLM-Omni"
author: "VeRL-Omni Team"
summary: "How DiffusionNFT, VeRL-Omni, and vLLM-Omni enable online RL post-training for MiniMax-H3 T2VA and FL2VA while addressing rollout performance and train-inference consistency."
image: /assets/logos/vllm-logo-text-light.png
tags:
  - multimodal
  - rlhf
  - ecosystem
  - performance
  - vllm-omni
published: true
---

## Introduction

MiniMax-H3 is a general-purpose multimodal generative model that accepts text, image, video, audio, and other inputs. It directly outputs video with a native stereo soundtrack, supporting durations of up to 15 seconds, resolutions up to 2K, 24 FPS, and 32 kHz stereo audio. Its weights are open-sourced on Hugging Face. RL post-training can further improve its capabilities in vertical domains.

For joint audio-video generative models, however, the hardest part of RL post-training is not choosing an algorithm or designing a reward. It is confirming that the entire pipeline is actually correct. A runnable script, a numerical loss, and a moving reward curve do not prove that training is heading in the right direction. **In joint audio-video settings, a problem can silently occur at any stage while the training curve continues to rise steadily—and that is the most treacherous part.**

This article documents our post-training work for MiniMax-H3 in VeRL-Omni and vLLM-Omni. Using DiffusionNFT as an example, we completed two online RL training loops: T2VA (text-to-audio-video) and FL2VA (first/last-frame-to-audio-video). The responsibilities are divided as follows: vLLM-Omni handles rollout, meaning audio-video sampling; Diffusers / FSDP2 trains the actor; and VeRL-Omni connects data, reward, NFT loss, and old-policy LoRA synchronization into a complete pipeline.

For joint audio-video RL post-training, rollout concentrates two major challenges. First, audio-video generation is computationally expensive. Rollout accounts for most end-to-end time, so its performance largely determines training efficiency. Second, the training engine (Diffusers / FSDP2) and inference engine (vLLM-Omni) differ in areas such as weight sharding, LoRA naming, and timestep value ranges. Rollout's critical configuration and scheduling therefore require dedicated handling to ensure that training and sampling use the same policy. This article follows those two themes: how to use vLLM-Omni's high-throughput rollout and parameter synchronization to make MiniMax-H3's joint audio-video online RL loop work correctly in VeRL-Omni.

## 1. Background: MiniMax-H3 and DiffusionNFT

### 1.1 MiniMax-H3: Joint Audio-Video Generation

MiniMax-H3 does not generate a video tensor alone. It produces both a video latent and an audio latent, which are ultimately decoded together into a video with a soundtrack. This characteristic imposes two requirements on RL training:

- The reward must evaluate both the visuals and the sound rather than focusing on only one modality.
- The training framework must ensure that neither audio nor video silently disappears between rollout, reward, data persistence, and training batches.

H3's DiT backbone also differs from common conventions in several ways. It is CFG-distilled, so inference does not require a negative prompt; its timestep uses a data fraction rather than the sigma commonly used in flow matching; and its velocity sign is opposite to the common convention. These details are hidden pitfalls when integrating H3 into a training framework, as discussed later.

### 1.2 How DiffusionNFT Works

DiffusionNFT stands for Diffusion Negative-aware FineTuning, and its paper is titled **DiffusionNFT: Online Diffusion Reinforcement with Forward Process**. It is an online RL method for diffusion / flow-matching models. Instead of estimating the policy gradient along the reverse sampling chain, it injects the reward signal into a supervised flow-matching objective in the forward diffusion process. The rollout side retains only the final generated clean latent, prompt embedding, and training timestep. The training side then applies forward noising again and converts reward into reward probability, moving high-reward samples closer to the positive target while pulling low-reward samples in the opposite direction.

We integrated DiffusionNFT first because the most important initial task when adding H3 was to verify three fundamental facts:

- Whether vLLM-Omni rollout truly generated **both** video and audio.
- Whether the CLAP / ImageBind reward truly received the complete audio and video.
- Whether an actor's updated LoRA was truly synchronized back to the next rollout.

DiffusionNFT does not require recording every transition's log-prob on the rollout side, so its pipeline is shorter and these three points can be verified one by one. We later integrated FL2VA as well. T2VA and FL2VA now share the same rollout, reward, actor, and LoRA synchronization backbone; they differ only in conditional-frame input and the scope of the training-loss mask.

The following diagram combines both task paths. The upper section shows the conditional inputs and H3 rollout for T2VA / FL2VA; the middle shows the audio-visual rewards and training-data format; and the lower section shows DiffusionNFT forward-process optimization and old-policy refresh. Two data flows must be distinguished: **decoded video/audio used for scoring flows to CLAP and ImageBind, while clean latents, timestep, and condition metadata used for actor updates flow to FSDP2.**

![System architecture and main data flows](/assets/figures/2026-09-18-minimax-rl/image.png)

The diagram shows the system layers and main data flows. DiffusionNFT currently uses **global-standard-deviation reward normalization** by default. Section 4.2 describes the rollout-policy update details.

## 2. Why Rollout Is Critical to Audio-Video RL Post-Training

For RL post-training of a joint audio-video generative model, rollout concentrates two risks: **compute cost** and **train-inference consistency**.

### 2.1 Compute Cost: Audio-Video Generation Makes Rollout the Time Bottleneck

Unlike text or image-only RL, every joint audio-video rollout must execute H3's complete denoise loop, decode video frames, and synthesize a 32 kHz stereo soundtrack. Generating a single sample costs far more than generating text. DiffusionNFT also requires multiple rollouts for the same prompt (`ROLLOUT_N=16` by default) to create within-group reward differences, further multiplying the total rollout workload.

The timing breakdown in Section 7.3 shows that when end-to-end time is divided into rollout, reward, actor update, and checkpoint, rollout and reward are the two largest components. The reward stage must also perform audio decoding, video processing, and two scorers. **Rollout is the primary generation bottleneck, while reward is the accompanying bottleneck in joint audio-video settings; end-to-end optimization must match the throughput of both.**

### 2.2 Train-Inference Consistency: Different Training and Inference Engines

In H3's online RL pipeline, the vLLM-Omni inference engine performs rollout, while Diffusers / FSDP2 performs actor training. The two engines differ in several respects:

- **Weight layout:** The training side sees separate attention projections—`to_q / to_k / to_v / to_out.0`—while H3 in vLLM-Omni combines them into a fused DiT.
- **LoRA naming:** Diffusers LoRA module names do not map one-to-one to vLLM-Omni's fused structure. Without a mapping, an adapter can register successfully while matching zero layers in practice.
- **Timestep / velocity conventions:** H3's diffusion timestep uses a data fraction rather than the sigma used by generic flow matching, and its velocity sign is opposite to the common convention. If the training side passes values according to the generic convention, the loss can still be numerical while the entire gradient direction is wrong.

These differences share a dangerous trait: **they do not raise errors**. The loss has a value, the adapter registers successfully, and the curve moves, but the gradient direction may be reversed or the update may have no effect at all. Rollout's critical configuration and scheduling—especially parameter synchronization from the training adapter to the rollout adapter, fused DiT mapping, target-module validation, and old-policy refresh—must therefore be handled explicitly. This is the subject of Section 4: configuring and scheduling train-inference consistency.

## 3. Rollout Performance Optimization

H3 integrates relatively smoothly into VeRL-Omni because several core modules in the repository are cleanly decoupled. vLLM-Omni handles generation, VeRL-Omni handles training orchestration, and the H3 adapter absorbs model-specific conventions: how to split packed latents, convert timesteps, handle the velocity sign, and map LoRA names. Most model-specific behavior is contained in the pipeline adapter, so the generic training loop does not need to know every H3 detail. Rollout is delegated to vLLM-Omni instead of the training engine precisely because rollout is the throughput bottleneck. vLLM-Omni's inference optimizations—continuous batching, fused kernels, and tensor parallelism—substantially reduce per-rollout time, while the training engine (Diffusers / FSDP2) was not designed for high-throughput generation.

### 3.1 Balancing Tensor Parallelism Between Throughput and Synchronization

The rollout tensor-parallel degree affects both throughput and whether train-inference synchronization can run, requiring a tradeoff:

| Task | Rollout TP | Description |
|---|---:|---|
| T2VA | 2 | Default configuration |
| FL2VA | 4 | Uses TP=4 to reserve GPU memory for actor-to-rollout synchronization |

FL2VA's conditional-frame protocol and frame-pinning logic consume additional GPU memory, so rollout TP is increased from 2 to 4 to leave headroom for actor-to-rollout LoRA synchronization.

### 3.2 Coordinating Rollout and Reward Throughput

End-to-end time can be divided into rollout, reward, actor update, and checkpoint. In joint audio-video training, the reward stage must decode audio, process video, and run two scorers, so it can become a bottleneck just like rollout. Optimizing training efficiency therefore requires looking beyond actor GPU utilization.

The number of reward workers is set to 1 to prevent multiple processes from filling the first GPU. This is a GPU-memory and compute tradeoff when reward and rollout share one GPU. Rollout and reward throughput must be planned together: if reward is slow, sampled rollouts accumulate and the actor waits for data; if reward is fast but rollout is slow, the actor remains idle. Section 7.3 presents the actual timing breakdown for each stage.

## 4. Train-Inference Consistency: Critical Rollout Configuration and Scheduling

### 4.1 Preserving H3's Native Denoise Loop

vLLM-Omni rollout preserves H3's native denoise loop, so the training side does not need to reproduce H3's sampling process. This is the first safeguard for train-inference consistency. If training independently reproduced sampling, any difference in scheduler, timestep, or velocity convention could silently split the generation distributions of training and sampling. Preserving the native loop means that the same code controls how sampling proceeds, while training only receives the result.

### 4.2 LoRA Parameter Synchronization and Old-Policy Refresh

After actor training, the LoRA weights must be synchronized back to the old rollout policy; otherwise, the next rollout still uses the old strategy. Synchronization is configured through `policy_state_adapters`, with both `default` and `old` adapters maintained:

```bash
actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
algorithm.old_policy_update_interval=2 \
```

The `old` policy is not updated with an unconditional hard copy. Instead, it uses a delayed linear decay schedule (`delayed_linear_to_0_999`) and refreshes every two training steps. This schedule lets the old policy smoothly follow the training policy, avoiding distribution jumps caused by a hard copy at every step. It also controls the policy divergence between training and sampling: if they diverge too far, ref KL becomes ineffective and train-inference consistency can no longer be maintained. This synchronization is central to train-inference consistency. The training adapter updates in the Diffusers namespace, while the rollout adapter takes effect in vLLM-Omni's fused namespace, so the two must be mapped as described in Section 4.3.

### 4.3 Fused DiT Mapping and Target-Module Validation

The training side sees separate attention projections, while H3 in vLLM-Omni fuses them into a single structure. Without the mapping, the adapter can register successfully but match zero layers in practice, so the next rollout still behaves like the base model.

The target modules must therefore be listed explicitly rather than using all-linear:

```bash
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=128 \
actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"
```

**H3 cannot directly use all-linear for its target modules.** Nor can every training-side LoRA be synchronized to the rollout side; fused QKV, for example, requires explicit mapping. The adapter validates target modules in advance and aborts with an error when they are misconfigured, preventing training from silently going off course. This is the final gate for train-inference consistency: any error in fused DiT mapping or LoRA naming appears during validation rather than remaining hidden in the training curve.

### 4.4 Converting Timestep / Velocity Conventions

H3's diffusion timestep uses a data fraction—the data proportion, or interpolation ratio from data to noise—rather than the sigma used in generic flow matching. Its velocity sign is also opposite to the common convention. During integration, the H3 adapter only needs to convert the timestep according to the native convention and normalize the velocity sign. Otherwise, the loss may still have a value while the gradient direction is wrong.

## 5. Experimental Design and Configuration

The experiment's four questions all concern whether data, conditions, and configuration have been silently mismatched. This section brings together task definitions, the training-data format returned by rollout, the conditional-frame protocol, and reproducible configuration.

### 5.1 Experimental Goals and Scope

The experiments in this article do not pursue a horizontal comparison for benchmark SOTA. The first stage is more fundamental and easier to overlook: **we need to confirm that H3's joint audio-video data, reward signals, and policy updates are not silently mismatched within the same online RL pipeline.**

The experiments therefore focus on four questions:

1. **Is the joint output complete?** Do both the video and audio returned by rollout enter persisted data, reward, and the training-data format?
2. **Does the reward truly constrain both audio and video?** Does CLAP receive both text and audio, and does ImageBind receive the audio and video from the same sample?
3. **Does the update reach the sampling policy?** After the actor-side LoRA update, does the old rollout policy refresh according to the intended schedule?
4. **Are conditional frames handled correctly?** Input keyframes in FL2VA must remain conditions and must not be reoptimized by the DiffusionNFT objective.

This also determines how results should be interpreted: reward curves, component rewards, fixed-sample videos, and training logs must be considered together. An increase in any single metric is insufficient to prove an overall improvement in generation quality.

### 5.2 Two Tasks: T2VA and FL2VA

| Task | Model Input | Conditional Constraint | Output Optimized During Training |
|---|---|---|---|
| **T2VA** (text-to-audio-video) | Text prompt | No first frame and no negative prompt; H3 itself is CFG-distilled | All generated video / audio latents |
| **FL2VA** (first/last-frame conditioned text-image-to-audio-video) | Text prompt + one or two keyframes | First frame, last frame, or both first and last frames | Generated video and audio latents; conditional-frame latents remain fixed |

T2VA tests whether text, motion, and audio-visual semantics are aligned. FL2VA goes further by testing whether the model can fill in intermediate motion and the soundtrack under given keyframe constraints, which is suitable for tasks such as character consistency, shot transitions, and advertising-asset extension.

Both tasks share the rollout, reward, actor, and LoRA synchronization backbone. They differ in conditional input and the scope of the training-loss mask. FL2VA rollout uses vLLM-Omni's official first/last-frame protocol: **latents corresponding to conditional images remain fixed, while the DiffusionNFT forward-process objective applies only to generated video and audio latents**. This determines whether the model learns completion or incorrectly rewrites keyframes. The data format also includes conditional-frame segment information and `frame_indices`, which the actor uses to fix conditional-frame latents. `FRAME_INDICES` must correspond to the `frame_mode` used during data conversion; Section 6.2 lists the exact launch values.

### 5.3 Training-Data Format: Joint Packing and Pipeline Constraints

To verify complete joint output, the fields returned by rollout must support both actor updates and audio-visual reward. The rollout side must return the state required for DiffusionNFT training:

```python
rl = {
    "latents_clean": pack_video_audio_rows(video_rows, audio_rows),
    "train_timesteps": train_timesteps,
    "latent_meta": latent_meta,
}
```

Rollout is responsible for generation and packing, while the actor is responsible for unpacking and training. Any missing or misaligned field gives the training side an incorrect target.

`latents_clean` contains concatenated video and audio latents, while `latent_meta` guides their separation on the training side. Audio is the modality most likely to disappear silently in the middle of the pipeline: many diffusion / video training frameworks process visuals by default, but H3 outputs a joint audio-video result. **The reward pipeline is closed only when audio actually reaches CLAP / ImageBind and the exported video also includes a soundtrack.**

Frame count and resolution in the recipe are also constrained by the pipeline. The pipeline aligns `NUM_FRAMES=96` to 107 frames, which must satisfy the 17n+5 boundary for video durations of 4–15 seconds at 24 FPS. The sampled side length must be a multiple of 32; otherwise, the H3 pipeline silently rounds it down. Section 5.5 lists these defaults.

### 5.4 Dataset and Conditional Images

The T2VA data format is a prompt-only Parquet file. The converter accepts two input formats: a plain-text file with one prompt per line, or JSONL records containing a prompt field. It outputs separate training and test Parquet files. Section 6.1 provides the conversion command.

FL2VA uses the ConsisID prompt list released by DanceGRPO to construct conditional-image data. For every prompt, it generates a FLUX.1-dev reference image using a fixed seed plus the index, then pairs the prompt and image with that same index. This makes interruption recovery, train-test splits, and conditional frames reproducible. After splitting with seed 42, the dataset contains 27,687 training samples and 128 test samples. The recipe provides the exact conditional-image generation, JSONL splitting, and Parquet conversion commands. Conditional images can be reused directly; the rollout pipeline resizes them at runtime with LANCZOS.

### 5.5 Reproducible Training Configuration

The following table lists key defaults from the current main-branch recipe. These parameters have passed end-to-end verification and can serve directly as a starting point for reproducing the experiment.

| Configuration | T2VA Recipe | FL2VA Recipe |
|---|---|---|
| Number of GPUs | 8 | 8 |
| Rollout TP | 2 | 4 |
| Rollouts per prompt, n | 16 | 16 |
| Training / validation resolution | 256x384 / 512x768 | 288x448 / 576x928 |
| Frame count | 121 | `NUM_FRAMES=96`, aligned by the pipeline to 107 frames |
| Rollout / validation steps | 10 / 40 | 10 / 40 |
| LoRA | rank 64, alpha 128 | rank 64, alpha 128 |
| Old-policy update | Delayed linear decay, refreshed every 2 training steps | Same as T2VA |
| Reward | CLAP + ImageBind audio-video | CLAP + ImageBind audio-video |

### 5.6 Evaluation Metrics and Checks

The training process records four categories of information:

- **Optimization statistics:** loss, gradient norm (grad norm), reward probability, and reference KL (ref KL).
- **Reward breakdown:** CLAP, ImageBind, and weighted reward.
- **Fixed-sample comparison:** compare the base model and updated policy using the same prompt, seed, and inference configuration.
- **System-pipeline checks:** whether audio reaches CLAP / ImageBind at 32 kHz, whether exported MP4 files contain an AAC soundtrack, and whether LoRA matches the layers actually executed during rollout.

FL2VA additionally requires checking conditional-frame preservation: whether first-frame identity is maintained, whether the prompt's scene is established, and whether temporal consistency is maintained throughout the video.

## 6. Training Workflow

### 6.1 T2VA Data Preparation and Launch

First, convert raw prompt-only data to Parquet:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2va_data.py \
  --input_dir /path/to/raw_prompts \
  --output_dir /path/to/h3_t2va_data
```

Then launch training:

```bash
export MODEL_PATH=/path/to/MiniMax-H3
export DATA_DIR=/path/to/h3_t2va_data

NUM_GPUS=8 ROLLOUT_TP=2 ROLLOUT_N=16 INFER_STEPS=10 \
TOTAL_TRAINING_STEPS=1000 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

`MODEL_PATH` points to the local MiniMax-H3 root directory. It must contain `FL2VA/` for rollout—the vLLM-Omni checkpoint directory shared by T2VA and FL2VA—and `transformer/` for training.

### 6.2 Launching FL2VA Training

FL2VA uses a separate image-conditioned rollout adapter and agent loop. It cannot use the T2VA script with only the data directory changed. Confirm that `DATA_DIR` already contains `train.parquet` and `test.parquet`, and set `MODEL_PATH` to the MiniMax-H3 root containing both `FL2VA/` and `transformer/`.

The shortest launch command for first-frame-conditioned training is:

```bash
export MODEL_PATH=/path/to/MiniMax-H3
export DATA_DIR=/path/to/h3_fl2va

FRAME_INDICES='[0]' \
NUM_GPUS=8 ROLLOUT_TP=4 ROLLOUT_N=16 INFER_STEPS=10 \
TOTAL_TRAINING_STEPS=1000 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

`FRAME_INDICES` must correspond to the `frame_mode` used during data conversion: use `'[0]'` for first-frame data, `'[-1]'` for last-frame data, and `'[0,-1]'` for first-and-last-frame data. To switch condition modes, reconvert the Parquet file and pass the corresponding `FRAME_INDICES`:

```bash
# First-and-last-frame conditioning: prepare_fl2va_data.py --frame_mode first_last
FRAME_INDICES='[0,-1]' \
MODEL_PATH=/path/to/MiniMax-H3 DATA_DIR=/path/to/h3_fl2va_first_last \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

During validation, check not only reward but also whether first-frame identity is preserved, whether the prompt's scene is established, and whether temporal consistency is maintained across all 107 frames.

## 7. Experimental Results

The following curves come from an online T2VA training run. Their purpose is to show whether the optimization signals and system paths behave normally, not to claim that the model already outperforms the base model on every task. FL2VA uses video comparisons under first-frame conditioning to check the conditional-frame pipeline.

### 7.1 Training Reward Curve

The mean training reward rises steadily from about 0.27 to above 0.4, showing that under the current prompt distribution, the joint CLAP + ImageBind signal can create learnable within-group preferences. This metric only indicates that the direction preferred by the reward model has been optimized. On its own, it cannot be interpreted as improved visual aesthetics or long-range consistency.

![Training reward curve](/assets/figures/2026-09-18-minimax-rl/train-reward.png)

Actor dynamics must be read together with reward. In particular, monitor gradient norm (grad norm), reward probability, and reference KL (ref KL). If reward rises while gradient norms remain abnormal, reward probability saturates, or ref KL suddenly becomes ineffective, the curve may still converge to the wrong objective.

![Actor training dynamics](/assets/figures/2026-09-18-minimax-rl/actor-training-dynamics.png)

### 7.2 Validation Reward Breakdown

CLAP, ImageBind, and weighted reward are recorded separately on the validation set. **Combined reward is interpretable only when both component rewards change in a direction consistent with the fixed-sample videos.** For example, if an increase in combined reward comes only from CLAP, the audio may align better with the text, but this does not prove that the audio-visual relationship or visual quality also improved.

![Evaluation reward curve](/assets/figures/2026-09-18-minimax-rl/eval-reward.png)

### 7.3 Training-Time Analysis: Rollout and Reward Throughput Bottlenecks

End-to-end time is divided into rollout, reward, actor update, and checkpoint.

![Time consumed by each stage](/assets/figures/2026-09-18-minimax-rl/time-consumption.png)

The breakdown shows that rollout and reward together account for most end-to-end time, while actor update and checkpoint account for relatively little. End-to-end optimization should therefore reduce per-rollout time, increase rollout throughput, and match reward throughput to it.

### 7.4 Video Comparison: Base Model vs. DiffusionNFT (T2VA)

Reward curves only reflect the overall numerical trend, so the final assessment must return to the generated results. We selected four prompts from the test set and generated audio-video outputs with both the MiniMax-H3 base model and the DiffusionNFT-fine-tuned model using the **same seed and inference configuration**. We focused on three dimensions:

- **Motion consistency:** whether object trajectories are coherent and free from jitter or discontinuities.
- **Audio-visual alignment:** whether sound effects occur in sync with on-screen actions.
- **Visual-detail stability:** whether materials, lighting, and edges remain stable throughout the duration.

The left column is the base model, and the right column is the DiffusionNFT-fine-tuned model. The original English prompts are preserved without rewriting.

| ID | Prompt | MiniMax H3 (base) | MiniMax H3 + DiffusionNFT |
|---:|---|---|---|
| 1 | stickman monigote shooting a energy sphere from his hands | ![01-stickman-base](/assets/figures/2026-09-18-minimax-rl/01-stickman-base.gif) | ![01-stickman-DiffusionNFT](/assets/figures/2026-09-18-minimax-rl/01-stickman-DiffusionNFT.gif) |
| 2 | a husky dog with sunglasses riding on santas sled | ![02-husky-base](/assets/figures/2026-09-18-minimax-rl/02-husky-base.gif) | ![02-husky-DiffusionNFT](/assets/figures/2026-09-18-minimax-rl/02-husky-DiffusionNFT.gif) |
| 3 | minimalist polygonal human skull in green flames with strong movement, uhd | ![03-skull-base](/assets/figures/2026-09-18-minimax-rl/03-skull-base.gif) | ![03-skull-DiffusionNFT](/assets/figures/2026-09-18-minimax-rl/03-skull-DiffusionNFT.gif) |
| 4 | 17th century sailing ship making a path through the waves during a storm | ![04-ship-base](/assets/figures/2026-09-18-minimax-rl/04-ship-base.gif) | ![04-ship-DiffusionNFT](/assets/figures/2026-09-18-minimax-rl/04-ship-DiffusionNFT.gif) |

### 7.5 Video Comparison: Base vs. DiffusionNFT Under FL2VA First-Frame Conditioning

T2VA constrains only the semantic relationship between text and audio-video, while FL2VA also requires the model to follow a given first-frame image. The comparison therefore asks more than whether the visuals look good: **with the first frame fixed, can the model complete the subsequent motion and soundtrack more coherently?** We selected two prompts from the test set and used the same FLUX.1-dev first frame as the condition. Under the **same seed and inference configuration**, we continued the audio-video with both the base model and the FL2VA DiffusionNFT-fine-tuned model.

- **First-frame consistency:** whether the generated result continues faithfully from the given conditional image instead of redrawing a new scene.
- **Motion consistency:** whether motion unfolds coherently from the first frame without jitter or drift.
- **Audio-visual alignment:** whether effects such as dripping and water sounds occur in sync with on-screen actions.

The second column shows the conditional first frame, and the two middle output columns show the base model and FL2VA DiffusionNFT-fine-tuned model, respectively. The original English prompts are preserved without rewriting; their accompanying Chinese descriptions have been translated into English.

| ID | Conditional First Frame | Prompt | MiniMax H3 (base) | MiniMax H3 + FL2VA DiffusionNFT |
|---:|---|---|---|---|
| 23 | ![Conditional first frame showing a red bottle](/assets/figures/2026-09-18-minimax-rl/fl2va-23-bottle-condition.jpg) | Shows a close-up of a woman holding a red bottle with a blue substance dripping from it.<br />A close-up of a woman holding a red bottle, with blue liquid dripping from it. | ![fl2va-23-bottle-base](/assets/figures/2026-09-18-minimax-rl/fl2va-23-bottle-base.gif) | ![fl2va-23-bottle-DiffusionNFT](/assets/figures/2026-09-18-minimax-rl/fl2va-23-bottle-DiffusionNFT.gif) |
| 56 | ![Conditional first frame showing a pool](/assets/figures/2026-09-18-minimax-rl/fl2va-56-pool-condition.jpg) | Shows a man wearing a white shirt, brown apron, and a white hat standing in a pool filled with water.<br />A man wearing a white shirt, brown apron, and white hat stands in a pool filled with water. | ![fl2va-56-pool-base](/assets/figures/2026-09-18-minimax-rl/fl2va-56-pool-base.gif) | ![fl2va-56-pool-DiffusionNFT](/assets/figures/2026-09-18-minimax-rl/fl2va-56-pool-DiffusionNFT.gif) |

These are qualitative comparisons from the FL2VA integration-validation stage, and the sample size is limited. They show only that the conditional-frame pipeline and post-training updates produce visible differences in generated results; they are not a comprehensive benchmark of conditional-generation quality.

## 8. Lessons Learned

This article follows two themes: rollout performance and train-inference consistency. The key lessons from each are summarized below.

### 8.1 Rollout Performance: Coordinating Rollout and Reward Throughput

- **Rollout is the primary generation bottleneck, while reward is the accompanying bottleneck in joint audio-video settings.** End-to-end optimization must match their throughput. The reward stage handles audio decoding, video processing, and two scorers, so actor GPU utilization alone is insufficient.
- **TP configuration must balance throughput against GPU memory for actor-to-rollout synchronization.** T2VA uses TP=2; FL2VA increases it to TP=4 because conditional frames consume additional GPU memory, leaving headroom for LoRA synchronization.

### 8.2 Train-Inference Consistency: Silent Errors That Run Successfully but Incorrectly

When integrating H3, the most dangerous errors are not those that crash the program. They are **errors that run successfully but produce incorrect behavior**: the loss has a value and the curve moves, but the gradient direction is reversed or the updated weights have no effect. Nearly all such errors occur in the train-inference consistency pipeline. We investigated them in rollout-pipeline order:

| Stage | Surface Symptom | Actual Problem | Solution |
|---|---|---|---|
| Training-adapter integration | Loss has a value | H3's timestep / velocity definitions are opposite to generic flow matching | Convert timestep according to H3's convention and normalize the velocity sign |
| Rollout-adapter integration | Can generate audio and video | Training and rollout use different weight layouts | Capture clean latents and complete fused DiT mapping |
| LoRA synchronization | Adapter registers successfully | Fused attention / feed-forward weights on the vLLM side are not actually matched | Explicitly map every projection into the fused structure without omitting any slice |
| Prompt integration | Text can be passed in | Text is decoded and then retokenized, which may cause prompt drift | Use H3's native text path and abort immediately on errors |
| Audio-reward integration | Total reward has a value | Audio may never enter CLAP / ImageBind | Separate the joint audio-video output, pass audio through to reward, and export MP4 files with a soundtrack |

The rows follow the order in which we encountered these pitfalls: every stage looked correct but was actually wrong, and every root cause came from differences between the training and inference engines.

In summary, RL post-training for MiniMax-H3 should begin by verifying rollout performance and each part of the train-inference consistency pipeline before tuning results for vertical domains.

The full MiniMax-H3 DiffusionNFT training implementation is open source and can be reproduced through the references below.

## References

- [DiffusionNFT paper: Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117)
- [PR #383: MiniMax H3 DiffusionNFT T2VA training](https://github.com/verl-project/verl-omni/pull/383)
- [PR #401: MiniMax H3 FL2VA DiffusionNFT](https://github.com/verl-project/verl-omni/pull/401)
- [MiniMax-H3 T2VA/FL2VA recipe](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/README.md)
- [H3 T2VA launch script](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh)
- [H3 FL2VA launch script](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh)
- [MiniMax-H3 Model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
