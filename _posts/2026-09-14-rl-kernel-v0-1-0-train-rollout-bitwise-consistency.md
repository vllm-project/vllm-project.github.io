---
layout: post
title: "RL-Kernel × vime × AMD：让训练与 Rollout 实现 Bitwise 一致"
author: "RL-Kernel Team"
date: 2026-09-14
summary: "RL-Kernel 与 vime 在 CUDA 和 ROCm 上实现训练与 rollout 的 selected-token logprob 逐 Bit 一致，并对比原生路径的 200-step 端到端表现。"
image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/image35.png
social_image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/image35.png
tags:
  - reinforcement-learning
  - post-training
  - performance
  - hardware
  - ecosystem
---

On-policy RL 默认 rollout engine 与 training engine
在参数更新前评估同一个
policy。但真实系统通常由两套引擎分别完成生成和训练。模型、权重和输入相同，并不意味着执行路径相同：kernel、batch
shape、并行布局、归约顺序和中间精度都可能改变 token
probability，最终形成 train-rollout mismatch。

vime 和 RL-Kernel 分别处理这条链路的两端。vime 管理
token、状态和权重版本的生命周期；RL-Kernel 对齐
RMSNorm、Attention、GEMM、SwiGLU、linear logp 及 distributed collectives
中的归约与舍入边界。前者保证两边处于同一条训练时间线，后者保证两边遵守同一份数值执行契约。

## **引言**

训练和 rollout 对执行引擎的优化目标不同。rollout 追求
sampling、prefill、decode 和 KV cache 的吞吐；training 需要处理
forward、backward、optimizer state
与多维并行。两套系统实现的是同一个数学模型，却未必执行同一个浮点程序。

差异一旦在参数更新前进入 importance ratio 和 clipped
objective，就会表现得像一次真实的 policy shift。复用 rollout logprob
可以避开部分影响，但无法验证两套引擎独立计算时是否得到同一结果。要验证
Bitwise 一致性，必须同时固定比较对象、数值契约和实际执行路径。

下文从浮点非结合律出发，把 Attention、logp、RMSNorm、GEMM 和 collectives
放进同一个嵌套归约框架，再说明 vime 与 RL-Kernel 的分工，以及如何用
P/P、P/R、R/P、R/R 对照定位分叉。ROCm 和 CUDA 分别给出 200-step 结果。

### **本文要回答的问题**

- 为什么模型、权重和输入相同，training 与 rollout 的 logprob
  仍可能不同。

- 怎样用 comparability gate、数值执行契约和 P/P、P/R、R/P、R/R
  对照找到第一处分叉。

- vime 与 RL-Kernel 分别负责时间同步和数值对齐的哪一部分。

- 怎样确认 0 mismatch 确实来自目标 backend。

- ROCm 和 CUDA 200-step 结果证明了什么。

## **消除 Train-Rollout Mismatch**

### **同一模型为什么不是同一次计算**

在Qwen3-8B 实验中，RL-Kernel 与 vime 集成的严格 R/R 路径运行了 200 个
GRPO step。训练侧重算与 rollout 侧记录的 selected-token
logprob，在每一步都满足 mismatch_count = 0、max_abs_diff = 0。按
validator 的口径累计，共覆盖 147,379,363 个 active tokens。

这里的逐 Bit 一致指运行时逐元素零容差相等，它比较的是同一次
run、同一权重版本下，同一个 token 概率在两套引擎中的计算结果。

明明模型、权重和输入都相同，为什么原本会不一致？又为什么
RMSNorm、Attention、GEMM、logp
和通信这几个看似分散的模块，必须一起处理？

本文尝试从一个统一视角回答它：

> 模型公式只定义数学结果，并不唯一指定执行路径。有限精度破坏了路径之间的一部分等价关系，所以逐
> Bit 一致的本质，是让训练和推理遵守同一份数值契约。

## **参数更新前的虚假 Policy Shift**

rollout engine 在前缀 hₜ 上生成 token aₜ，并记录

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image14.png"
style="width:1.25667in;height:0.25333in" />

训练开始前，training engine 使用同一权重版本重新打分：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image21.png"
style="width:1.23333in;height:0.25in" />

若 policy 尚未更新，并且两侧确实计算同一个逻辑对象，importance ratio
应满足：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image27.png"
style="width:1.89in;height:0.36667in" />

令 δₜ = ℓₜᵀ − ℓₜᴿ。当 δₜ 很小时，ρₜ ≈ 1 + δₜ；它还会进入 PPO和GRPO 的
clipped objective：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image28.png"
style="width:2.84667in;height:0.34in" />

其中 Âₜ 是优势估计。当 δₜ 超过 log(1 + εhigh) 或低于 log(1 − εlow)
时，mismatch 甚至可能改变 clipping 的分支。它在参数更新之前又制造了一次
policy shift**。**

这个总误差还可以拆开。记 serving prefill 与独立 decode replay 对同一
token 的概率为 sₜᴾ、sₜᴰ，则

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image5.png"
style="width:1.5in;height:0.45667in" />

第一项比较 training scoring 与 serving prefill，第二项比较 prefill 与
decode，第三项检查权重版本、cache state
和记录身份。这个分解很重要：最终只有一个
ratio，但事实上背后却横跨引擎之间的算术、推理内部的两条路径和系统状态三个接口。任何一项没有固定，都不应把总差异含糊地称为
kernel error。

问题的源头可以压缩成一个根因：浮点加法不满足结合律。以
round-to-nearest-even 的 BF16 为例，

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image11.png"
style="width:1.64in;height:0.26333in" />

而

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image9.png"
style="width:2.01in;height:0.26333in" />

两式在实数里只是换了括号，在 BF16 里却成了两个答案。训练侧面向 packed
sequence、反向传播和跨卡并行，推理侧面向 prefill、decode、动态 batch 和
KV
cache。二者即使共享参数，也可能因为目标不同而选择不同的分块、归约顺序和中间精度。

所以，同一份权重只固定了加哪些数，没有固定先加谁。

这里还要区分两个常被混在一起的问题：复用 rollout logprob，是决定 loss
使用哪个已记录数值。算子对齐，是验证两套引擎独立计算时是否得到同一结果。前者可以绕开一部分后果，却不能证明后者成立。

## **建立可比性：固定数学对象与数值契约**

讨论浮点误差前，需要先判断两套引擎是否在回答同一个问题。把理想数学对象写成：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image24.png"
style="width:1.09333in;height:0.22667in" />

其中 x 是输入，θ 是权重，s 是 KV cache 等状态，ξ
是参与计算的随机状态。训练与 rollout 真正执行的是：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image19.png"
style="width:3.42667in;height:0.26333in" />

这里 C 是数值执行契约。

对 selected-token logprob，这至少要求以下逻辑对象相同：

- checkpoint 与权重版本；

- prefix、目标 token、active mask；

- position、RoPE、causal/padding mask；

- KV cache 映射后得到的逻辑 K/V；

- head、sequence、vocabulary ownership；

- 真实词表范围以及参与被比较计算的随机状态。

若其中一项不同，应记作 comparable = false。物理 page 编号或 shard
布局可以不同，只要映射回去仍是同一个逻辑张量；固定已采样 token 做 replay
时，也不必复演 sampling RNG 的全部消费历史。

通过这道 gate 后，再逐层描述每个计算节点。先写

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image10.png"
style="width:0.91667in;height:0.23333in" />

其中 Dᵢ 是输出 yᵢ 实际依赖的输入元素集合。若节点还能写成

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image1.png"
style="width:1.09333in;height:0.39in" />

那么 Rᵢ 就是归约域。两侧必须先满足

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image3.png"
style="width:1.67in;height:0.25in" />

接着把归约域分块：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image18.png"
style="width:3.82667in;height:0.28in" />

Πᵢ 决定先产生哪些 partial
summaries，跨卡归约树决定块内与块间怎样合并。即使 Rᵢ
完全相同，只要跨卡归约树不同，也可能给出不同 bit。

然后记录节点的精度元组

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image22.png"
style="width:2.82in;height:0.23667in" />

并把每一次舍入写成

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image26.png"
style="width:0.78333in;height:0.23667in" />

将能够独立改变结果的量收拢起来，最小算术契约可以写成

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image15.png"
style="width:1.87333in;height:0.23in" />

其中，Dᵥ 汇总该节点各输出的依赖集合与归约域；Πᵥ、Tᵥ
分别记录归约分区和有序合并树；Pᵥ 是精度元组；Rᵥ 记录 Qₚ 出现的位置；Aᵥ
记录 exp、log、rsqrt、SiLU 等具体数值原语。

Fusion、materialization 与 recompute
边界不单列在这个算术契约，它们只有在改变 Tᵥ、Pᵥ、Rᵥ 或 Aᵥ
时才会改变数值，更适合作为执行机制记录。状态与控制也不塞进这个元组：cache、RNG
等状态由 comparability gate 检查，dispatch、CUDA Graph
等控制条件属于触发轴。这样可以避免同一原因在不同层级重复出现。

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image29.png"
style="width:1.44in;height:0.25333in" />

上述式子指训练和推理的算术契约的差集，这个差集可以产生候选的mismatch根因。

## **3. Transformer 中的嵌套归约结构**

RMSNorm、Attention、GEMM、linear logp 和 collectives
本质上都是在做这样的归约：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image25.png"
style="width:1.18in;height:0.38in" />

按归约域 R 分块，先求局部
Agg(R⁽ʲ⁾)，再合并。实数理论通常把不同的分法和括号视为等价；但有限精度执行时却并不等价。

| **模块**                 | **归约域** | **归约语义**                 |
|--------------------------|------------|------------------------------|
| RMSNorm                  | hidden 维  | 平方和，决定整条向量的尺度   |
| GEMM                     | K 维       | 乘积项之和，决定一个输出元素 |
| Attention                | 可见 Key   | max、sum-exp 与加权 Value    |
| linear logp              | vocabulary | max、sum-exp 与目标 logit    |
| AllReduce、ReduceScatter | Rank       | 各卡产生的局部贡献           |

从这个视角看，所谓 Split-K、Split-KV、vocab shard、context parallel 和
rank tree，只是把不同的数学轴切开了。

### **3.1 Attention 与 Logprob 的共同归一化结构**

给定一组 score sᵢ，令

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image6.png"
style="width:1.96667in;height:0.37333in" />

m 被域内最大值固定，l
记录相对它的指数和。对最终实数结果，二者可以合成一个 LSE；但实际 kernel
会分别更新和合并它们，所以 bitwise 契约也必须保留这两个中间状态。

linear logp 只需保留 (m, l) 和目标 logit：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image16.png"
style="width:1.76in;height:0.26in" />.

Attention 则多携带一个向量

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image30.png"
style="width:2.57667in;height:0.28667in" />.

因此二者本质上是在不同空间做同一种 LSE 汇总：Attention
在上下文空间归一化负责该看哪个 token，logp 在词表空间归一化负责该选哪个
token。Attention 的 Split-KV merge 与 logp 的跨 TP vocab
merge，数学上是同一种问题的两个实例。

如果把两块 (m₁, l₁, o₁)、(m₂, l₂, o₂) 合并，必须先令 m = max(m₁,
m₂)，再计算

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image8.png"
style="width:3.23333in;height:0.21333in" />.

在实数中，这个合并运算满足结合律，所以可以从任意分区恢复同一个全局结果。对任意多个块，合并后直接写成

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image4.png"
style="width:3.44333in;height:0.39333in" />

右边只依赖所有块的集合，这就是结合性的简短证明。

但实际计算用的是带舍入与近似的合并 ⊕̂。一般有

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image17.png"
style="width:1.82667in;height:0.25in" />

其中 σⱼ = (mⱼ, lⱼ, oⱼ)。因此分区 Πᵢ、归约树 Tᵢ、exp 原语 Aᵥ 和 m, l, o
的精度 Pᵥ 是这个归一化本身的一部分。

### **3.2 RMSNorm、GEMM 与通信中的归约顺序**

RMSNorm 用了 Σᵢ xᵢ² ；GEMM 用了 Σₖ aᵢₖbₖⱼ ；row-parallel GEMM
结束后，AllReduce 又把各 Rank 的局部和继续相加。

设 K 维按 Rank 分成不相交的 K₀, …, Kₚ₋₁，那么

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image2.png"
style="width:2.29667in;height:0.75in" />

内层的 Yᵢⱼ⁽ʳ⁾ 由本地 GEMM 产生，外层的 ΣᵣYᵢⱼ⁽ʳ⁾ 由 AllReduce
完成。二者在数学上是一重求和，工程上却被 kernel 边界切成两级归约。因此
collective是同一棵总归约树跨出 GPU 后的下半段。

RMSNorm 与 softmax
还有另一层联系：它们都先把一个域压成少数全局统计量，再把统计量广播回每个局部输出。RMSNorm
为

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image13.png"
style="width:2.68667in;height:0.53in" />

softmax 则把同一组 (m, l) 用到每个 score。于是归约中的一个 bit
会通过共享尺度同时耦合整条 hidden、整行 Attention
或整个词表分布。三者的语义含义不同，数值结构却都是一样的。

这也说明只固定一层是不够的。关闭 GEMM 的 Split-K，只消除了 kernel
内的一类 partial merge；若 TP AllReduce 仍使用不同的 Rank
tree，完整的加法括号依然不同。反过来，固定 Rank tree 也不能替代 kernel
内的 warp/CTA 归约契约。

### **3.3 Fusion 与舍入边界**

假设一侧执行

GEMM -\> 写回 BF16 -\> SiLU -\> multiply

另一侧执行

GEMM -\> 保留 FP32 accumulator -\> SiLU -\> multiply -\> 写回 BF16

二者在纸上都是
SwiGLU，差异却已经发生在中间张量是否被物化。Fusion、recompute 和
communication staging 之所以可能影响结果，是因为它们可能移动契约中的
Rᵥ，也就是舍入发生的位置。

同理，两个实现即使都标成 FP32，也可能因 exp、log、rsqrt、SiLU、FMA 或
fast-math 原语不同而产生不同 bit。dtype 只描述容器，不能完整描述计算。

### **3.4 从连续数值误差到离散路径分叉**

微小的 score 差异可能改变 sampling、argmax、top-k
或阈值判断，随后两条轨迹不再可比。这里真正相关的离散边界是
mask、position、cache lookup、selected token 和 vocabulary ownership。

这也是 fixed replay 的意义：先冻结 token，把采到了不同结果和同一 token
的概率算得不同拆成两个问题。

## **4. 从时间同步到代数对齐：vime 与 RL-Kernel 的协同**

有了上面的形式化，vime 与 RL-Kernel 的关系就清楚了。

同一条训练时间线

prompt -\> vLLM rollout -\> token / rollout logp -\> Megatron scoring
-\> backward -\> update

vime 解决的是 是否处在同一个时刻：哪批 token、哪版权重、哪份 rollout
记录进入哪次更新。RL-Kernel
解决的是同一时刻是否使用同一种代数实现：这些贡献如何分块、合并和舍入。可以简写为：vime
对齐因果时间，RL-Kernel 对齐数值代数**。**

缺少前者，完全确定的 kernel
也可能在比较不同权重；缺少后者，同一权重仍可能被两套实际实现解释。二者分别封住
F 的状态自由度与 F̂C 的实现自由度。

Qwen3/H100 strict 路径把这套思想落到了五处：

| **模块**      | **当前路径固定了什么**                                                                                   | **处理对象**         |
|---------------|----------------------------------------------------------------------------------------------------------|----------------------|
| RMSNorm       | 共享 primitive、hidden width、eps、residual-add 与输出边界                                               | hidden               |
| Attention     | position/mask/GQA/cache 身份；共同 FA4 arithmetic identity；num_splits=1；FP32 LSE；final-write downcast | Key 上的 (m, l, o)   |
| GEMM / SwiGLU | no-Split-K；BF16 operands、FP32 accumulation；统一 epilogue 与激活物化边界                               | K 维乘积和           |
| linear logp   | 真实 vocab 151,936；padding 128 lanes 屏蔽；固定本地树和跨 Rank LSE merge；FP32 logp                     | vocab 上的 (m, l)    |
| Collectives   | 逻辑 ownership、payload dtype 和显式 Rank tree                                                           | 各卡 partial summary |

num_splits=1 与 no-Split-K
是当前系统中最容易审计的选择，只要两边完整固定 partition、partial state
与 merge
tree，同样可以形成有效契约。一致性只要求它们不能悄悄改变可观测的数值语义。

### **Backward 的独立执行图**

rollout 没有 backward，因此 forward train-rollout parity
不可能推出跨引擎 backward parity。而且反向传播会引入新的归约轴。例如

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image7.png"
style="width:1.17in;height:0.28667in" />ᵀ

在 forward GEMM 中被归约的是 hidden/K 维，这里被归约的却是 token
维。microbatch 切法、梯度累计顺序、重计算保存值、atomic 和梯度
collective 都可能重新改变括号。

写成 VJP，就是

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image12.png"
style="width:1.50667in;height:0.25in" />.

所以更准确的说法是：本次证据验证了跨引擎 forward logprob；training
backward 的可重复性需要把 VJP
当作独立计算图，再检查其域、分区、树、保存精度和通信。配置
deterministic_backward=true 是契约的一部分。

## **5. Train-Rollout Mismatch 的关键触发轴**

batch size、sequence length、prefill/decode、workspace、CUDA Graph、GPU
型号和 topology 经常与 mismatch 同时变化，但它们通常只是触发条件。

触发条件 → 路径选择 → ΔCᵥ → 第一处数值分叉

例如：

batch size 改变

-\> cuBLASLt heuristic 改变

-\> Split-K 数量改变

-\> K 维 partial merge tree 改变

-\> GEMM 输出 bit 改变

batch 导致 mismatch只描述了相关性；batch 触发另一种 Split-K
归约树才接近根因。

### **单变量消融：一次只改变一个触发轴**

设完全对齐的基线契约为 C，只改变第 k 个字段：

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image23.png"
style="width:2.90333in;height:0.26in" />.

先找第一处非零的 Δₖ，再沿计算图观察它如何传播到
hidden、logits、selected-token logp 和 ρₜ。这比看到最终 logp
不同后逐个怀疑模块更有解释力。

训练侧和rollout 侧的算子替换可以组成一个方阵，其中 P 表示 production
implementation，R 表示 RL-Kernel：

| **组合** | **Training** | **Rollout** | **能回答的问题**                         |
|----------|--------------|-------------|------------------------------------------|
| R/R      | R            | R           | 完全对齐的控制基线                       |
| P/R      | P            | R           | 只替换 training side 后是否分叉          |
| R/P      | R            | P           | 只替换 rollout side 后是否分叉           |
| P/P      | P            | P           | 原生路径表现；不能靠对角线差异归责某一侧 |

定位 Attention 时，FFN 与 logp 必须保持 R/R；定位 FFN 时同理。任何 R
side 的 silent fallback
都应使实验失败，否则对齐结果可能根本没有运行对齐实现。

这里的 R/R、P/R、R/P、P/P 是算子两侧实现矩阵。后文 G00-G11 则指是否复用
rollout logprob × 是否启用对齐算子的系统实验编码。

### **严格 Bitwise 一致性需要执行来源证明**

可信的 0 mismatch 至少需要五层证据：

1.  comparability gate 证明比较对象、状态与 ownership 相同；

2.  固定输入验证同一路径可以重复执行；

3.  training 与 rollout 对同一 operator input 做 cross-engine parity；

4.  同一权重版本下在线比较 selected-token logprob；

5.  完整 workflow 封存每步结果，并记录 backend、device、fallback 与
    CUDA和HIP Graph route。

配置文件显示启用
RL-Kernel，无法自动证明它在运行时被执行。应用日志中仍然出现
NCCL，也无法说明目标 payload
是否使用固定树。数值结果回答算出了什么，execution provenance
回答由哪条路径算出，两类证据必须同时成立。

## **Qwen3-8B 的实验结果与验证边界**

本次 CUDA 结果对应的实验配置：200-step records、aggregate
tables、bootstrap statistics、validation 与绘图脚本。

### **实验配置**

| **项目**                | **配置**                                      |
|-------------------------|-----------------------------------------------|
| Model / dtype           | Qwen3-8B / BF16                               |
| Hardware                | 1 node，8×NVIDIA H100 80GB                    |
| Megatron                | TP4 / CP2 / PP1，使用 8 张 GPU                |
| Rollout                 | 2 个 vLLM engines，每个 TP4                   |
| Placement               | actor 与 rollout colocated                    |
| Horizon                 | 200 rollout/training steps                    |
| Seeds                   | training 1234，rollout 1234                   |
| Sampling                | 每步 8 prompts × 16 samples，global batch 128 |
| Response limit          | 7,168 tokens                                  |
| Dynamic batching        | maximum 4,096 tokens/GPU                      |
| vLLM memory utilization | 0.4                                           |
| CUDA Graph              | FULL_DECODE_ONLY，保留生产图执行路径          |
| KL loss                 | enabled，coefficient 0.001                    |
| Snapshot requirement    | 每个 step 必须恰好包含 8 个 Rank files        |

### **Bitwise 一致性**

| **实验项**    | **路径**       | **Active-token comparisons** | **Mismatches** | **Agreement** |
|---------------|----------------|------------------------------|----------------|---------------|
| G10           | production P/P | 140,601,694                  | 58,230,217     | 58.58%        |
| Optimized G11 | strict R/R     | 147,379,363                  | 0              | 100%          |

G11 每一步的 runtime mismatch_count 与 max_abs_diff 都为 0；G10
每一步都有非零 mismatch，单步 max_abs_diff 的最大值为 1.591547。累计数由
rounds.csv 中每步的样本均值乘 global batch 128 后求和，因此是按
validator 口径重建的运行时统计的。

两侧使用相同 workload 配置，但采样轨迹不同，所以 active-token 总数不同。

**图：G10 vs Optimized G11 · Training and Bitwise Consistency（CUDA）**

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image35.png"
style="width:6.5in;height:4.16667in" />

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image32.png"
style="width:6.5in;height:3.31944in" />

## **ROCm 上的 Bitwise 对齐进展**

我们在 Megatron training 与 vLLM rollout 组成的完整训推系统中完成了 ROCm
200-step strict R/R 验证，实现全程 zero
mismatch。这是一项针对累积集成栈的端到端验证，而非对某个单独 PR
的隔离测试。ROCm 路径沿用同一套 correctness boundary，同时保留
AITER、CK、paged KV、HIP Graph 和 ROCm collectives
的原生执行路径，并通过 runtime readback 验证实际 backend、执行路径与
fallback 状态。

### **实验配置**

| **项目**                | **配置**                                                                                                         |
|-------------------------|------------------------------------------------------------------------------------------------------------------|
| Model / dtype           | Qwen3-8B / BF16                                                                                                  |
| Hardware                | 1 node，8× AMD Instinct MI300X 192GB                                                                             |
| Megatron                | TP4 / CP2 / PP1，使用 8 张 GPU                                                                                   |
| Rollout                 | 2 个 vLLM engines，每个 TP4                                                                                      |
| Placement               | actor 与 rollout colocated                                                                                       |
| Horizon                 | 200 rollout/training steps                                                                                       |
| Seeds                   | training 1234，rollout 1234                                                                                      |
| Sampling                | 每步 1 prompt × 8 samples，global batch 8                                                                        |
| Response limit          | 7,168 tokens                                                                                                     |
| Dynamic batching        | maximum 4,096 tokens/GPU                                                                                         |
| vLLM memory utilization | 0.38                                                                                                             |
| HIP Graph               | FULL_AND_PIECEWISE，保留生产 Graph 执行路径                                                                      |
| KL loss                 | enabled，coefficient 0.001                                                                                       |
| Validation requirement  | frozen inputs 与 frozen sources 在运行前后保持一致；每个 step 均需通过 runtime provenance 与 mismatch validation |

### **Bitwise 一致性**

| **实验项**    | **路径**       | **Active-token comparisons** | **Mismatches** | **Agreement** |
|---------------|----------------|------------------------------|----------------|---------------|
| G10           | production P/P | 10,912,549                   | 8,662,719      | 20.62%        |
| Optimized G11 | strict R/R     | 9,400,614                    | 0              | 100%          |

Optimized G11 在全部 200 个 step 中，runtime mismatch_count 与
max_abs_diff 均为 0；strict validator 在来自 1,600 个样本的 9,400,614 个
active-token comparisons 上确认 torch.equal == true。G10 同样完成了全部
200 个 step，但每一步都存在非零 mismatch；累计共有 8,662,719 /
10,912,549 个 active-token comparisons 不一致，Agreement 为 20.62%。按
rounds.csv 的逐步统计口径，G10 单步 max_abs_diff 的最大值为 17.970963。

G10 最终未通过 strict validator，是因为原生 P/P 路径不会生成 RL-Kernel
operator readbacks；其 200-step 训练本身已经完成。两组实验的 Ray
submission 都成功记录了 200/200 个 rollout 与 training steps。

两侧使用相同且冻结的 workload 与 source 配置，但采样轨迹不同，因此
active-token comparisons 总数不同。累计数由 rounds.csv
中每步的样本均值乘 global batch 8 后求和，是按 validator
口径重建的运行时统计。两组实验运行前后的 frozen-input 与 frozen-source
audit d都保持一致，统计过程中没有进行缺失值填补或数据行删除。

从下图可以看到G10在75 step以后reward崩潰，这是因为G10生成response
token超过了生成上限的7168，我们可以观察发现vime原生的loss和grad
norm都比RL-Kernel更激进，这是训推不一致造成的训练不稳定，由此可以证明全对齐的重要性。

**图：G10 vs Optimized G11 · Training and Bitwise Consistency（ROCm）**

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image34.png"
style="width:6.5in;height:4.08333in" /><img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image31.png"
style="width:3.4208in;height:1.78969in" /><img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image20.png"
style="width:3.40417in;height:1.81556in" />

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image33.png"
style="width:6.5in;height:3.25in" />

## **把整条链路串起来**

不熟悉分布式 kernel
的读者，可能会觉得前面的模块跨度很大。把它们重新排成一条线，逻辑其实很短：

1.  同一模型公式只定义实数函数 F，并没有唯一指定带数值契约的实际实现
    F̂C；

2.  在参数更新前，train 与 rollout 的 logprob 差会制造虚假的 policy
    ratio。

3.  比误差前先过 comparability gate，确保 token、权重、位置、mask、cache
    和 ownership 描述的是同一个对象。

4.  把 Transformer 看成嵌套归约：RMSNorm 汇总 hidden，GEMM
    汇总特征，Attention 汇总 Key，logp 汇总词表，collective 汇总
    Rank；

5.  对每个归约只需追问同一组问题：谁参与、怎样分块、按什么树合并、何时舍入、调用什么原语；

6.  Attention 与 logp 都包含了 LSE ，collective
    算子进行跨卡归约，所以这些模块必须形成一份首尾相接的契约。

7.  vime 固定 token、状态与权重的时间关系，RL-Kernel
    固定关键算子的数值代数，二者共同把同一policy落实成可执行条件。

8.  batch、sequence length 或 CUDA Graph 只是触发轴，必须继续追到具体的
    partition、merge tree 或 rounding boundary。

9.  最后用单因素替换找第一处分叉，并同时封存输出与 execution
    provenance，才有资格把结果称为逐 Bit 一致。

从这个角度看，train-rollout mismatch
是一个抽象泄漏：上层把数学等价误当成了数值等价。RL-Kernel
的价值不只是提供若干确定性算子，还提升为跨引擎接口。更进一步说，数值契约给优化划定了可观测边界。训练和推理仍然可以拥有不同的内存布局、并行方式和调度策略；只要这些变化没有改变依赖域和舍入边界，它们就仍属于同一个可验证实现，真正应该消除是未经声明的算术差异。

## **下一步**

- 扩展到更多模型与多模态架构。

- 继续推进 MUSA、昇腾及更多硬件平台的适配。

- 推进与 Miles、AReaL 的集成。

RL post-training 的模型、硬件和执行框架还会继续变化。RL-Kernel 希望把
correctness boundary 留在系统里：每次替换
kernel、升级框架或迁移硬件，都能明确回答数值语义是否保持，以及差异从哪里开始。

## **致谢**

RL-Kernel v0.1.0
的发布离不开硬件伙伴、开源生态盟友以及核心开发团队的鼎力支持。

#### 硬件与算力伙伴

由衷感谢 AMD 团队 Liz Li、Yuhan Yang 为 RL-Kernel 提供 AMD Instinct GPU
算力资源、深度技术协作与长期支持。我们期待在 ROCm 平台上持续推进
RL-Kernel 的跨平台一致性验证、Kernel 级性能优化及大规模 RL Workload
落地。

同时也非常感谢摩尔线程 Lei Ding 推进 MUSA 平台适配 以及华为 Yang Chen
推进昇腾平台适配。感谢Embedded LLM 为项目研发和社区协作提供支持。

异构硬件平台的一致性执行与泛化是RL-Kernel的长期核心目标，我们热切期待与更多硬件厂商及开源社区合作，共同构建开放、高效的
RL-Kernel 基础设施。

#### 开源生态与框架合作

感谢 vLLM 社区对 RL-Kernel 的深度合作，特别致谢来自 Inferact 的 vime
Maintainer Ao Shen，在 RL-Kernel 与 vime
的集成、社区协作及后续维护中给予的充分信任与全力支持。本项目的顺利推进，建立在
vLLM Rollout、vime Orchestration 与 Megatron Training
所共同构建的开源生态之上。

#### 核心贡献者 - v0.1.0

特别感谢 RL-Kernel 核心贡献者团队在 v0.1.0 版本的架构设计、Kernel
实现、dense模型算子级训推一致性、分布式验证与社区建设中的辛苦付出

Chutian Wang：

- 负责 CUDA 平台跨节点和卡间通信模块开发

- 跨平台消融矩阵工程搭建。

Jiajie Li：

- 主导 vime 框架调研、Fork 版本 Roadmap 规划与 PR 交付

- 主导 CUDA和ROCm 双平台下 vime 与 RL-Kernel 的集成

- 主导 Distributed Attention 开发

- 完成 linear_logp 替换试验，含 TP 并行化与技术 Blog 撰写。

- 主导 CUDA和ROCm 双平台下 vime 原生与 RL-Kernel+vime
  的端到端训推测试及性能调优。

Siru He：

- 主导 WS1 GTest 单元测试框架搭建

- 主导 GEMM 算子在 CUDA和ROCm 双平台下的分布式适配与 PR 交付。

Xiaosong Ma：

- 开发 WS1 Attention算子

- 主导 WS1 单算子全链路开发测试与 GTest 框架建设

- 完成 CUDA 平台端到端训推测试中的 GEMM 性能优化

- 完成 ROCm 平台下的通信 PR 交付

- 参与review 工作和 协助 ROCm平台下 vime 原生与 RL-Kernel+vime
  的端到端训推测试及性能调优。

Kaijie Lin：

- 主导 Logprob 单算子与分布式 PR 开发

- 实现 ROCm 平台基于 Triton 的确定性融合 Linear-Logp 算子

- 完成 vime 集成试验中的 Logprob TP 并行化开发

Jian Zhang：

- 负责 RoPE 算子开发

- 参与 Distributed Attention 开发

- 主导昇腾平台的算子适配与相关 PR 交付

Huihong Lu：

- 负责 Triton 版 Logprob 单算子开发，含 ROCm 平台支持

- 参与 Logprob 分布式适配开发工作

Yunxiang Cai：

- 负责 RMSNorm 单算子开发

- Triton 路径下的 Attention 单算子实现

Vensen Mu：

- 负责 GEMM 单算子开发与 CUDA 平台训推测试中的性能优化

- 主导 ROCm 平台的深度适配

- 完成 ROCm 平台下的通信 PR 交付

- 主导 ROCm 环境下 vime 原生与 RL-Kernel+vime 的端到端训推 Benchmark
  与性能调优

Bosong Yang

- 参与 Distributed Attention 算子的研发

Zhewei Liu

- 参与 Distributed Attention 算子的研发

Houhong Liang

- 负责 CUDA 平台端到端训推链路的性能 Profile 与 Benchmarking 调优

Ryan Huang

- 参与Logprob 单算子与分布式 PR 开发

最后感谢参与社区贡献的贡献者们：Xiaopeng Du, Yuepeng Pan, Yiyang Fei,
Ziying Tao, Zhifu Liu, Zhengtao Chen, Mengjie Li, Zien Liu, github:
haoruilee, github: luoyueyuguang, github: hongleng, github: smarslou
