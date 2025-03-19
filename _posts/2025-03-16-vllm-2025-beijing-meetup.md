---
layout: post
title: "vLLM Beijing Meetup: Innovation, Ecosystem and Community"
author: "vLLM / vLLM Ascend / verl / LLaMAFactory Team"
image: /assets/logos/vllm-logo-text-light.png
---

On March 16, 2025, vLLM community hosted the first vLLM China meetup (vLLM Beijing Meetup) with Huawei. Maintainers from vLLM, verl, LLaMA-Factory and vLLM Ascend collaboratively demonstrated best practices for leveraging vLLM's capabilities in post-training, finetune, inference and deployment scenarios.

<p align="center">
    <picture>
    <img src="/assets/figures/vllm-2025-beijing-meetup/0.png" width="45%">
    </picture>
</p>

It is worth noting that this meetup marked the first vLLM community exchange event held in China. It not only provided a great platform for communication among major Chinese enterprises and universities but also strengthened the connection between the vLLM community and Chinese developers.

In the future, we hope that through the joint efforts of the vLLM community and Chinese users, vLLM can be further refined, made more efficient, and user-friendly.

## Talks

### vLLM 2025 Q1 Update

<p align="center">
    <picture>
    <img src="/assets/figures/vllm-2025-beijing-meetup/1.png" width="45%">
    </picture>
</p>

Zhang Chen, one of the maintainers of vLLM, shared the recent work of the vLLM community. She also highlighted the new features of the vLLM V1 Engine such as the simplified scheduler. The V1 Engine is more efficient and will be enabled by default to support use cases in the upcoming v0.8.0 release.

<p align="center">
    <picture>
    <img src="/assets/figures/vllm-2025-beijing-meetup/2.png" width="45%">
    </picture>
</p>

You Kaichao, another maintainer of vLLM, highlights the DeepSeek supports and performance improvements, then gives a brief introduction about vLLM ecosystem project including vLLM Ascend[3], vLLM Spyre[7], LLM Compressor[8], AIBrix[9], vLLM Production Stack[10] and partner projects like Hugging Face, PyTorch, and also shared vLLM team has created Zhihu[11] and WeChat public account to facilitate better connections between the vLLM team and Chinese developers.

### vLLM Hardware Plugin Mechanism and Ascend Best Practices

<p align="center">
    <picture>
    <img src="/assets/figures/vllm-2025-beijing-meetup/3.png" width="45%">
    </picture>
</p>

Wang Xiyuan, the vllm-project/vllm-ascend[3] project maintainer, shared the internal design and implementation of vLLM hardware plugin[6] and the best practices from Ascend NPU. He introduced the internal implementation details of vLLM Ascend and the practices in versioning policy and user experience improvement. Finally, he announced vLLM Ascend v0.7.3rc1[12] release and thanks for contributions and feedback from the vLLM Ascend users and developers.

### verl: A Hybrid Controller-based RLHF Framework

<p align="center">
    <picture>
    <img src="/assets/figures/vllm-2025-beijing-meetup/4.png" width="45%">
    </picture>
</p>

Zhang Chi, creator and maintainer of the volcengine/verl[4] project, shared ByteDance's research and work in the field of reinforcement learning fine-tuning frameworks. He focused on the pain points currently addressed by verl and the internal design of the Hybrid Controller. Based on the new feature called SPMD and sleep mode supported in vLLM, verl now can easily integrate with vLLM without any patch. Users can use verl and vLLM together smoothly from vLLM 0.7.x now.

### Best Practices for Efficient Fine-Tuning Framework LLaMA-Factory with vLLM

<p align="center">
    <picture>
    <img src="/assets/figures/vllm-2025-beijing-meetup/5.png" width="45%">
    </picture>
</p>

Zheng Yaowei, LLaMA-Factory[5] author & maintainer from Beihang University, shared the current development status in the field of large model fine-tuning and the latest graphical interface launched by LLaMA-Factory. He also introduced how LLaMA-Factory works with vLLM to provide developers with ultimate ease of use through the fine-tuning, inference demo video demonstration.

You can find the slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).

[3]: https://github.com/vllm-project/vllm-ascend
[4]: https://github.com/volcengine/verl
[5]: https://github.com/hiyouga/LLaMA-Factory
[6]: https://github.com/vllm-project/vllm/issues/11162
[7]: https://github.com/vllm-project/vllm-spyre
[8]: https://github.com/vllm-project/llm-compressor
[9]: https://github.com/vllm-project/aibrix
[10]: https://github.com/vllm-project/production-stack
[11]: https://www.zhihu.com/people/vllm-team
[12]: https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc1
