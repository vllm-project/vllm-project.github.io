---
layout: post
title: "The First vLLM Meetup in Korea"
author: "vLLM Team" 
---

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/image-3.png" width="100%">
</picture><br>
</p>

The first vLLM meetup in Korea was held on August 19, 2025, in Seoul, hosted by Rebellions and Red Hat with support from PyTorch Korea User Group and SqueezeBits.

Here’s are numbers that mattered:
350+ signed up, attendees came from more than 75 companies, and 80% were industry professionals - and 80% of those were software engineers and researchers. A strong showing for vLLM’s debut in Korea.

The event brought together local developers, researchers, and AI infrastructure engineers to share insights on efficient LLM inference and explore how vLLM is enabling scalable, hardware-friendly deployment - now including NPUs.

## Highlights

### Intro to vLLM + llm-d and Deep dive into vLLM TPU Integration by Nicolo Lucchesi

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/vllm_meetup_nicolo.jpg" width="100%">
</picture><br>
</p>


Nicolò Lucchesi, Senior ML Engineer at Red Hat, opened the event by highlighting the original innovation behind vLLM — solving long-standing challenges in KV caching and dynamic batching with a novel paged attention architecture. He emphasized that “modern problems require traditional solutions,” noting that the exact challenges in scheduling and memory management had already been tackled in operating systems, and vLLM simply applies the same proven ideas to AI inference.

He also introduced llm-d, a project enabling distributed inference. llm-d is a Kubernetes-native orchestration layer that coordinates multiple vLLM instances with auto-scaling support — “vLLM meeting Kubernetes.”

Nicolò concluded with ongoing work to integrate AI accelerators like Google TPU, broadening vLLM’s accessibility across hardware platforms.

### Building, Testing and Contributing to vLLM by Daniele Trifirò

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/vllm_meetup_Daniele.png" width="100%">
</picture><br>
</p>

Daniele Trifirò, Software Engineer at Red Hat, shared how developers can build, test, and contribute to the vLLM project — with a focus on real-world AI serving. He highlighted the fast-paced development cycle, where weekly releases and a growing contributor base are pushing out massive changes in code. Building vLLM isn’t always straightforward due to hardware requirements, and Daniele offered practical tips and insights to help new contributors get started.

He also explained the need for hardware-specific compilation, noting how memory usage can spike dramatically during builds depending on the target (e.g., CUDA, ROCm, TPU). To improve flexibility and developer access, he introduced vLLM’s new hardware plugin system. This plugin architecture makes vLLM more device-agnostic and further strengthens its position as a robust and scalable AI serving ecosystem.

### Supercharging Rebellions NPU with vLLM by Hong-seok Kim

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/vllm_meetup_HSkim.png" width="100%">
</picture><br>
</p>


Hong-Seok Kim, Chief Software Architect at Rebellions, spoke about the growing importance of vLLM for AI accelerator startups and shared how Rebellions is contributing to the broader AI inference serving ecosystem. He highlighted how vLLM’s hardware plugin system enables companies like Rebellions to support developers in deploying LLMs on custom hardware — delivering a near-seamless experience comparable to running on GPUs. 

Thanks to vLLM, engineers can now run MoE (Mixture of Experts) models directly on Rebellions’ NPU, while also leveraging core optimizations like parallelism and continuous batching — all without complex integration steps. This opens the door to efficient, scalable AI serving on next-generation accelerators.


### Quantization and Evaluation with vLLM by Hyungjun Kim

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/vllm_meetup_HJKim.jpg" width="100%">
</picture><br>
</p>


Hyungjun Kim from SqueezeBits explored how quantization is becoming an essential part of LLM deployment — and how it can be effectively used within the vLLM ecosystem. He outlined two primary ways to serve quantized models with vLLM: Load a pre-quantized model for serving, or Quantize the model yourself and then deploy.

To simplify the process, the vLLM project includes an open-source subproject called LLM Compressor, which helps developers integrate quantization into their pipelines more easily.
Hyungjun also introduced Fits on Chips, an open-source toolkit from SqueezeBits that evaluates LLM serving performance within vLLM. The toolkit helps compare throughput, latency, accuracy, and hardware, giving a clear view of the most efficient serving configurations.

## Looking Ahead

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/image-2.png" width="100%">
</picture><br>
</p>

The meetup also looked ahead to how the vLLM community in Korea can continue to grow. We’re planning to host regular vLLM Korea meetups in collaboration with local engineering groups — including the PyTorch Korea User Group and Python Korea. These gatherings will include hands-on workshops, developer meetups, and small-group sessions aimed at strengthening both community bonds and technical contributions to the vLLM ecosystem.

In the early days of open source, contributions were more evenly distributed. But with the rise of LLMs and the need for AI accelerators, it’s become harder for individual engineers and academics to gain real-world experience. We believe that through community-driven infrastructure and collaboration, we can build a sustainable, hands-on learning environment — and we welcome new volunteers to help shape the future of vLLM.

<p align="center">
<picture>
<img src="/assets/figures/vllm-meetup/image-6.png" width="100%">
</picture><br>
</p>

This inaugural meetup marked an exciting step for Korea’s vLLM community, reaffirming what matters most: practical, scalable solutions for real-world AI serving. Rebellions, Red Hat, and passionate engineers across the region are committed to supporting more community-driven events and continued contributions to the vLLM project.

Thanks to everyone who joined and made this first gathering a success — we’re just getting started.
