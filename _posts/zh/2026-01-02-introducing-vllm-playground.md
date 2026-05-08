---
layout: post
title: "vLLM Playground：管理和交互 vLLM 服务的现代化 Web 界面"
author: micytao
image: /assets/figures/vllm-playground/vllm-playground-newUI.png
lang: zh
translation_key: introducing-vllm-playground
tags:
  - frontend
  - ecosystem
---

作为一名希望看到 vLLM 持续壮大、触达更多开发者的社区成员，我很高兴地宣布 **[vLLM Playground](https://github.com/micytao/vllm-playground)** 正式发布。它是一个现代化、功能丰富的 Web 界面，专门用于管理和交互 vLLM 服务。无论你是在 macOS 上本地开发、在 Linux GPU 机器上测试，还是在企业级 Kubernetes / OpenShift 集群中部署，vLLM Playground 都能提供统一、直观的使用体验。

<p align="center">
<picture>
<img src="/assets/figures/vllm-playground/vllm-playground-newUI.png" width="100%">
</picture>
</p>

## 为什么需要 vLLM Playground？

搭建和管理 vLLM 服务，通常意味着你需要熟悉命令行、容器编排以及大量配置选项。vLLM Playground 通过下面这些能力，把这些门槛降了下来：

- **零配置要求**：不需要手动安装 vLLM，容器会自动处理一切
- **一键操作**：通过直观 UI 启动/停止服务、切换模型、调整配置
- **跨平台支持**：支持 macOS（Apple Silicon）、Linux（CPU/GPU）以及企业级 Kubernetes 环境
- **一致的界面体验**：从本地开发到云端部署，界面和工作流保持一致

## 愿景与路线图

vLLM Playground 的目标很简单：**紧跟 vLLM 官方项目的节奏，让每一项新特性都更容易被访问、被尝试。**

vLLM 演进得非常快，不断加入结构化输出、工具调用、投机解码、多模态支持等能力。但要真正体验这些特性，往往需要翻文档、写脚本并维护复杂配置。vLLM Playground 正是在这里补上缺口，它通过一个可视化、交互式界面，让你在 vLLM 新特性刚发布时就能第一时间上手实验。

**后续路线图包括：**

- **🔗 MCP Server 集成**：通过 Model Context Protocol 增强工具能力
- **➕ RAG 支持**：支持 Retrieval-Augmented Generation，实现知识增强回答
- **🎯 特性对齐**：随着 vLLM 新能力落地，持续在 UI 中提供对应支持

## 快速开始

上手非常简单：

```bash
# 从 PyPI 安装
pip install vllm-playground

# 预下载容器镜像（可选，GPU 镜像约 10GB）
vllm-playground pull

# 启动 playground
vllm-playground
```

打开 `http://localhost:7860`，点击 “Start Server”，你就可以开始运行 vLLM 了。容器编排器会自动为你拉取适合当前平台的镜像，并管理 vLLM 的生命周期。

## 核心特性

### 🎨 现代深色主题 UI

新的界面采用更简洁、专业的设计，主要包括：

- **精简聊天界面**：干净、低干扰的聊天 UI，并配有可展开的行内面板
- **图标工具栏**：快速访问设置、系统提示词、结构化输出、工具调用等高级功能
- **实时指标**：每次响应都显示 token 数量和生成速度
- **可调整大小面板**：可根据自己的工作流习惯自定义布局

### 🏗️ 结构化输出

可以通过四种强大的模式，把模型输出约束为指定格式：

| 模式 | 描述 | 示例用例 |
|------|------|----------|
| **Choice** | 强制输出为某几个固定值之一 | 情感分析（positive / negative / neutral） |
| **Regex** | 输出必须匹配正则表达式 | 邮箱、电话、日期格式校验 |
| **JSON Schema** | 生成满足给定 Schema 的合法 JSON | API 响应、结构化信息抽取 |
| **Grammar (EBNF)** | 定义更复杂的输出结构 | 自定义 DSL、形式语言 |

<p align="center">
<picture>
<img src="/assets/figures/vllm-playground/vllm-playground-structured-outputs.png" width="100%">
</picture>
</p>

### 🔧 工具调用 / 函数调用

你可以让模型调用你自定义的工具和函数：

- **服务端配置**：在启动前于服务器配置面板中启用
- **自动检测解析器**：可自动为 Llama 3.x、Mistral、Hermes、Qwen、Granite 和 InternLM 选择解析器
- **预置工具**：内置天气、计算器和搜索工具
- **自定义工具创建**：可以定义工具名、描述以及 JSON Schema 参数
- **并行工具调用**：支持同时调用多个工具

### 🐳 容器编排

vLLM Playground 会在隔离容器中管理 vLLM，提供：

- **自动生命周期管理**：启动、停止、健康检查、日志流式输出
- **智能容器复用**：配置不变时可快速重启
- **跨平台镜像支持**：
  - GPU：`vllm/vllm-openai:v0.11.0`（官方）
  - CPU x86：`quay.io/rh_ee_micyang/vllm-cpu:v0.11.0`
  - macOS ARM64：`quay.io/rh_ee_micyang/vllm-mac:v0.11.0`

### 📊 GuideLLM 基准测试集成

借助 [GuideLLM](https://github.com/neuralmagic/guidellm)，Playground 提供了完整的性能测试能力：

- 请求统计（成功率、持续时间、平均时间）
- Token 吞吐量分析（平均值/中位数 tokens/s）
- 延迟分位数（P50、P75、P90、P95、P99）
- 可配置的负载模式和请求速率
- 可导出 JSON 以做详细分析

<p align="center">
<picture>
<img src="/assets/figures/vllm-playground/guidellm.png" width="100%">
</picture>
</p>

### 📚 vLLM 社区 Recipes

支持一键加载来自官方 [vLLM Recipes Repository](https://github.com/vllm-project/recipes) 的模型配置：

- **17+ 模型类别**：包括 DeepSeek、Qwen、Llama、Mistral、InternVL、GLM、NVIDIA Nemotron 等
- **可搜索目录**：可按模型名、类别或标签过滤
- **一键加载**：自动填充优化过的 vLLM 设置
- **硬件指引**：直接查看每个模型推荐的 GPU 配置

<p align="center">
<picture>
<img src="/assets/figures/vllm-playground/vllm-recipes-1.png" width="100%">
</picture>
</p>

### ☸️ OpenShift / Kubernetes 部署

Playground 还支持企业级云环境部署：

- 通过 Kubernetes API 动态创建 vLLM Pod
- 自动识别并支持 GPU 与 CPU 模式
- 基于 RBAC 的安全模型
- 自动化部署脚本
- 与本地环境完全一致的 UI 和工作流

```bash
cd openshift/
./deploy.sh --gpu    # 用于 GPU 集群
./deploy.sh --cpu    # 用于仅 CPU 集群
```

## 架构概览

vLLM Playground 采用一种混合架构，可以在本地和云环境中无缝运行：

```text
┌─────────────────────────────────────────────────────────────┐
│                     Web UI (FastAPI)                        │
│              app.py + index.html + static/                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├─→ container_manager.py (Local)
                         │   └─→ Podman CLI
                         │       └─→ vLLM Container
                         │
                         └─→ kubernetes_container_manager.py (Cloud)
                             └─→ Kubernetes API
                                 └─→ vLLM Pods
```

容器管理器会在构建阶段完成切换（Podman → Kubernetes），从而保证本地与云端拥有一致的用户体验。

## macOS Apple Silicon 支持

它也完整支持 macOS ARM64：

- 专门为 Apple Silicon 构建的 CPU 优化容器镜像
- 自动平台检测
- 基于 Podman 的 rootless 容器执行
- 为最佳性能预先配置好的 CPU 设置

```bash
# 只需启动 Web UI - 它会自动处理容器
python run.py
# 或者直接使用 CLI
vllm-playground
```

## CLI 命令

```bash
vllm-playground                    # 使用默认配置启动
vllm-playground --port 8080        # 自定义端口
vllm-playground pull               # 预下载 GPU 镜像 (~10GB)
vllm-playground pull --cpu         # 预下载 CPU 镜像
vllm-playground pull --all         # 预下载全部镜像
vllm-playground stop               # 停止运行中的实例
vllm-playground status             # 检查是否正在运行
```

## 参与贡献

vLLM Playground 是一个 Apache-2.0 协议的开源项目，欢迎贡献。

- **GitHub**：[https://github.com/micytao/vllm-playground](https://github.com/micytao/vllm-playground)
- **PyPI**：[https://pypi.org/project/vllm-playground/](https://pypi.org/project/vllm-playground/)
- **Issues & PRs**：欢迎提交 Bug 报告、功能请求和 Pull Request

现在就可以试试：

```bash
pip install vllm-playground
vllm-playground
```

希望 vLLM Playground 能让你的 vLLM 开发与部署体验更顺畅。Happy serving!
