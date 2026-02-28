---
layout: post
title: "Chasing 100% Accuracy: A Deep Dive into Debugging Kimi K2's Tool-Calling on vLLM"
author: "Linian Wang (Peking University)"
image: /assets/figures/kimi-k2-accuracy/k2-vendor-verifier.jpeg
tags:
  - model-support
  - developer
---

**TL;DR:** For best compatibility with vLLM, use Kimi K2 models whose chat templates were updated after commit 94a4053eb8863059dd8afc00937f054e1365abbd ([Kimi-K2-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905)) or commit 0102674b179db4ca5a28cd9a4fb446f87f0c1454 ([Kimi-K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct)). The updates are committed per model.

### Introduction

Agentic workflows are reshaping our interaction with Large Language Models, and robust tool-calling is the engine driving this revolution. Moonshot AI's Kimi K2 model is renowned for its exceptional tool-calling capabilities. To validate its performance on the high-performance vLLM serving engine, I turned to the official [K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) benchmark.

My goal was ambitious: replicate the near-perfect performance seen on Moonshot AI's native API. Their official endpoints set a high bar, executing thousands of tool calls with zero schema validation errors—the gold standard for reliability.

**Benchmark: K2-Vendor-Verifier on Moonshot AI's API**

| Model Name | Provider | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | --- | --- | --- | --- | --- | --- |
| `Moonshot AI` | MoonshotAI | 2679 | 1286 | 35 | **0** | **1286** |
| `Moonshot AI Turbo` | MoonshotAI | 2659 | 1301 | 40 | **0** | **1301** |

However, my initial attempt to run K2 on vLLM yielded shockingly different results. The out-of-the-box performance wasn't just suboptimal; it was broken.

**Initial Test Results on vLLM**

- **vLLM Version:** `v0.11.0`
- **HF Model:** `moonshotai/Kimi-K2-Instruct-0905` at commit `09d5f937b41ae72c90d7155c9a901e2b5831dfaf`

| Model Name | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | --- | --- | --- | --- | --- |
| `Kimi-K2-Instruct-0905` (Initial HF Version) | 3705 | 248 | 44 | 30 | **218** |

Out of over 1200 potential tool calls, only 218 were successfully parsed—a success rate below 20%. This wasn't just a minor bug; it was a fundamental breakdown in communication between the model and the serving engine. This blog post documents my deep dive into debugging this discrepancy, uncovering three critical compatibility issues between Kimi K2's `chat_template` and vLLM. This journey not only helped us dramatically improve performance but also offers valuable lessons for anyone integrating complex models with modern serving frameworks.

### The Debugging Journey: Uncovering Three Core Issues

### Problem 1: The Case of the Missing `add_generation_prompt`

My first clue was a fundamental breakdown in the model's behavior. In the benchmark, requests that should have triggered tool calls were instead ending with `finish_reason: stop`. But the root issue was broader: the model wasn't generating a structured assistant reply at all. Instead of responding to the user, it was simply continuing the conversation with plain text, a behavior that would degrade performance in any chat scenario, not just tool-calling.

**The Investigation:**

To isolate the problem, I devised a crucial experiment. Instead of using vLLM's high-level `/v1/chat/completions` endpoint, I performed a two-step manual process: first, I called the tokenizer's `apply_chat_template` function externally to generate the full prompt string. Then, I sent this string to the lower-level `/v1/completions` endpoint. This manual process bypassed vLLM's internal template application and, crucially, resolved the majority of failures. The issue was clearly in how vLLM was *using* the chat template.

**The Root Cause:**

A deeper look revealed that the Kimi tokenizer's `apply_chat_template` function signature includes `**kwargs` to accept extra, model-specific parameters. One such parameter, `add_generation_prompt=True`, is essential for correctly formatting the prompt to signal the start of the assistant's turn, guiding it towards generating a tool call.

A correct prompt should end with special tokens that prime the model to act as the assistant:

```
Correct Prompt Suffix: ...<|im_assistant|>assistant<|im_middle|>
```
However, because vLLM was not passing `add_generation_prompt=True`, the prompt was truncated right after the user's message.
This malformed prompt left the model without the crucial instruction to begin its turn. As a result, it wouldn't know to generate a tool call, a text reply, or any structured response, leading it completely astray.
This happened because vLLM, for security reasons as detailed in [PR #25794](https://github.com/vllm-project/vllm/pull/25794), inspects the function signature and only passes arguments that are explicitly defined. Since `add_generation_prompt` was hidden in `**kwargs`, vLLM discarded it, causing the prompt formatting to fail silently.

**The Fix:**

After identifying the root cause, I collaborated with the Kimi team. They were incredibly responsive and, based on my findings, updated the model's `tokenizer_config.json` on the Hugging Face Hub. The fix was to explicitly declare `add_generation_prompt` as a supported parameter for the chat template. This allowed vLLM to pass the argument correctly, fixing the primary source of failed tool calls. Additionally, I submitted [this PR](https://github.com/vllm-project/vllm/pull/27622), where I whitelist standard chat-template parameters when tokenizers accept them via `**kwargs`, preventing silent tool-call failures.

### Problem 2: How an Empty `content` Derailed the Prompt

With the first issue resolved, a new, more subtle class of prompt formatting errors emerged.

**The Investigation:**

I traced these errors to conversations containing historical tool calls where the `content` field was an empty string (`''`). I discovered a subtle but critical transformation: vLLM, in its quest for a standardized internal representation, automatically promotes a simple empty string `content: ''` into a more complex list-of-dicts structure: `content: [{'type': 'text', 'text': ''}]`.

**The Root Cause:**

Kimi's Jinja-based chat template was designed to render a string `content`. When it was unexpectedly handed a list, it failed to process it correctly, inserting the literal string representation of the list into the final prompt.

**Incorrect Prompt Snippet:**

```
...<|im_end|><|im_assistant|>assistant<|im_middle|>[{'type': 'text', 'text': ''}]<|tool_calls_section_begin|>...
```

**Correct Prompt Snippet:**

```
...<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|>...
```

This critical formatting error created a malformed prompt that was enough to confuse the model's generation logic.

**The Fix:**

I proposed a change to make the `chat_template` logic more resilient. The Kimi team agreed and swiftly implemented an update. The template now explicitly checks the type of the `content` field. If it's a string, it renders it directly; if it's an iterable (like a list), it correctly processes it, preventing the formatting error.

### Problem 3: A Tool-Call ID Parser That Was Too Strict

Finally, I noticed that even when the model generated a syntactically correct tool call, vLLM would sometimes fail to parse it. This issue was particularly insidious because it often stemmed not from the current turn, but from the conversational history provided to the model.

**The Investigation:**

By inspecting the raw `text_completion` output from vLLM, the culprit became obvious. I found that in certain edge cases, particularly when misled by a malformed conversation history, the model would generate tool-call IDs that didn't strictly conform to Kimi's official specification. For instance, consider this output:

```
...<|tool_calls_section_begin|><|tool_call_begin|>search:2<|tool_call_argument_begin|>... 
```

Here, the model output an ID of `search:2`. However, the [official Kimi documentation](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/main/docs/tool_call_guidance.md) specifies a format of `functions.func_name:idx`.

**The Root Cause:**

Why would the model generate a non-compliant ID? As the Kimi team explained, a common reason is being "misled" by the conversation history. The Kimi-K2 model expects all tool call IDs in historical messages to follow the `functions.func_name:idx` format. However, if a history message from a different system contained a tool call with a malformed ID like `search:0`, the Kimi model might get confused by the unfamiliar format and attempt to generate a "similar" but incorrect ID in its response.

Interestingly, this is not an issue on Kimi's official API because, before invoking the K2 model, their API automatically renames all historical tool call IDs to conform to the `functions.func_name:idx` standard. This pre-processing step acts as a guardrail that was missing in my direct vLLM setup.

vLLM's tool-call parser logic was too brittle to handle this deviation. It relied strictly on the official format, using code equivalent to `function_id.split('.')[1].split(':')[0]` to extract the function name. When it encountered `search:2`, the initial split on `.` failed, raising an `IndexError` and causing the entire valid tool call to be discarded.

**The Fix:**

The most effective fix, recommended by the Kimi team, is for users and vendors to adopt a similar pre-processing step: ensure all historical tool call IDs are normalized to the `functions.func_name:idx` format before sending them to the model. In my case, fixing the first two prompt-formatting issues also significantly reduced the frequency of these non-compliant IDs, as a correctly formatted context makes the model more likely to generate correct outputs. Additionally, I have proposed to the vLLM community that the parser's robustness be improved to better handle minor format deviations (see [this PR](https://github.com/vllm-project/vllm/pull/27565)).

### Final Results and a New Discovery

After the Kimi team applied all fixes and updated the tokenizer on the Hub, I re-ran the K2-Vendor-Verifier and saw a dramatic improvement.

**Final Test Results on vLLM (After Fixes)**

| Metric | Value | Description |
| --- | --- | --- |
| Tool-Call F1 Score | 83.57% | The harmonic mean of precision and recall, measuring if the model triggers tool calls at the right time. |
| Precision | 81.96% | TP / (TP + FP). |
| Recall | 85.24% | TP / (TP + FN). |
| Schema Accuracy | 76.00% | Percentage of tool calls that are syntactically correct and pass validation. |
| Successful Tool Calls | 1007 | Total number of tool calls that were successfully parsed and validated. |
| Total Tool Calls Triggered | 1325 | Total attempts by the model to call a tool. |
| Schema Validation Errors | 318 | Number of triggered tool calls that failed parsing or validation. |
| Overall Success Rate | 99.925% | Percentage of the 4,000 total requests that completed successfully (3997/4000). |

The number of successfully parsed tool calls skyrocketed from **218** to **971**—a **4.4x** improvement that brought us much closer to the official API's performance. However, a new issue surfaced: 316 `schema_validation_error_count`. Digging in, I found the model on vLLM would sometimes call tools that were **not declared in the current request** (e.g., using an `img_gen` tool from chat history even if it wasn't provided in the current turn).

This is a known model hallucination issue. Proprietary services like Moonshot AI's API deploy a crucial safeguard known as an **"Enforcer."** This component acts as a gatekeeper, implementing constrained decoding to ensure the model *can only* generate tokens corresponding to the tools explicitly provided in the request. vLLM currently lacks this feature, presenting an exciting opportunity for future contributions from the open-source community. The Kimi team is actively working with the vLLM team to integrate the **"Enforcer"** component into vLLM.

### Key Takeaways and Best Practices

This deep dive offered several invaluable lessons for anyone working at the intersection of LLMs and serving infrastructure:

1. **The Devil is in the Chat Template:** The `chat_template` is the critical handshake between a model and its serving framework. When integrating a new model, meticulously validate every piece of its template logic against the framework's specific behaviors and assumptions.
2. **Peel Back the Abstraction Layer:** High-level APIs like `/chat/completions` are convenient but can obscure root causes. When debugging, don't hesitate to drop down to lower-level endpoints like `/completions`. Manually building the input is a powerful technique to isolate the problem.
3. **A Pro-Tip: Token IDs are the Ultimate Ground Truth:** For the most subtle issues, inspecting the final sequence of token IDs sent to the model is the only way to be certain. While I didn't need to resort to this for the issues above, it's a critical tool in the toolbox. Techniques like using the OpenAI-compatible API to return token IDs can be a lifesaver. For those interested, we also highlighted this in our [Agent Lightning post](https://blog.vllm.ai/2025/10/22/agent-lightning.html).
4. **Understand Framework Design Philosophy:** vLLM's strict handling of `**kwargs` is not a bug, but a deliberate security choice. Understanding these design decisions helps in quickly identifying the root cause rather than getting stuck on unexpected behavior.
5. **The Open Ecosystem Challenge:** Advanced features like a tool-call "Enforcer" are hallmarks of polished, proprietary services. Implementing these capabilities robustly and elegantly in open-source projects like vLLM is a vital challenge for the community to address.

### Conclusion

Through systematic and collaborative debugging, we successfully resolved the critical tool-calling compatibility issues for the Kimi K2 model on vLLM, boosting its success rate by over 4x and bringing its performance in line with expectations. This process was not just a technical challenge but also a testament to the power of careful, methodical investigation in a complex software ecosystem.

I hope this detailed account serves as a useful roadmap for other developers integrating complex models into vLLM and beyond. As the open-source community continues to mature, we look forward to an even more seamless model integration experience and more powerful agentic capabilities for everyone.

![](/assets/figures/kimi-k2-accuracy/k2-vendor-verifier.jpeg)

### Acknowledgements

I'd like to extend my sincere gratitude to the engineers at the Kimi team. Their deep technical expertise was crucial in pinpointing the root causes, and they swiftly implemented the necessary fixes on the Hugging Face Hub once the issues were identified. This journey and its successful outcome would not have been possible without their active collaboration and support.

In addition, I’d like to thank Kaichao You and Chauncey Jiang from the vLLM team for helping me onboard the vLLM project and explain all the details of vLLM’s toolcall functionality. vLLM plays an important role in LLM serving, and diving deep into vLLM helps me understand the nuts and bolts of LLMs.