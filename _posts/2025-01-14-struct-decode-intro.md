---
layout: post
title: "Structured Decoding in vLLM: a gentle introduction"
author: "Guest Post by BentoML and Red Hat"
---

**TL/DR**:

- Structured decoding allows precise control over LLM output formats
- vLLM now supports both [outlines](https://github.com/dottxt-ai/outlines) and [XGrammar](https://github.com/mlc-ai/xgrammar) backends for structured decoding
- Recent XGrammar integration brings up to 5x improvement in time per output token (TPOT) under load
- Upcoming v1 release focuses on enhanced performance and schedule-level mask broadcasting for mixed-requests batch support

_[vLLM](https://blog.vllm.ai/2023/06/20/vllm.html) is the high-throughput and efficient inference engine for running **large-language models** (LLMs). In this post, we will explore the annotated history of language models, describe the current state of structured decoding in vLLM, as well as the recent integration with [XGrammar](https://github.com/vllm-project/vllm/pull/10785), and [share our tentative roadmap for future improvements](https://github.com/vllm-project/vllm/issues/8779)._

> We would also invite users to tackle this blog post from a philosophical perspective, and in the process trying to posit that structured decoding represents a fundamental shift in how we think about LLM outputs. It also plays an important role in building complex agentic system.

For more information about vLLM, please check out our [documentation](https://docs.vllm.ai/en/latest/).

## Language models: A brief historical context

In 1950, Alan Turing proposed that a high-speed digital computer, programmed with rules, could exhibit emergent behaviour of intelligence (Turing, 1950). This led to two main approaches in AI development:

1. Good Old-Fashioned AI (GOFAI): A paradigm quickly emerged among researchers in the 1950s, where expert systems were designed to replicate the decision-making capabilities of a human specialist[^1], (or symbolic reasoning system), referred to by Haugland as Good Old-Fashioned AI (GOFAI) (Haugeland, 1997). However, it quickly ran into funding problems due to its semantic representation not being able to scale up to generalised tasks (Also known as the “AI Winter” (Hendler, 2008)).

2. New-Fangled AI (NFAI): Concurrently, Donald Norman’s Parallel Distributed Processing (Rumelhart et al., 1986) group investigated variations of Rosenblatt’s perception (Rosenblatt, 1958), where they proposed *hidden layers* within the network alongside with inputs and outputs to extrapolate appropriate responses based on what it had learned during training process. These connectionist networks were often built on top of statistical methods[^2]. Given the abundance of data and Moore’s Law[^3] resulting in an unprecedented amount of compute available, we see the complete dominance of connectionist networks in both research and production use-cases, most notably variants of *decoder-only* transformers[^4] for *text generations* tasks. As such, most modern transformers variants are considered **NFAI** systems.

In summary:

- GOFAI are _deterministic_ and rule-based, given its intentionality is injected through explicit programming
- NFAI are often considered as “black-box” models (in: input \- out: some output), data-driven given the networked complexity nature of its internal representations

## Why do we need structured decoding?

<figure>
  <img src="/assets/figures/struct-decode-intro/shogoth-gpt.png" />
<figcaption>
Shogoth as GPTs. In a sense, RLHF, or any post-training methods, is an injection of rules (a GOFAI system) into any large compound AI systems
</figcaption>
</figure>

LLMs excel at the following heuristic: given a blob of text, the model will generate a contiguous piece of text that it predicts as the most probable tokens. For example, if you give it a Wikipedia article, the model should produce text consistent with the remainder of said article.

These models work well given the following assumption: the input prompt must be coherent and well-structured surrounding a given problem the users want to achieve. In other words, LLMs can be unpredictable when you need output in specific formats. Think of asking a model to generate JSON \- without guidance, it might produce valid text that breaks JSON specification[^5].

This is where structured decoding comes in. It enables LLMs to generate outputs that follow a desired structure while preserving the non-deterministic nature of the system.

Companies like OpenAI have recognized this need, implementing features like [JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode) to constrain[^6] the output format. If you have built with these functionalities before (such as agentic workflows, function calling, coding assistant), chances are you are using structured decoding under the hood.

> Guided decoding is to LLMs what **validation** is to APIs - it acts as a guarantee that what comes out matches what you expect. Guided decoding ensures structure integrity that allows developers to integrate LLMs into their application with ease!

## Structured decoding and vLLM

In simple terms, structured decoding gives LLMs a "template" to follow. Users provide a schema that "influences" the model's output, ensuring compliance with the desired structure:

<svg id="mermaid-1736451893652" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1285px;" viewBox="0 0 1285 307.20703125" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#mermaid-1736451893652{font-family:"Berkeley Mono","JetBrains Mono",ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;font-size:16px;fill:#797593;}#mermaid-1736451893652 .error-icon{fill:#b4637a;}#mermaid-1736451893652 .error-text{fill:#4b9c85;stroke:#4b9c85;}#mermaid-1736451893652 .edge-thickness-normal{stroke-width:1px;}#mermaid-1736451893652 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1736451893652 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1736451893652 .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-1736451893652 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1736451893652 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1736451893652 .marker{fill:#797593;stroke:#797593;}#mermaid-1736451893652 .marker.cross{stroke:#797593;}#mermaid-1736451893652 svg{font-family:"Berkeley Mono","JetBrains Mono",ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;font-size:16px;}#mermaid-1736451893652 p{margin:0;}#mermaid-1736451893652 .label{font-family:"Berkeley Mono","JetBrains Mono",ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;color:#797593;}#mermaid-1736451893652 .cluster-label text{fill:#4b9c85;}#mermaid-1736451893652 .cluster-label span{color:#4b9c85;}#mermaid-1736451893652 .cluster-label span p{background-color:transparent;}#mermaid-1736451893652 .label text,#mermaid-1736451893652 span{fill:#797593;color:#797593;}#mermaid-1736451893652 .node rect,#mermaid-1736451893652 .node circle,#mermaid-1736451893652 .node ellipse,#mermaid-1736451893652 .node polygon,#mermaid-1736451893652 .node path{fill:#fffaf3;stroke:#b4637a;stroke-width:1px;}#mermaid-1736451893652 .rough-node .label text,#mermaid-1736451893652 .node .label text,#mermaid-1736451893652 .image-shape .label,#mermaid-1736451893652 .icon-shape .label{text-anchor:middle;}#mermaid-1736451893652 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-1736451893652 .rough-node .label,#mermaid-1736451893652 .node .label,#mermaid-1736451893652 .image-shape .label,#mermaid-1736451893652 .icon-shape .label{text-align:center;}#mermaid-1736451893652 .node.clickable{cursor:pointer;}#mermaid-1736451893652 .root .anchor path{fill:#797593!important;stroke-width:0;stroke:#797593;}#mermaid-1736451893652 .arrowheadPath{fill:#0b0b0b;}#mermaid-1736451893652 .edgePath .path{stroke:#797593;stroke-width:2.0px;}#mermaid-1736451893652 .flowchart-link{stroke:#797593;fill:none;}#mermaid-1736451893652 .edgeLabel{background-color:#8f9fa926;text-align:center;}#mermaid-1736451893652 .edgeLabel p{background-color:#8f9fa926;}#mermaid-1736451893652 .edgeLabel rect{opacity:0.5;background-color:#8f9fa926;fill:#8f9fa926;}#mermaid-1736451893652 .labelBkg{background-color:rgba(143, 159, 169, 0.5);}#mermaid-1736451893652 .cluster rect{fill:#fffaf3;stroke:hsl(342.962962963, 0%, 44.7058823529%);stroke-width:1px;}#mermaid-1736451893652 .cluster text{fill:#4b9c85;}#mermaid-1736451893652 .cluster span{color:#4b9c85;}#mermaid-1736451893652 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"Berkeley Mono","JetBrains Mono",ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;font-size:12px;background:#b4637a;border:1px solid hsl(342.962962963, 0%, 44.7058823529%);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1736451893652 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#797593;}#mermaid-1736451893652 rect.text{fill:none;stroke-width:0;}#mermaid-1736451893652 .icon-shape,#mermaid-1736451893652 .image-shape{background-color:#8f9fa926;text-align:center;}#mermaid-1736451893652 .icon-shape p,#mermaid-1736451893652 .image-shape p{background-color:#8f9fa926;padding:2px;}#mermaid-1736451893652 .icon-shape rect,#mermaid-1736451893652 .image-shape rect{opacity:0.5;background-color:#8f9fa926;fill:#8f9fa926;}#mermaid-1736451893652 :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-1736451893652 .default&gt;\*{fill:#fff!important;stroke:#333!important;stroke-width:2px!important;}#mermaid-1736451893652 .default span{fill:#fff!important;stroke:#333!important;stroke-width:2px!important;}</style><g><marker id="mermaid-1736451893652_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-1736451893652_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-1736451893652_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-1736451893652_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="mermaid-1736451893652_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="mermaid-1736451893652_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><g class="root"><g class="clusters"><g class="cluster " id="Decoding" data-look="classic"><rect style="" x="266.5" y="8" width="739" height="144.00390625"></rect><g class="cluster-label " transform="translate(598, 8)"><foreignObject width="76" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Decoding</p></span></div></foreignObject></g></g><g class="cluster " id="subGraph0" data-look="classic"><rect style="" x="8" y="172.00390625" width="501.5" height="127.203125"></rect><g class="cluster-label " transform="translate(163.75, 172.00390625)"><foreignObject width="190" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Structured automaton</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M150,235.605L159.708,235.605C169.417,235.605,188.833,235.605,208.25,235.605C227.667,235.605,247.083,235.605,263.458,235.605C279.833,235.605,293.167,235.605,299.833,235.605L306.5,235.605" id="L_A_B_0" class=" edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style="" marker-end="url(#mermaid-1736451893652_flowchart-v2-pointEnd)"></path><path d="M484.5,71.602L488.667,71.602C492.833,71.602,501.167,71.602,523.75,71.602C546.333,71.602,583.167,71.602,619.335,72.868C655.503,74.134,691.007,76.667,708.758,77.934L726.51,79.2" id="L_C_D_1" class=" edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style="" marker-end="url(#mermaid-1736451893652_flowchart-v2-pointEnd)"></path><path d="M465.5,235.605L472.833,235.605C480.167,235.605,494.833,235.605,520.583,214.705C546.333,193.805,583.167,152.004,619.336,129.46C655.506,106.916,691.011,103.629,708.764,101.986L726.517,100.343" id="L_B_D_2" class=" edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style="" marker-end="url(#mermaid-1736451893652_flowchart-v2-pointEnd)"></path><path d="M980.5,88.402L984.667,88.402C988.833,88.402,997.167,88.402,1005.5,88.402C1013.833,88.402,1022.167,88.402,1029.833,88.402C1037.5,88.402,1044.5,88.402,1048,88.402L1051.5,88.402" id="L_D_E_3" class=" edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style="" marker-end="url(#mermaid-1736451893652_flowchart-v2-pointEnd)"></path></g><g class="edgeLabels"><g class="edgeLabel" transform="translate(208.25, 235.60546875)"><g class="label" transform="translate(-33.25, -13.6015625)"><foreignObject width="66.5" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel "><p>Defines</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(620, 71.6015625)"><g class="label" transform="translate(-85.5, -13.6015625)"><foreignObject width="171" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel "><p>Token distribution</p></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel "></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel "></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default  " id="flowchart-A-13" transform="translate(91.5, 235.60546875)"><rect class="basic label-container" style="fill:#e1f3f8 !important;stroke:#333 !important;stroke-width:2px !important" x="-58.5" y="-28.6015625" width="117" height="57.203125"></rect><g class="label" style="" transform="translate(-28.5, -13.6015625)"><rect></rect><foreignObject width="57" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Schema</p></span></div></foreignObject></g></g><g class="node default  " id="flowchart-B-14" transform="translate(388, 235.60546875)"><rect class="basic label-container" style="fill:#e1f3f8 !important;stroke:#333 !important;stroke-width:2px !important" x="-77.5" y="-28.6015625" width="155" height="57.203125"></rect><g class="label" style="" transform="translate(-47.5, -13.6015625)"><rect></rect><foreignObject width="95" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Logit Bias</p></span></div></foreignObject></g></g><g class="node default  " id="flowchart-C-15" transform="translate(388, 71.6015625)"><rect class="basic label-container" style="fill:#f8e1e1 !important;stroke:#333 !important;stroke-width:2px !important" x="-96.5" y="-28.6015625" width="193" height="57.203125"></rect><g class="label" style="" transform="translate(-66.5, -13.6015625)"><rect></rect><foreignObject width="133" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Language Model</p></span></div></foreignObject></g></g><g class="node default  " id="flowchart-D-16" transform="translate(855.5, 88.40234375)"><rect class="basic label-container" style="fill:#f8e1e1 !important;stroke:#333 !important;stroke-width:2px !important" x="-125" y="-28.6015625" width="250" height="57.203125"></rect><g class="label" style="" transform="translate(-95, -13.6015625)"><rect></rect><foreignObject width="190" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Constrained Sampling</p></span></div></foreignObject></g></g><g class="node default  " id="flowchart-E-20" transform="translate(1166.25, 88.40234375)"><rect class="basic label-container" style="fill:#e1f8e1 !important;stroke:#333 !important;stroke-width:2px !important" x="-110.75" y="-28.6015625" width="221.5" height="57.203125"></rect><g class="label" style="" transform="translate(-80.75, -13.6015625)"><rect></rect><foreignObject width="161.5" height="27.203125"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel "><p>Structured Output</p></span></div></foreignObject></g></g></g></g></g></svg>

From a technical perspective, an inference engine can modify the probability distribution for next-tokens by applying bias (often via logit masks) for all tokens from any given schemas. To apply these biases, [outlines](https://github.com/dottxt-ai/outlines) proposed guided generations via finite-state machine (FSM) for any given schemas (Willard & Louf, 2023). This allows us to track the current state during decoding and filter out invalid tokens by applying logit bias to the output.

<figure>
  <img src="/assets/figures/struct-decode-intro/constrained-json-fsm.webp" />
<figcaption>
courtesy of <a href="https://lmsys.org/blog/2024-02-05-compressed-fsm/" target="_blank"><i>LMSys, 2024</i></a>.
</figcaption>
</figure>

_in vLLM, you can use this by passing a JSON schema to the sampling params (either through Python SDK or HTTP requests)._

> Note: in some cases, it can even [improve](https://blog.dottxt.co/coalescence.html) the native decoding performance for LLM!

### Previous limitations in vLLM

There are few limitations with current vLLM's support of the Outlines backend:

1. **Slow decoding**: FSM has to be constructed at a token-level, meaning it can only transition the state one token per step. Therefore, it can only decode _one_ token at a time, resulting in slow decoding.
2. **Batch processing bottlenecks**: Implementation in [vLLM](https://github.com/vllm-project/vllm/blob/80c751e7f68ade3d4c6391a0f3fce9ce970ddad0/vllm/model_executor/guided_decoding/outlines_logits_processors.py) relies heavily on logit processor[^7]. As such, this is on the critical path of the sampling process. In batching use-case, compiling FSM per requests as well as computing the mask synchronous means that **all requests** in any given batches will get blocked, resulting in high time-to-first-tokens (TTFT) and lower throughput.
   - We found that compiling FSM is proven to be a relatively expensive task, making it a significant contributor to the increased TTFT.
3. **Performance issues with CFG mode**: With outlines integrations, while JSON mode is relatively fast, the CFG mode runs significantly slower, and can occasionally [crashes](https://github.com/vllm-project/vllm/issues/10081) the engine.
4. **Limited advanced feature support**: Techniques like [jump-forward decoding](https://lmsys.org/blog/2024-02-05-compressed-fsm/) are currently not possible with logit-processor approach. It requires prefilling a set of k-next tokens, whereas for logit processors we can only deal with the next-token.

### Integration with XGrammar

[XGrammar](https://github.com/mlc-ai/xgrammar) introduces a new technique that batch constrained decoding via pushdown automaton (PDA). You can think of a PDA as a "collection of FSMs, and each FSM represents a context-free grammar (CFG)." One significant advantage of PDA is its recursive nature, allowing us to execute multiple state transitions. They also include additional [optimisation](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) (for those who are interested) to reduce grammar compilation overhead.

This advancement addresses **limitation (1)** by moving grammar compilation out of Python into C, utilising `pthread`. Additionally, XGrammar lays the groundwork for addressing **limitation (4)** in future releases. Below are performance comparisons between the XGrammar and Outlines backends:

<figure>
  <img src="/assets/figures/struct-decode-intro/vllm-new-xgrammar.png" />
  <img src="/assets/figures/struct-decode-intro/vllm-xgrammar-decode-time-per-output-token.png" />
<figcaption>
courtesy of Michael Goin (Red Hat).
</figcaption>
</figure>

In vLLM’s v0 architecture, we've implemented XGrammar as a [logit processor](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/xgrammar_decoding.py), optimizing it with caching for tokenizer data. While the performance improvements are encouraging, we believe there's still significant room for optimization.

There are still a few usability concerns in XGrammar v0 integration to match feature parity with all use cases:

- It is yet to support grammars other than GBNF format (PR on vLLM: [github](https://github.com/vllm-project/vllm/pull/10870))
- It is yet to support regex
- It is yet to support complex JSON that uses regex patterns or numeric ranges
  - There are a few PR trying to cover this usage. There was one [bugfix PR on vLLM](https://github.com/vllm-project/vllm/pull/10899) and one [upstream](https://github.com/mlc-ai/xgrammar/pull/106)

> vLLM now has a basic support for XGrammar by default. In case where we know XGrammar is insufficient to serve the request, we fall back to Outlines.
>
> Note that vLLM also includes support for lm-format-enforcer. However, from our testing we found that in some long context test cases, lm-format-enforcer fails to enforce correct outputs, and not up to par with Outlines in terms of performance.

## Tentative plans for v1

With the release of [v1](https://github.com/vllm-project/vllm/issues/8779) on the horizon, we're working on a tentative plan for structured decoding:

1. Moving guided decoding towards scheduler-level [\[10\]](https://www.notion.so/Blog-4X-structured-decoding-speed-in-vLLM-8c3f2d44f6504202abbdb534983f2b2e?pvs=21)
   - Reason: We have more context regarding which requests that use structured decoding at a scheduler-level, therefore it shouldn't block other requests within the batch (tentatively addressing **limitation (2)**). In a sense, this moves guided decoding outside of the critical path.
   - This would allow for more natural vertical integration with jump-forward decoding (address **limitation (4)**).
2. Allowing bit-mask calculation in one process instead of each GPU workers
   - Reason: We can broadcast this bit-mask to each GPU worker instead of repeating this process per GPU worker.
   - We will look to carefully analyze the bandwidth implications of broadcasting masks for every sample per request that use guided decoding.
3. Good baseline for speculative decoding and tool-use
   - Reason: XGrammar includes plans to support tool-use, such that we can move away from Python's [tool parser](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers).
   - Tree scoring in speculative decoding can then use the same API as jump-forward decoding (which depends on the integration of guided decoding at the scheduler level).

_NOTE: if you have any more suggestions we are more than happy to take it into consideration. Consider joining [vLLM slack](https://www.notion.so/bentoml/slack.vllm.ai) via `#feat-structured-output`._

## Acknowledgements

We want to thank the vLLM team, XGrammar team, Michael Goin (Red Hat), Chendi Xue (Intel), and Russell Bryant (Red Hat) for their valuable feedback and collaboration on bringing XGrammar to vLLM and the continuous effort to improve structured decoding in vLLM.

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2016). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv preprint arXiv:1409.0473
- Haugeland, J. (1997). *Mind Design II: Philosophy, Psychology, and Artificial Intelligence*. The MIT Press. [https://doi.org/10.7551/mitpress/4626.001.0001](https://doi.org/10.7551/mitpress/4626.001.0001)
- Hendler, J. (2008). Avoiding Another AI Winter. *IEEE Intelligent Systems*, *23*(2), 2–4. [https://doi.org/10.1109/MIS.2008.20](https://doi.org/10.1109/MIS.2008.20)professional
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models*. arXiv preprint arXiv:2001.08361
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, *65*(6), 386–408. [https://doi.org/10.1037/h0042519](https://doi.org/10.1037/h0042519)
- Rumelhart, D. E., McClelland, J. L., & Group, P. R. (1986). *Parallel Distributed Processing, Volume 1: Explorations in the Microstructure of Cognition: Foundations*. The MIT Press. [https://doi.org/10.7551/mitpress/5236.001.0001](https://doi.org/10.7551/mitpress/5236.001.0001)
- Shortliffe, E. H. (1974). *MYCIN: A Rule-Based Computer Program for Advising Physicians Regarding Antimicrobial Therapy Selection* (Technical Report STAN-CS-74-465). Stanford University.
- Statistical Machine Translation. (n.d.). *IBM Models*. Statistical Machine Translation Survey. [http://www2.statmt.org/survey/Topic/IBMModels](http://www2.statmt.org/survey/Topic/IBMModels)
- Turing, A. M. (1950). i.—Computing Machinery And Intelligence. *Mind*, *LIX*(236), 433–460. [https://doi.org/10.1093/mind/LIX.236.433](https://doi.org/10.1093/mind/LIX.236.433)
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). *Attention Is All You Need*. arXiv preprint arXiv:1706.03762
- Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv preprint arXiv:2307.09702

---

[^1]:
    Allen Newell and Herbert Simon’s work at RAND initially showed that computers can simulate important aspects of intelligence.

    Another notable application was found in the medical domain (Haugeland, 1997). MYCIN, developed at Stanford University in the 1970s, diagnosed and recommended treatments for blood infections (Shortliffe, 1974). MYCIN’s developers recognized the importance of justifying recommendations, implementing what were known as “rule traces” to explain the system’s reasoning in human-understandable terms.

[^2]:
    In the 1990s, IBM released a sequence of complex statistical models that is trained to perform machine translations [tasks](https://en.wikipedia.org/wiki/IBM_alignment_models) (Statistical Machine Translation, n.d.) (see also: this [lecture](https://www.cs.cornell.edu/courses/cs5740/2017sp/lectures/08-alignments.pdf) from Cornell).

    In 2001, Bag of words (BoW)-variants model was trained on 0.3B tokens and was considered SOTA at the time (Mikolov et al., 2013). These earlier works proved to the research community that statistical modelling triumphs over symbolic counterpart for language processing given it can capture the general patterns for large corpuses of text.

[^3]:
    In 2017, The landmark paper “Attention is all You Need” introduced Transformers architecture (Vaswani et al., 2023\) for neural machine translations tasks, which is based on the attention mechanism first proposed by (Bahdanau et al., 2016).

    OpenAI then introduced the scaling law for neural language models (Kaplan et al., 2020), which sets off the race towards building these systems based on foundational language models.

[^4]:
    Prior to Attention-based transformers, seq-to-seq models uses RNNs given its ability for longer context length and better memory. However, they are more susceptible to vanishing/exploding gradients comparing to feed-forward network, and thus LSTM (Hochreiter & Schmidhuber, 1997) was proposed to solve this problem. Yet, one of the main problems with LSTM is that they tend to have poor memory recall with data they have seen many steps ago.

    The Attention paper addresses this problem by encoding additional positional data into the inputs. The paper also additionally proposed a encoder-decoder architecture for translation tasks, however, most of text-generation models nowadays are decoder-only, given its superior performance over zero-shot tasks.

    One of the many reasons why attention-based transformers works better than LSTM is because transformers are very scalable and hardware-aware (you can’t just arbitrary add more LSTM block and hope for better long-term retention). For more information, please refer back to the original paper.

[^5]:
    One might argue that we can reliably achieve these through few-shot promptings, i.e “Give me a JSON that yields the address of users. Example output can be …”. However, there is no guarantee that the generated outputs is a valid JSON. This is because these models are probabilistic systems, as they are “sampling” the next results based on the distribution of data that it was trained on.

    One might also argue that one should use specific fine-tuned models for JSON outputs to perform such cases. However, fine-tuning often requires extensive training and a lot more labor to curate data, monitor progress, and perform evaluation, which is a huge resources not everyone can afford to do.

[^6]: Note that the phrase "[structured/constrained/guided] decoding" are used interchangeably, but they all refer to the same mechanism of "using a format for the model to structurally sampling outputs.”

[^7]: See this [blog post](https://huggingface.co/blog/logits-processor-zoo) from HuggingFace for using logit processors to control the generation process.
