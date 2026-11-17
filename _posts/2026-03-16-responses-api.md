---
layout: post
title: "Enabling ResponsesAPI and MCP on vLLM"
author: "Meta"
image: /assets/logos/vllm-logo-text-light.png
---

The OpenAI **ResponsesAPI** is the successor to the Chat Completions API, designed to support agentic workflows with built-in tool use, multi-turn conversation management, and streaming. We have implemented the ResponsesAPI in vLLM, enabling any model served by vLLM to participate in agentic pipelines that call tools, execute code, search the web, and reason through complex tasks -- all through a single `POST /v1/responses` endpoint.

This blog post covers:

- **ResponsesAPI implementation** in vLLM: endpoint design, streaming and non-streaming modes, and the full set of supported features
- **MCP (Model Context Protocol) integration**: how vLLM connects to external tool servers and executes tool calls during generation
- **Context architectures**: HarmonyContext for GPT-OSS models and ParsableContext/SimpleContext for all other models

## ResponsesAPI

**ResponsesAPI** is a modern interface for interacting with large language models that unifies text generation, multimodal inputs, and tool use into a single API primitive. Introduced as the successor to earlier interfaces like Chat Completions and Assistants, it provides a flexible abstraction for building agentic applications—allowing models to generate structured outputs, call tools, maintain conversation state, and integrate external data sources in one request. The API treats a “response” as the fundamental unit of interaction, combining inputs, model reasoning, tool calls, and outputs into a structured object. Developers can learn more about the official specification in the OpenAI Responses [API documentation](https://developers.openai.com/api/reference/resources/responses). At the same time, efforts like the [OpenResponses Initiative](https://www.openresponses.org/) aim to define an open, provider-agnostic standard inspired by this interface, enabling interoperable tooling and reducing vendor lock-in across LLM platforms.

### Endpoint Overview

vLLM exposes three endpoints under ResponsesAPI:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/responses` | POST | Create a new response (streaming or non-streaming) |
| `/v1/responses/{response_id}` | GET | Retrieve a stored response |
| `/v1/responses/{response_id}/cancel` | POST | Cancel a background response |

The primary endpoint is `POST /v1/responses`. It accepts a `ResponsesRequest` with an `input` (a string or a list of conversation items), an optional `instructions` field for system messages, and a `tools` list for tool definitions. The `stream` flag determines whether the response is returned as a single JSON object or as a stream of Server-Sent Events (SSE).

### Non-Streaming Mode

In non-streaming mode, vLLM generates the full response before returning it. The response includes the model's output items (text messages, function calls, reasoning content), token usage statistics, and a final status (`completed`, `incomplete`, or `failed`).

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "input": "What is the capital of France?"
  }'
```

### Streaming Mode

When `stream: true`, vLLM emits events as SSE with monotonically increasing sequence numbers. The event lifecycle follows a structured pattern:

1. `response.created` and `response.in_progress` -- the response object is initialized
2. For each output item:
   - `response.output_item.added` -- a new output item begins (message, reasoning, function_call, etc.)
   - Content-specific delta events (e.g., `response.output_text.delta` for text tokens)
   - Content done events (e.g., `response.output_text.done`)
   - `response.output_item.done` -- the output item is finalized
3. `response.completed` -- the full response with usage statistics

This event structure matches the OpenAI ResponsesAPI specification, making vLLM a drop-in replacement for clients already using ResponsesAPI.

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "input": "Explain quicksort in one paragraph.",
    "stream": true
  }'
```

See [OpenResponses](https://www.openresponses.org/specification#streaming) for more details on the streaming protocol and the types of events that are streamed.

### Supported Features

The vLLM ResponsesAPI supports a broad set of features. To see the vLLM-specific implementation of ResponsesAPI:
- ResponsesRequest defined [here](https://github.com/vllm-project/vllm/blob/4ed51308c8826619459be858a6dc4333206f41c1/vllm/entrypoints/openai/responses/protocol.py#L140)
- ResponsesResponse is defined [here](https://github.com/vllm-project/vllm/blob/4ed51308c8826619459be858a6dc4333206f41c1/vllm/entrypoints/openai/responses/protocol.py#L468)

**Tool Calling (Function and MCP)**
ResponsesAPI distinguishes between two tool types:

- **Function tools** (`"type": "function"`): Tools defined inline with a JSON schema. The model generates a `function_call` output item, but the **client** is responsible for executing the function and returning the result. This follows a client-side execution model. See [this script](https://github.com/vllm-project/vllm/blob/4ed51308c8826619459be858a6dc4333206f41c1/examples/online_serving/openai_responses_client_with_tools.py) for an example of multi-turn function tools.
- **MCP tools** (`"type": "mcp"`): Tools hosted on external MCP servers. **vLLM itself** acts as the MCP client—it intercepts the tool call, executes it against the MCP server, injects the result back into the conversation, and continues generating. This happens entirely server-side within a single API request (see the [MCP section](#mcp-model-context-protocol-integration) below). See [this script](https://github.com/vllm-project/vllm/blob/4ed51308c8826619459be858a6dc4333206f41c1/examples/online_serving/openai_responses_client_with_mcp_tools.py) for an example of calling MCP tools with GPT-OSS.

Tools can be defined in the `tools` list in the request. The model's output is parsed for tool calls using configurable tool parsers, which is set with the `--tool-call-parser` flag.

**Reasoning.** The `reasoning` parameter (with an `effort` field) enables chain-of-thought reasoning. Reasoning content is tracked separately from regular output and appears as `ResponseReasoningItem` output items. Streaming emits `response.reasoning_text.delta` and `response.reasoning_text.done` events, allowing clients to display the model's thinking process in real time (see https://github.com/vllm-project/vllm/pull/29947)

**Structured Output.** The `structured_outputs` field supports JSON Schema-constrained generation. When a JSON schema is provided, vLLM enforces the schema during decoding using guided generation, ensuring the output is valid JSON conforming to the specified schema. When a choice is specified, vLLM will only produce output from the options listed. (see this [PR](https://github.com/vllm-project/vllm/pull/33709) for more context).

**Logprobs.** When `include` contains `"message.output_text.logprobs"`, the response includes per-token log probabilities. The `top_logprobs` parameter controls how many top alternatives are returned per token position.

**vLLM-Specific Extensions.** Beyond the standard API, vLLM adds parameters for `priority` (request scheduling priority), `cache_salt` (prefix cache isolation), `seed` (deterministic sampling), `repetition_penalty`, custom `stop` sequences, and `enable_response_messages` (returns raw prompt and output token IDs for debugging).

**Debugging.** We have also implemented the ability to return raw input and output tokens for ResponsesAPI. You can enable this by using the `enable_response_messages` flag. (See https://github.com/vllm-project/vllm/pull/29549)

## MCP: Model Context Protocol Integration

The **Model Context Protocol (MCP)** allows LLMs to call external tools during generation, with vLLM handling the tool execution server-side -- unlike function tools, where the client is responsible for handling tool calls. vLLM implements MCP as a first-class feature of ResponsesAPI: when a model generates a tool call, vLLM intercepts it, calls the appropriate MCP tool server, and feeds the result back to the model for the next turn of generation -- all within a single API request.

### Built-in Tools

vLLM supports three categories of built-in tools:

**Web Search** (`web_search_preview`). Enables the model to search the web during generation. Streaming events follow: `response.web_search_call.in_progress` -> `response.web_search_call.searching` -> `response.web_search_call.completed`. The search results are injected back into the conversation for the model to synthesize.

**Code Interpreter** (`code_interpreter`). Enables the model to write and execute Python code in a sandboxed Docker environment. Streaming events include `response.code_interpreter_call_code.delta` for the generated code and `response.code_interpreter_call.completed` for the execution result.

**Container** (`container`). Enables the model to execute shell commands in a stateful Docker container, supporting arguments like `cmd`, `workdir`, `env`, and `timeout`.

### Tool Server Architecture

vLLM provides the following `ToolServer` implementation:

**`MCPToolServer`**: Connects to external MCP-compatible tool servers over SSE. Multiple servers can be specified via comma-separated URLs. Each server exposes its tools via the MCP protocol, and vLLM discovers available tools at startup via `session.list_tools()`. Tool sessions are created per-request with unique session IDs.

### Agentic Loop

The core of MCP integration is the agentic loop in `_generate_with_builtin_tools`. This loop:

1. Generates tokens from the model
2. Checks if the model requested a tool call (`need_builtin_tool_call()`)
3. If yes, calls the tool via the MCP session (`call_tool()`)
4. Appends the tool result to the conversation context
5. Renders the updated conversation as a new prompt
6. Repeats from step 1

This loop continues until the model produces a final response without requesting a tool call, or until the `max_tool_calls` limit is reached. The entire multi-turn interaction happens server-side within a single API request.

### MCP Streaming Events

When streaming is enabled with MCP tools, vLLM emits fine-grained events for each tool call:

```
response.mcp_call.in_progress        -- tool call begins
response.mcp_call_arguments.delta    -- argument tokens stream in
response.mcp_call_arguments.done     -- arguments are complete
response.mcp_call.completed          -- tool execution result is available
```

This allows clients to display the model's tool interactions in real time, showing what tool is being called, with what arguments, and what the result was.

## Context Architecture

A key design decision in the ResponsesAPI is how to manage the conversation state during multi-turn tool-calling loops. vLLM implements the following context architectures to support different model families.

### HarmonyContext (GPT-OSS Models)

`HarmonyContext` is designed for GPT-OSS models that use OpenAI's Harmony message format. See [OpenAI's harmony guide](https://developers.openai.com/cookbook/articles/openai-harmony) for more context. These models use a channel-based parsing system where the model's output is split into channels (`analysis` for reasoning, `commentary`, and `final` for the actual response). The context tracks messages in the Harmony `Message` format and uses the Harmony tokenizer's `render_for_completion()` to produce token IDs for the next turn.

Key characteristics:
- Uses `openai_harmony` message types (`Author`, `Message`, `Role`, `StreamState`, `TextContent`)
- Tool recipients are identified by message `recipient` field (e.g., `browser.search`, `python`, `container.exec`)
- Token rendering uses the Harmony encoding's stop tokens for assistant actions
- Per-turn token metrics (input, output, cached, tool output tokens) are tracked for accurate usage reporting

`StreamingHarmonyContext` extends this for token-by-token streaming, processing each token through the Harmony parser and tracking parser state transitions to emit the correct streaming events.

### ParsableContext (MCP for All Other Models)

`ParsableContext` is the context for non-GPT-OSS models (Llama, Mistral, Qwen, etc.). It uses vLLM's standard chat template system to render conversations and parses tool calls from the model output using configurable tool parsers.

Key characteristics:
- Uses `ResponseInputOutputItem` types from the OpenAI SDK (e.g., `ResponseFunctionToolCall`, `ResponseFunctionToolCallOutputItem`)
- Tool calls are identified by the `name` field matching built-in tool names (`code_interpreter`, `web_search_preview`, `container`)
- Prompt rendering uses vLLM's chat template system via `_render_next_turn()`
- Use the `VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT` environment variable to enable this context

### SimpleContext

For non-MCP-tool-calling scenarios, `SimpleContext` provides a lightweight context that accumulates raw text and token IDs without any parsing overhead. It is the default for models that do not have tool use enabled.

Eventually, all three context architectures will be merged into a single, unified context architecture.

## Metrics & Token Usage Tracking

ResponsesAPI provides detailed token usage information in the response, including:

- `input_tokens`: Total prompt tokens across all turns
- `output_tokens`: Total generated tokens
- `input_tokens_details.cached_tokens`: Tokens served from the prefix cache
- `output_tokens_details.reasoning_tokens`: Tokens spent on reasoning (chain-of-thought)  ([PR](https://github.com/vllm-project/vllm/pull/33513))

For multi-turn tool-calling interactions, the `TurnMetrics` class tracks per-turn metrics. Tool output tokens are calculated as the difference between consecutive turns' prompt sizes minus the previous turn's output, capturing the token cost of tool results injected between turns.

## Evals

With vLLM's ResponsesAPI implementation, we were able to replicate Kimi K2's HLE score of 23.9. We used the open-source HLE test harness with OpenAI's o3-mini as a judge. We also ran GPT-OSS against the vLLM ResponsesAPI with MCP tools (including browser, python, and container). With high reasoning on GPT-OSS 120B, we achieved a score of 0.97 on AIME 2025, which matches OpenAI's GPT-OSS model card.

## Getting Started

To use the ResponsesAPI with vLLM:

```bash
# with Tool calling and reasoning
vllm serve Qwen/Qwen3-8B \
--reasoning-parser qwen3 \
--tool-call-parser qwen3 \
--enable-auto-tool-choice
```

Then use the OpenAI Python SDK to make requests:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Non-streaming
response = client.responses.create(
    model="Qwen/Qwen3-8B",
    input="What is the capital of France?",
)
print(response.output_text)

# Streaming
stream = client.responses.create(
    model="Qwen/Qwen3-8B",
    input="Explain quicksort step by step.",
    stream=True,
)
for event in stream:
    print(event)

# With function calling
response = client.responses.create(
    model="Qwen/Qwen3-8B",
    input="What is the weather in San Francisco?",
    tools=[{
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }],
)
```

To use our MCP server:
```bash
# With MCP tool server
# see https://github.com/vllm-project/vllm/pull/29798
VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT=1 \
vllm serve moonshotai/Kimi-K2-Thinking \
--trust-remote-code \
--tensor-parallel-size 8 \
--enable-auto-tool-choice \
--tool-call-parser kimi_k2 \
--reasoning-parser kimi_k2 \
--tool-server=localhost:8081/container,localhost:8081/browser,localhost:8081/python


# with MCP calling
curl -X POST "http://localhost:8000/v1/responses"   -H "Content-Type: application/json"   -H "Authorization: Bearer dummy-api-key"   -d '{
        "model": "moonshotai/Kimi-K2-Thinking",
        "input": "Multiply 64548*15151 using the python tool.",
        "tools": [
          {
            "type": "mcp",
            "server_label": "code_interpreter",
            "headers": {"test": "test"},
            "server_url": "IGNORED"
          }
        ]
      }'
```

## Future Work

- **Offloading response storage & API Layer**: Currently, stored responses are held in memory with no eviction. We would like to support offloading ResponsesAPI state management to a third-party database, and potentially offload the API layer outside of the core vLLM engine (see https://github.com/vllm-project/vllm/issues/26934)
- **Expanded tool support**: Add MCP support for all tools. (See https://github.com/vllm-project/vllm/issues/30115)
- **Open Responses Conformity**: [OpenResponses](https://www.openresponses.org/), launched in January 2026, is an open initiative to standardize a vendor-neutral API for LLM interactions based on ResponsesAPI abstraction, covering structured outputs, tool calls, multimodal inputs, and reasoning traces. For vLLM, supporting OpenResponses enables compatibility with a growing ecosystem of agent frameworks and SDKs while giving users a portable interface that works across both hosted APIs and open-source model deployments.

To see more details about the future work and explore opportunities to contribute, please see this vLLM feature development map for H1 2026: https://github.com/vllm-project/vllm/issues/34857

## Acknowledgements

This work was a collaboration across the vLLM community. Thanks to all contributors who helped design and implement ResponsesAPI and MCP integration, including the following (but not limited to):

**Meta**: Andrew Xia, Daniel Salib, Ye Hu, Zhiwei Zhao, Alec Solder, Ye (Charlotte) Qi

**DaoCloud**: Chauncey Jiang

**RedHat**: Flora Feng, Ben Browning, Michael Goin
