---
layout: post
title: "Enabling the Responses API and MCP on vLLM"
author: "Meta"
image: /assets/logos/vllm-logo-text-light.png
---

The OpenAI **Responses API** is the successor to the Chat Completions API, designed to support agentic workflows with built-in tool use, multi-turn conversation management, and streaming. We have implemented the Responses API in vLLM, enabling any model served by vLLM to participate in agentic pipelines that call tools, execute code, search the web, and reason through complex tasks -- all through a single `POST /v1/responses` endpoint.

This blog post covers:

- **The Responses API implementation** in vLLM: endpoint design, streaming and non-streaming modes, and the full set of supported features
- **MCP (Model Context Protocol) integration**: how vLLM connects to external tool servers and executes tool calls during generation
- **Two context architectures**: HarmonyContext for GPT-OSS models and ParsableContext for all other models

## The Responses API

### Endpoint Overview

vLLM exposes three endpoints under the Responses API:

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
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "input": "What is the capital of France?",
    "stream": false
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

This event structure matches the OpenAI Responses API specification, making vLLM a drop-in replacement for clients already using the Responses API.

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "input": "Explain quicksort in one paragraph.",
    "stream": true
  }'
```

### Supported Features

The vLLM Responses API supports a broad set of features:

**Function Calling.** Tools of type `function` can be defined in the `tools` list. The model's output is parsed for tool calls using configurable tool parsers (Hermes, Llama, Mistral, etc.). The `tool_choice` parameter supports `"auto"`, `"none"`, `"required"`, or a named function. When the model emits a function call, it is returned as a `function_call` output item with streaming events `response.function_call_arguments.delta` and `response.function_call_arguments.done`.

**Reasoning.** The `reasoning` parameter (with an `effort` field) enables chain-of-thought reasoning. Reasoning content is tracked separately from regular output and appears as `ResponseReasoningItem` output items. Streaming emits `response.reasoning_text.delta` and `response.reasoning_text.done` events, allowing clients to display the model's thinking process in real time.

**Structured Output.** The `text.format` field supports JSON Schema-constrained generation. When a JSON schema is provided, vLLM enforces the schema during decoding using guided generation, ensuring the output is valid JSON conforming to the specified schema.

**Logprobs.** When `include` contains `"message.output_text.logprobs"`, the response includes per-token log probabilities. The `top_logprobs` parameter controls how many top alternatives are returned per token position.

**Background Mode.** Setting `background: true` (with `store: true`) queues the response for asynchronous generation. The response is returned immediately with status `"queued"` and can be polled or retrieved later via `GET /v1/responses/{response_id}`. Background responses can be cancelled via the cancel endpoint.

**Conversation Continuation.** The `previous_response_id` field chains responses together for multi-turn conversations. The previous response's output items are prepended to the new request's input, enabling stateful multi-turn dialogue without the client needing to manage conversation history.

**vLLM-Specific Extensions.** Beyond the standard API, vLLM adds parameters for `priority` (request scheduling priority), `cache_salt` (prefix cache isolation), `seed` (deterministic sampling), `repetition_penalty`, custom `stop` sequences, and `enable_response_messages` (returns raw prompt and output token IDs for debugging).

## MCP: Model Context Protocol Integration

The **Model Context Protocol (MCP)** allows LLMs to call external tools during generation. vLLM implements MCP as a first-class feature of the Responses API: when a model generates a tool call, vLLM intercepts it, calls the appropriate MCP tool server, and feeds the result back to the model for the next turn of generation -- all within a single API request.

### Built-in Tools

vLLM supports three categories of built-in tools:

**Web Search** (`web_search_preview`). Enables the model to search the web during generation. Streaming events follow: `response.web_search_call.in_progress` -> `response.web_search_call.searching` -> `response.web_search_call.completed`. The search results are injected back into the conversation for the model to synthesize.

**Code Interpreter** (`code_interpreter`). Enables the model to write and execute Python code in a sandboxed Docker environment. Streaming events include `response.code_interpreter_call_code.delta` for the generated code and `response.code_interpreter_call.completed` for the execution result.

**Container** (`container`). Enables the model to execute shell commands in a stateful Docker container, supporting arguments like `cmd`, `workdir`, `env`, and `timeout`.

### Tool Server Architecture

vLLM provides two `ToolServer` implementations:

**`MCPToolServer`**: Connects to external MCP-compatible tool servers over SSE. Multiple servers can be specified via comma-separated URLs. Each server exposes its tools via the MCP protocol, and vLLM discovers available tools at startup via `session.list_tools()`. Tool sessions are created per-request with unique session IDs.

**`DemoToolServer`**: A lightweight local alternative that uses built-in tool implementations (Exa-based web search, Docker-based Python execution) without requiring an external MCP server. This is useful for development and testing.

```bash
# Starting vLLM with an MCP tool server
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tool-server-url http://localhost:3001/sse
```

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

## Context Architecture: HarmonyContext vs. ParsableContext

A key design decision in the Responses API is how to manage the conversation state during multi-turn tool-calling loops. vLLM implements two context architectures to support different model families.

### HarmonyContext (GPT-OSS Models)

`HarmonyContext` is designed for GPT-OSS models that use OpenAI's Harmony message format. These models use a channel-based parsing system where the model's output is split into channels (`analysis` for reasoning, `commentary`, and `final` for the actual response). The context tracks messages in the Harmony `Message` format and uses the Harmony tokenizer's `render_for_completion()` to produce token IDs for the next turn.

Key characteristics:
- Uses `openai_harmony` message types (`Author`, `Message`, `Role`, `StreamState`, `TextContent`)
- Tool recipients are identified by message `recipient` field (e.g., `browser.search`, `python`, `container.exec`)
- Token rendering uses the Harmony encoding's stop tokens for assistant actions
- Per-turn token metrics (input, output, cached, tool output tokens) are tracked for accurate usage reporting

`StreamingHarmonyContext` extends this for token-by-token streaming, processing each token through the Harmony parser and tracking parser state transitions to emit the correct streaming events.

### ParsableContext (All Other Models)

`ParsableContext` is the context for non-GPT-OSS models (Llama, Mistral, Qwen, etc.). It uses vLLM's standard chat template system to render conversations and parses tool calls from the model output using configurable tool parsers.

Key characteristics:
- Uses `ResponseInputOutputItem` types from the OpenAI SDK (e.g., `ResponseFunctionToolCall`, `ResponseFunctionToolCallOutputItem`)
- Tool calls are identified by the `name` field matching built-in tool names (`code_interpreter`, `web_search_preview`, `container`)
- Prompt rendering uses vLLM's chat template system via `_render_next_turn()`
- Supports the `VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT` environment variable to enable this context

### SimpleContext

For non-tool-calling scenarios, `SimpleContext` provides a lightweight context that accumulates raw text and token IDs without any parsing overhead. It is the default for models that do not have tool use enabled.

### Choosing the Right Context

The context is selected automatically based on the model type and request configuration:

| Condition | Context Used |
|---|---|
| GPT-OSS model, streaming | `StreamingHarmonyContext` |
| GPT-OSS model, non-streaming | `HarmonyContext` |
| Non-GPT-OSS, tools enabled, experimental parser | `ParsableContext` |
| Non-GPT-OSS, no tools or simple request | `SimpleContext` |

## Token Usage Tracking

The Responses API provides detailed token usage information in the response, including:

- `input_tokens`: Total prompt tokens across all turns
- `output_tokens`: Total generated tokens
- `input_tokens_details.cached_tokens`: Tokens served from the prefix cache
- `output_tokens_details.reasoning_tokens`: Tokens spent on reasoning (chain-of-thought)

For multi-turn tool-calling interactions, the `TurnMetrics` class tracks per-turn metrics. Tool output tokens are calculated as the difference between consecutive turns' prompt sizes minus the previous turn's output, capturing the token cost of tool results injected between turns.

## Getting Started

To use the Responses API with vLLM:

```bash
# Basic serving
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct

# With tool calling support
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# With MCP tool server
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tool-server-url http://localhost:3001/sse
```

Then use the OpenAI Python SDK to make requests:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Non-streaming
response = client.responses.create(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    input="What is the capital of France?",
)
print(response.output_text)

# Streaming
stream = client.responses.create(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    input="Explain quicksort step by step.",
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)

# With function calling
response = client.responses.create(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
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

## Future Work

- **Response storage with eviction policies**: Currently, stored responses are held in memory with no eviction. We plan to add configurable TTLs and storage backends.
- **Expanded tool support**: Adding more built-in tool types and improving MCP server discovery.
- **Parallel tool calls**: Improving support for models that emit multiple tool calls in a single turn.
- **Optimized multi-turn performance**: Leveraging prefix caching more aggressively across tool-calling turns to reduce redundant computation.

To see more details about the future work and explore opportunities to contribute, please see this vLLM feature development map: https://github.com/vllm-project/vllm/issues/34857

## Acknowledgements

This work was a collaboration across the vLLM community. Thanks to all contributors who helped design and implement the Responses API and MCP integration.

**Meta**: Andrew Xia, Daniel Salib, Ye Hu, Alec Solder, Ye (Charlotte Qi)

**vLLM**: Chauncey Jiang
