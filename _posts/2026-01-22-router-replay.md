---
layout: post
title: "Router Replay: Debugging and Observability for vLLM Semantic Router"
author: "vLLM Semantic Router Team"
image: /assets/logos/vllm-logo-text-light.png
---

## Expanding vLLM Semantic Router's Observability

Today, we're excited to introduce **Router Replay** — a powerful observability plugin for vLLM Semantic Router that captures routing decisions, matched signals, selected models, and request/response payloads for comprehensive debugging and analysis.

Debugging routing decisions in production LLM systems can be challenging. When a request is routed to the wrong model, or when you need to understand why a specific routing decision was made, having visibility into the decision-making process is crucial. Router Replay solves this by automatically capturing all routing metadata in a single, queryable record.

Router Replay enables you to:

- **Debug routing issues** by inspecting exactly which signals matched and why a specific model was selected
- **Audit routing decisions** by maintaining a record of all routing choices
- **Analyze routing patterns** to understand how your router behaves in production
- **Replay scenarios** by examining captured request/response pairs to reproduce issues

## Why Router Replay?

In a typical vLLM Semantic Router deployment, requests flow through multiple decision points:

1. **Signal Matching**: Keywords, embeddings, domains, and other signals are evaluated
2. **Decision Selection**: The router selects which decision (route) to use based on matched signals
3. **Model Selection**: A specific model is chosen from the decision's model references
4. **Request Processing**: The request is forwarded to the selected model
5. **Response Handling**: The response is processed and returned

Without Router Replay, understanding what happened at each step requires extensive logging, correlation of multiple log files, and manual reconstruction of the routing flow. Router Replay captures all of this information in a single, queryable record.

### The Problem: Invisible Routing Decisions

Consider a scenario where you're running a production router with multiple decisions:

- `math_route`: Routes math queries to a reasoning-capable model
- `code_route`: Routes programming questions to a code-specialized model  
- `general_route`: Handles general-purpose queries

When a user reports that their math question was routed incorrectly, you need to answer:

- Which signals matched?
- Why was this decision selected?
- What was the original request?
- Which model was actually used?
- Was the response cached?

Without Router Replay, answering these questions requires piecing together information from multiple sources — router logs, Envoy logs, and application logs — which is time-consuming and error-prone.

## How Router Replay Works

Router Replay is implemented as a plugin that integrates seamlessly into your routing decisions. When enabled, it automatically captures routing metadata for every request that matches a decision with the plugin configured.

### Basic Setup

Getting started with Router Replay is straightforward. First, initialize your configuration:

```bash
# Install vLLM Semantic Router CLI
pip install vllm-sr

# Initialize configuration
vllm-sr init
```

The initialization creates a `config.yaml` file with example configurations. You can add Router Replay to any decision by including it in the `plugins` section:

```yaml
decisions:
  - name: "math_route"
    description: "Route math queries with reasoning"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "router_replay"
        configuration:
          enabled: true
          max_records: 200              # Maximum records in memory
          capture_request_body: true    # Capture request payloads
          capture_response_body: false  # Capture response payloads
          max_body_bytes: 4096          # Max bytes per body (truncates if exceeded)
```

### Configuration Options

Router Replay offers flexible configuration options:

- **`enabled`**: Enable or disable the plugin (default: `true`)
- **`max_records`**: Maximum number of records to keep in memory (default: `200`, must be > 0)
- **`capture_request_body`**: Whether to capture request payloads (default: `false` for privacy)
- **`capture_response_body`**: Whether to capture response payloads (default: `false`)
- **`max_body_bytes`**: Maximum bytes to capture per body, truncates if exceeded (default: `4096`)

### Validating Your Configuration

Before deploying, validate your configuration:

```bash
vllm-sr validate --config config.yaml
```

This command checks that:
- Plugin configurations are valid
- Field constraints are satisfied (e.g., `max_records > 0`)
- All required fields are present
- Plugin types are correctly specified

Example validation output:

```
✓ Configuration is valid

Summary:
  Decisions: 3
  Plugins: 5 total (3 decisions)
    Types: router_replay: 1, semantic-cache: 2, system_prompt: 2
  Models: 2
  Signals: 4
```

### Starting the Router

Once configured, start the router:

```bash
vllm-sr serve
```

The router will start with Router Replay enabled for all configured decisions. Each routing decision that matches a request will automatically create a replay record.

## Accessing Replay Records

Router Replay exposes a RESTful API endpoint to query captured records. The endpoint is available at `/v1/router_replay` through Envoy (default port: `8888`), which routes requests through the ext_proc pipeline to the router.

> **Note**: Router Replay is currently accessible via the API endpoint. Dashboard UI integration for visualizing replay records is planned for future releases. For now, you can query records programmatically using the API or with tools like `curl` and `jq`.

### Listing All Records

To retrieve all captured records:

```bash
curl http://localhost:8888/v1/router_replay
```

Response:

```json
{
  "object": "router_replay.list",
  "count": 5,
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2025-01-15T10:30:45Z",
      "request_id": "req_abc123",
      "decision": "math_route",
      "category": "mathematics",
      "original_model": "openai/gpt-oss-120b",
      "selected_model": "openai/gpt-oss-120b",
      "reasoning_mode": "high",
      "signals": {
        "keyword": ["calculate", "equation"],
        "embedding": [],
        "domain": ["mathematics"]
      },
      "request_body": "{\"messages\":[{\"role\":\"user\",\"content\":\"Calculate 2+2\"}]}",
      "response_status": 200,
      "from_cache": false,
      "streaming": false
    }
  ]
}
```

### Retrieving a Specific Record

To get a specific record by ID:

```bash
curl http://localhost:8888/v1/router_replay/550e8400-e29b-41d4-a716-446655440000
```

### Understanding the Record Structure

Each replay record contains:

- **Routing Metadata**:
  - `decision`: The decision (route) that was matched
  - `category`: The semantic category selected
  - `original_model`: The model requested in the original request
  - `selected_model`: The model actually used
  - `reasoning_mode`: Whether reasoning was enabled and at what effort level

- **Signal Information**:
  - `signals.keyword`: Matched keyword signals
  - `signals.embedding`: Matched embedding signals
  - `signals.domain`: Matched domain signals
  - `signals.fact_check`: Matched fact-check signals
  - `signals.user_feedback`: Matched user feedback signals
  - `signals.preference`: Matched preference signals

- **Request/Response Data** (if enabled):
  - `request_body`: The original request payload (truncated if exceeds `max_body_bytes`)
  - `response_body`: The response payload (if `capture_response_body` is enabled)
  - `request_body_truncated`: Boolean indicating if request was truncated
  - `response_body_truncated`: Boolean indicating if response was truncated

- **Status Information**:
  - `response_status`: HTTP status code
  - `from_cache`: Whether the response came from cache
  - `streaming`: Whether the response was streamed

## Advanced Configuration

### Storage Backends

By default, Router Replay uses in-memory storage, which is perfect for development and debugging. The in-memory backend is production-ready and provides fast access to recent routing decisions.

For production deployments requiring persistence across restarts or distributed storage, Router Replay supports additional storage backends including Redis, PostgreSQL, and Milvus. These backends are available in the codebase and can be configured via the `store_backend` option in your configuration. For detailed configuration examples and setup instructions, refer to the [vLLM Semantic Router documentation](https://github.com/vllm-project/semantic-router).

### Per-Decision Isolation

Router Replay automatically creates isolated storage per decision. This means:

- Each decision maintains its own record collection
- Records are prefixed with the decision name
- You can configure different storage backends per decision
- Records are automatically namespaced to prevent collisions

### Privacy and Security Considerations

Router Replay respects privacy by default:

- **Request/response bodies are not captured by default** — you must explicitly enable `capture_request_body` or `capture_response_body`
- **Body truncation** — even when enabled, bodies are truncated to `max_body_bytes` to prevent excessive storage
- **TTL support** — persistent backends support automatic expiration via `ttl_seconds`
- **Isolated storage** — each decision's records are stored separately

For production deployments handling sensitive data:

```yaml
plugins:
  - type: "router_replay"
    configuration:
      enabled: true
      capture_request_body: false   # Don't capture request bodies
      capture_response_body: false  # Don't capture response bodies
      # Only capture routing metadata
```

## Real-World Use Cases

### Use Case 1: Debugging Incorrect Routing

**Scenario**: A user reports that their math question was routed to the general model instead of the math-specialized model.

**Solution**: Query Router Replay records to see what happened:

```bash
# Get the most recent records
curl http://localhost:8888/v1/router_replay | jq '.data[0]'
```

Inspect the record to see:
- Which signals matched (or didn't match)
- Why the decision was selected
- What the original request looked like

This helps you understand if:
- The keyword signal wasn't configured correctly
- The embedding similarity threshold was too low
- The domain classifier missed the math domain

### Use Case 2: Analyzing Routing Patterns

**Scenario**: You want to understand which models are being used most frequently and for what types of queries.

**Solution**: Query Router Replay records and analyze the data:

```bash
# Get all records and analyze
curl http://localhost:8888/v1/router_replay | jq '.data[] | {decision, selected_model, category}'
```

This gives you insights into:
- Model usage distribution
- Category distribution
- Decision selection patterns

### Use Case 3: Reproducing Production Issues

**Scenario**: A user reports a specific error, and you need to reproduce it locally.

**Solution**: Use Router Replay to capture the exact request:

```bash
# Get the specific record
curl http://localhost:8888/v1/router_replay/{record_id} | jq '.request_body'
```

Then replay the request locally to debug the issue.

### Use Case 4: Auditing and Compliance

**Scenario**: You need to maintain an audit log of routing decisions for compliance.

**Solution**: Configure Router Replay with a persistent storage backend (when available) and long TTL. For now, use the in-memory backend with appropriate `max_records` to maintain recent audit history:

```yaml
plugins:
  - type: "router_replay"
    configuration:
      enabled: true
      max_records: 10000  # Increase for longer audit history
      capture_request_body: false  # Privacy: don't capture bodies
      capture_response_body: false
      # Only capture routing metadata for compliance
```

## Performance Considerations

Router Replay is designed to be lightweight and non-intrusive:

- **Minimal overhead**: Recording happens asynchronously and doesn't block request processing
- **Configurable storage**: Use in-memory for development, persistent backends for production
- **Body truncation**: Automatically truncates large bodies to prevent storage bloat
- **TTL support**: Automatic expiration prevents unbounded growth
- **Async writes**: Optional async writes for persistent backends improve performance

For high-throughput scenarios:

```yaml
plugins:
  - type: "router_replay"
    configuration:
      enabled: true
      capture_request_body: false      # Reduce storage
      capture_response_body: false     # Reduce storage
      async_writes: true                # Non-blocking writes
      max_records: 100                  # Smaller memory footprint
```

## Get Started Today

Router Replay is available now in vLLM Semantic Router. To get started:

1. **Install the CLI**: `pip install vllm-sr`
2. **Initialize configuration**: `vllm-sr init`
3. **Enable Router Replay** in your decisions
4. **Validate**: `vllm-sr validate --config config.yaml`
5. **Start the router**: `vllm-sr serve`
6. **Query records**: `curl http://localhost:8888/v1/router_replay`

For more information, see the [vLLM Semantic Router documentation](https://github.com/vllm-project/semantic-router) and the [Router Replay API reference](https://github.com/vllm-project/semantic-router).

## Acknowledgments

Special thanks to the vLLM Semantic Router community for feedback and contributions. Router Replay was designed based on real-world debugging needs reported by users deploying production routing systems.

We're incredibly excited about this new observability capability, and we can't wait to see how you use Router Replay to debug and optimize your routing systems. Let's get to work!

Subscribe 

* © 2026. vLLM Team. All rights reserved.

vLLM is a fast and easy-to-use library for LLM inference and serving.

