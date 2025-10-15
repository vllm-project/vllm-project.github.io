---
layout: post
title: "Transformers backend integration in vLLM"
author: "The Hugging Face Team"
image: /assets/figures/transformers-backend/transformers-backend.png
---

The [Hugging Face Transformers library](https://huggingface.co/docs/transformers/main/en/index)
offers a flexible, unified interface to a vast ecosystem of model architectures. From research to
fine-tuning on custom dataset, transformers is the go-to toolkit for all.

But when it comes to *deploying* these models at scale, inference speed and efficiency often take
center stage. Enter [vLLM](https://docs.vllm.ai/en/latest/), a library engineered for high-throughput
inference, pulling models from the Hugging Face Hub and optimizing them for production-ready performance.

A recent addition to the vLLM codebase enables leveraging transformers as a backend to run models.
vLLM will therefore optimize throughput/latency on top of existing transformers architectures.
In this post, we’ll explore how vLLM leverages the transformers backend to combine **flexibility**
with **efficiency**, enabling you to deploy state-of-the-art models faster and smarter.

## Updates

This section will hold all the updates that have taken place since the blog post was first released (11th April 2025).

### Support for Vision Language Models (21st July 2025)

vLLM with the transformers backend now supports **Vision Language Models**. When user adds `model_impl="transformers"`,
the correct class for text-only and multimodality will be deduced and loaded.

Here is how one can serve a multimodal model using the transformers backend.
```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
--model_impl transformers \
```

To consume the model one can use the `openai` API like so:
```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
chat_response = client.chat.completions.create(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
                },
            },
        ],
    }],
)
print("Chat response:", chat_response)
```

You can also directly initialize the vLLM engine using the `LLM` API. Here is the same model being
served using the `LLM` API.

```python
from vllm import LLM, SamplingParams
from PIL import Image
import requests
from transformers import AutoProcessor

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
hf_processor = AutoProcessor.from_pretrained(model_id) # required to dynamically update the chat template

messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "url": "dummy_image.jpg"},
          {"type": "text", "text": "What is the content of this image?"},
        ],
    },
]
prompt = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

# initialize the vlm using the `model_impl="transformers"`
vlm = LLM(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    model_impl="transformers",
)

outputs = vlm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    },
    sampling_params=SamplingParams(max_tokens=100)
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

# OUTPUTS:
# In the tranquil setting of this image, two feline companions are enjoying a peaceful slumber on a
# cozy pink couch. The couch, adorned with a plush red fabric across the seating area, serves as their perfect resting place.
#
# On the left side of the couch, a gray tabby cat is curled up at rest, its body relaxed in a display
# of feline serenity. One paw playfully stretches out, perhaps in mid-jump or simply exploring its surroundings.
```

## Transformers and vLLM: Inference in Action

Let’s start with a simple text generation task using the `meta-llama/Llama-3.2-1B` model to see how
these libraries stack up.

**Infer with transformers**

The transformers library shines in its simplicity and versatility. Using its `pipeline` API, inference is a breeze:

```py
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
result = pipe("The future of AI is")

print(result[0]["generated_text"])
```

This approach is perfect for prototyping or small-scale tasks, but it’s not optimized for high-volume
inference or low-latency deployment.

**Infer with vLLM**

vLLM takes a different track, prioritizing efficiency with features like `PagedAttention`
(a memory-efficient attention mechanism) and dynamic batching. Here’s the same task in vLLM:

```py
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("The future of AI is", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

vLLM’s inference is noticeably faster and more resource-efficient, especially under load.
For example, it can handle thousands of requests per second with lower GPU memory usage.

## vLLM’s Deployment Superpower: OpenAI Compatibility

Beyond raw performance, vLLM offers an OpenAI-compatible API, making it a drop-in replacement for
external services. Launch a server:

```bash
vllm serve meta-llama/Llama-3.2-1B
```

Then query it with curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'
```

Or use Python’s OpenAI client:

```py
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
completion = client.completions.create(
    model="meta-llama/Llama-3.2-1B",
    prompt="San Francisco is a",
    max_tokens=7,
    temperature=0
)
print("Completion result:", completion.choices[0].text)
```

This compatibility slashes costs and boosts control, letting you scale inference locally with vLLM’s optimizations.

## Why do we need the transformers backend?

The transformers library is optimized for contributions and
[addition of new models](https://huggingface.co/docs/transformers/en/add_new_model). Adding a new
model to vLLM on the other hand is a little
[more involved](https://docs.vllm.ai/en/latest/contributing/model/index.html).

In the **ideal world**, we would be able to use the new model in vLLM as soon as it is added to
transformers. With the integration of the transformers backend, we step towards that ideal world.

Here is the [official documentation](https://docs.vllm.ai/en/latest/models/supported_models.html#custom-models)
on how to make your transformers model compatible with vLLM for the integration to kick in.
We followed this and made `modeling_gpt2.py` compatible with the integration! You can follow the
changes in this [transformers pull request](https://github.com/huggingface/transformers/pull/36934).

For a model already in transformers (and compatible with vLLM), this is what we would need to:

```py
llm = LLM(model="new-transformers-model", model_impl="transformers")
```

> [!NOTE]
> It is not a strict necessity to add `model_impl` parameter. vLLM switches to the transformers
> implementation on its own if the model is not natively supported in vLLM.

Or for a custom model from the Hugging Face Hub:

```py
llm = LLM(model="custom-hub-model", model_impl="transformers", trust_remote_code=True)
```

This backend acts as a **bridge**, marrying transformers’ plug-and-play flexibility with vLLM’s
inference prowess. You get the best of both worlds: rapid prototyping with transformers
and optimized deployment with vLLM.

## Case Study: Helium

[Kyutai Team’s Helium](https://huggingface.co/docs/transformers/en/model_doc/helium) is not yet supported by vLLM. You might want to run optimized inference on the model with vLLM, and this is where the transformers backend shines.

Let’s see this in action:

```bash
vllm serve kyutai/helium-1-preview-2b --model-impl transformers
```

Query it with the OpenAI API:

```py
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(model="kyutai/helium-1-preview-2b", prompt="What is AI?")
print("Completion result:", completion)
```

Here, vLLM efficiently processes inputs, leveraging the transformers backend to load
`kyutai/helium-1-preview-2b` seamlessly. Compared to running this natively in transformers,
vLLM delivers lower latency and better resource utilization.

By pairing Transformers’ model ecosystem with vLLM’s inference optimizations, you unlock a workflow
that’s both flexible and scalable. Whether you’re prototyping a new model, deploying a custom
creation, or scaling a multimodal app, this combination accelerates your path from research to production.