---
layout: post
title: "Introducing vLLM Inference Provider in Llama Stack"
author: "Yuan Tang (Red Hat) and Ashwin Bharambe (Meta)"
image: /assets/logos/vllm-logo-only-light.png
---

We are excited to announce that vLLM inference provider is now available in Llama Stack. This article provides an introduction to this integration and a tutorial to help you get started using it.

# What is Llama Stack?

Llama Stack defines and standardizes the set of core building blocks needed to bring generative AI applications to market. These building blocks are presented in the form of interoperable APIs with a broad set of Service Providers providing their implementations.

TODO(ashwin): more background information on Llama Stack 

# vLLM Inference Provider

There are two options: https://docs.vllm.ai/en/latest/serving/serving_with_llamastack.html
TODO(yuan): more details here

We will cover the remote vLLM provider here.

# Tutorial

## Prerequisites

* Linux operating system
* [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) if you'd like to download the model via CLI.
* [Podman](https://podman.io/) or [Docker](https://www.docker.com/) (can be specified via the `DOCKER_BINARY` environment variable when running `llama stack` CLI commands).
* [Kind](https://kind.sigs.k8s.io/) for Kubernetes deployment.
* Python >= 3.10 if you'd like to test the [Llama Stack Python SDK](https://github.com/meta-llama/llama-stack-client-python).


## Get Started via Containers

### Start vLLM Server

Use [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to download models. 

```bash
mkdir /tmp/test-vllm-llama-stack
huggingface-cli login --token <YOUR-HF-TOKEN>
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir /tmp/test-vllm-llama-stack/.cache/huggingface/hub/models/Llama-3.2-1B-Instruct
```

Build images from source. Use CPU for demonstration only
TODO: link to other builds guides other than cpus

```
git clone git@github.com:vllm-project/vllm.git /tmp/test-vllm-llama-stack
cd /tmp/test-vllm-llama-stack/vllm
podman build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
```



Start container:
```bash
podman run -it --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /tmp/test-vllm-llama-stack/.cache/huggingface/hub/models/Llama-3.2-1B-Instruct:/app/model \
   --entrypoint='["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/app/model", "--served-model-name", "meta-llama/Llama-3.2-1B-Instruct", "--port", "8000"]' \
    vllm-cpu-env
```

Testing:
```bash
# Get a list of models
curl http://localhost:8000/v1/models

# Test a prompt
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

### Start Llama Stack Server

Build the image with `llama stack build`:
```
git clone git@github.com:meta-llama/llama-stack.git /tmp/test-vllm-llama-stack/llama-stack
cd /tmp/test-vllm-llama-stack/llama-stack
pip install .

cat > /tmp/test-vllm-llama-stack/vllm-llama-stack-build.yaml << "EOF"
name: vllm
distribution_spec:
  description: Like local, but use vLLM for running LLM inference
  providers:
    inference: remote::vllm
    safety: inline::llama-guard
    agents: inline::meta-reference
    memory: inline::meta-reference
    datasetio: inline::localfs
    scoring: inline::basic
    eval: inline::meta-reference
    post_training: inline::torchtune
    tool_runtime: inline::brave-search
    telemetry: inline::meta-reference
image_type: docker
EOF

export DOCKER_BINARY=podman
LLAMA_STACK_DIR=. PYTHONPATH=. python -m llama_stack.cli.llama stack build --config /tmp/test-vllm-llama-stack/vllm-llama-stack-build.yaml
```

Edit the generated `vllm-run.yaml` to be `/tmp/test-vllm-llama-stack/vllm-llama-stack-run.yaml` with the following change in the `models` field:

```
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: vllm
  provider_model_id: null
```

```
export INFERENCE_ADDR=host.containers.internal
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-1B-Instruct
export LLAMASTACK_PORT=5000
```

Start LlamaStack Server with the image we built via `llama stack run`. 
```
LLAMA_STACK_DIR=. PYTHONPATH=. python -m llama_stack.cli.llama stack run \
--env INFERENCE_MODEL=$INFERENCE_MODEL \
--env VLLM_URL=http://$INFERENCE_ADDR:$INFERENCE_PORT/v1 \
--env VLLM_MAX_TOKENS=8192 \
--env VLLM_API_TOKEN=fake \
--env LLAMASTACK_PORT=$LLAMASTACK_PORT \
/tmp/test-vllm-llama-stack/vllm-llama-stack-run.yaml
```

Alternatively, we can run the following `podman run` command instead:
```
podman run --security-opt label=disable -it --network host -v /tmp/test-vllm-llama-stack/vllm-llama-stack-run.yaml:/app/config.yaml -v /tmp/test-vllm-llama-stack/llama-stack:/app/llama-stack-source \
--env INFERENCE_MODEL=$INFERENCE_MODEL \
--env VLLM_URL=http://$INFERENCE_ADDR:$INFERENCE_PORT/v1 \
--env VLLM_MAX_TOKENS=8192 \
--env VLLM_API_TOKEN=fake \
--env LLAMASTACK_PORT=$LLAMASTACK_PORT \
--entrypoint='["python", "-m", "llama_stack.distribution.server.server", "--yaml-config", "/app/config.yaml"]' \
localhost/distribution-vllm:dev
```

Test inference requests

Via Bash:
```
llama-stack-client --endpoint http://localhost:5000 inference chat-completion --message "hello, what model are you?"
```

Output:
```
ChatCompletionResponse(
    completion_message=CompletionMessage(
        content="Hello! I'm an AI, a conversational AI model. I'm a type of computer program designed to understand and respond to human language. My creators have 
trained me on a vast amount of text data, allowing me to generate human-like responses to a wide range of questions and topics. I'm here to help answer any question you 
may have, so feel free to ask me anything!",
        role='assistant',
        stop_reason='end_of_turn',
        tool_calls=[]
    ),
    logprobs=None
)
```

Via Python:
```python
import os
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

# List available models
models = client.models.list()
print(models)

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)
print(response.completion_message.content)
```

Output:
```
[Model(identifier='meta-llama/Llama-3.2-1B-Instruct', metadata={}, api_model_type='llm', provider_id='vllm', provider_resource_id='meta-llama/Llama-3.2-1B-Instruct', type='model', model_type='llm')]
Here is a haiku about coding:

Columns of code flow
Logic codes the endless night
Tech's silent dawn rise
```

## Deployment on Kubernetes

Create Kind cluster:
```
kind create cluster --image kindest/node:v1.32.0 --name llama-stack-test
```

Start vLLM server (remember to replace `<YOUR-HF-TOKEN>` with your actual token:
```
cat <<EOF |kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-models
spec:
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
type: Opaque
data:
  token: "<YOUR-HF-TOKEN>"
---
apiVersion: v1
kind: Pod
metadata:
  name: vllm-server
  labels:
    app: vllm
spec:
  containers:
  - name: llama-stack
    image: localhost/vllm-cpu-env:latest
    command:
        - bash
        - -c
        - |
          MODEL="meta-llama/Llama-3.2-1B-Instruct"
          MODEL_PATH=/app/model/$(basename $MODEL)
          huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN
          huggingface-cli download $MODEL --local-dir $MODEL_PATH --cache-dir $MODEL_PATH
          python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL --port 8000
    ports:
      - containerPort: 8000
    volumeMounts:
      - name: llama-storage
        mountPath: /app/model
    env:
      - name: HUGGING_FACE_HUB_TOKEN
        valueFrom:
          secretKeyRef:
            name: hf-token-secret
            key: token
  volumes:
  - name: llama-storage
    persistentVolumeClaim:
      claimName: vllm-models
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: NodePort
EOF
```

vLLM server started (this might take a couple of minutes to download the model):
```
$ kubectl logs vllm-server
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Modify previously created `vllm-llama-stack-run.yaml` to `/tmp/test-vllm-llama-stack/vllm-llama-stack-run-k8s.yaml` with the following inference provider:

```
providers:
  inference:
  - provider_id: vllm
    provider_type: remote::vllm
    config:
      url: ${env.VLLM_URL}
      max_tokens: ${env.VLLM_MAX_TOKENS:4096}
      api_token: ${env.VLLM_API_TOKEN:fake}
```

Build an image with LlamaStack run configuration and server source code:

```
cat >/tmp/test-vllm-llama-stack/Containerfile.llama-stack-run-k8s <<EOF
FROM distribution-vllm:dev

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/meta-llama/llama-stack.git /app/llama-stack-source

ADD ./vllm-llama-stack-run-k8s.yaml /app/config.yaml
EOF
podman build -f /tmp/test-vllm-llama-stack/Containerfile.llama-stack-run-k8s -t llama-stack-run-k8s /tmp/test-vllm-llama-stack
```


Start LlamaStack server:
```
cat <<EOF |kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llama-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: llama-stack-pod
  labels:
    app: llama-stack
spec:
  containers:
  - name: llama-stack
    image: localhost/llama-stack-run-k8s:latest
    imagePullPolicy: IfNotPresent
    command: ["python", "-m", "llama_stack.distribution.server.server", "--yaml-config", "/app/config.yaml"]
    ports:
      - containerPort: 5000
    volumeMounts:
      - name: llama-storage
        mountPath: /root/.llama
  volumes:
  - name: llama-storage
    persistentVolumeClaim:
      claimName: llama-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llama-stack-service
spec:
  selector:
    app: llama-stack
  ports:
  - protocol: TCP
    port: 5000
    targetPort: 5000
  type: ClusterIP
EOF
```

LlamaStack server started:
```
$ kubectl logs vllm-server
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     ASGI 'lifespan' protocol appears unsupported.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5000 (Press CTRL+C to quit)
```

Test:
```
kubectl port-forward service/llama-stack-service 5000:5000
llama-stack-client --endpoint http://localhost:5000 inference chat-completion --message "hello, what model are you?"
```
TODO: More interesting prompt to congratulate reaching the end of the article


## Acknowledgments

TBA
