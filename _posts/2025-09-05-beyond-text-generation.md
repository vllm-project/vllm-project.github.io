---
layout: post
title: "Serving Geospatial, Vision, and Beyond: Enabling Any-Modality Models in vLLM"
author: Christian Pinto (IBM Research Europe - Dublin), Michele Gazzetti (IBM Research Europe - Dublin), Michael Johnston (IBM Research Europe - Dublin)
image: /assets/logos/vllm-logo-text-light.png
---
## Introduction

Until recently, generative AI infrastructure has been tightly coupled with autoregressive text generation models that produce output token-by-token, typically in the form of natural language. 
But not all models work this way. 
A growing class of non-autoregressive models generate their outputs in a single inference pass, enabling faster and more efficient generation across a wide range of modalities.

These models are increasingly used in domains beyond text: from image classification and segmentation, to audio synthesis and structured data generation. 
Supporting such any-modality models where inputs and outputs may be images, audio, tabular data, or combinations thereof, requires a fundamental shift in how inference frameworks operate.

We've made that shift in vLLM.

In this article, we introduce a set of enhancements to vLLM that enable non-autoregressive, any-modality model serving. 
While our initial integration focuses on geospatial vision transformers, used for tasks like flood detection, burn scar identification, and land use classification from satellite imagery, the changes are generic and pave the way for serving a wide variety of non text-generating models.

As a concrete example, we've integrated all geospatial models from the [Terratorch](https://github.com/IBM/terratorch) framework (some of which were developed in collaboration with NASA and ESA) into vLLM via a generic backend, making them first-class citizens in the vLLM ecosystem.

In the sections that follow, we describe the technical changes made to vLLM, starting with the requirements and challenges of serving geospatial vision transformer models.

## Integrating geospatial vision transformer models in vLLM

Unlike text models, vision transformers don’t need token decoding i.e. they do not need output tokens to be transformed into text.
Instead, given one input image, a single inference generates the raw model output, and then this is post-processed into the output image.
In addition, sometimes the input image needs to be partitioned and batched into a number of sub-images, or patches.
These patches are then fed to the model for inference, with the resulting output images from each patch being stitched together to form the final output image.

<p align="center">
<picture>
<img src="/assets/figures/beyond-text/models-diff.png" width="80%">
</picture>
</p>

Given these requirements, the obvious choice was to integrate vision transformers in vLLM as pooling models.
In vLLM pooling models allow extracting the raw model output via an identity pooler. 
Identity poolers do not apply any transformation to the data and return it as is - exactly what we need. 
For the input, we pre-process images into tensors that are then fed to vLLM for inference, exploiting the existing multimodal input capabilities of vLLM.

Since we wanted to support multiple geospatial vision transformers out-of-the-box in vLLM we have also added a model implementation backend for TerraTorch models, following the same pattern as the backend for the HuggingFace Transformers library.

Getting this to work was no easy task, though. 
Enabling these model classes required changes to various parts of vLLM such as:

* adding support for attention free models
* improving support for models that do not require a tokenizer
* enabling processing of raw input data as opposed to the default multimodal input embeddings 
* extending the vLLM serving API.

## Meet IO Processor: Flexible Input/Output Handling for Any Model

So far so good! Well, this brings us only halfway towards our goal. 

With the above integration we can indeed serve geospatial vision transformer models -- though only in tensor-to-tensor format. 
Users still have to pre-process their image to a tensor format, before sending these tensors to the vLLM instance.
Similarly, post-processing of the raw tensor output has to happen outside vLLM. 
The impact: there is no endpoint that users can send an image to and get an image back. 

This problem existed because, before our changes, pre-processing of input data and post-processing of the model output was only partially supported in vLLM. 
Specifically, pre-processing of multi-modal input data was only possible via the processors available in the Transformers library. 
However, the transformers processors usually support only standard data types and do not handle more complex data formats such as `geotiff`, which are image files with enriched geospatial metadata. 
Also, on the output processing side vLLM only supported de-tokenization into text or the application of poolers to the model hidden states - no other output processing was possible. 

This is where the new IO Processor plugin framework we introduced comes in.
The IO Processor framework allows developers to customize how model inputs and outputs are pre- and post-processed, all within the same vLLM serving instance. 
Whether your model returns a string, a JSON object, an image tensor, or a custom data structure, an IO Processor can translate it into the desired format before returning it to the client.

<p align="center">
<picture>
<img src="/assets/figures/beyond-text/io-plugins-flow.png" width="70%">
</picture>
</p>

The IO Processor framework unlocks a new level of flexibility for vLLM users.
It means non-text models (e.g., image generators, image to segmentation mask, tabular to classification, etc.) can be served using standard vLLM infrastructure.
Via IO Processors users can plug in custom logic to transform or enrich outputs such as decoding model outputs into images, or formatting responses for downstream systems.
This maintains a unified serving stack, reducing operational complexity and improving maintainability.

### Using vLLM IO Processor plugins

Each IO Processor plugin implements a pre-defined [IO Processor interface](https://github.com/vllm-project/vllm/blob/main/vllm/plugins/io_processors/interface.py) and resides outside of the vLLM source code tree. 
At installation time each plugin registers one or more entrypoints in the `vllm.io_processor_plugins` group. 
This allows vLLM to automatically discover and load plugins at engine initialization time. 
A full plugin example for geospatial vision transformers is available [here](https://github.com/christian-pinto/prithvi_io_processor_plugin).

Using an IO Processor plugin is as easy as installing it in the same python environment with vLLM, and adding the `--io-processor-plugin <plugin_name>` parameter when starting the serving instance. 
Currently, one IO Processor plugin can be loaded for each vLLM instance.

Once the serving instance is started, pre- and post-processing is automatically applied to the model input and output when serving the `pooling` endpoint.
At this stage IO Processors are only available for pooling models, but in the future we expect other endpoints to be integrated too.

## Step-by-Step: Serving the Prithvi Model in vLLM

One example model class that can be served with vLLM using the Terratorch backend is [Prithvi for flood detection](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11).

### Install the python requirements

Install terratorch (>=1.1rc3) and vLLM in your python environment. 
At the time of writing this article the changes required for replicating this example are not yet part of a vLLM release (current latest is v0.10.1.1) and we advise users to install the [latest code](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#install-the-latest-code_1).

Download and install the IO Processor plugin for flood detection with Prithvi.

```bash
git clone git@github.com:christian-pinto/prithvi_io_processor_plugin.git
cd prithvi_io_processor_plugin
pip install .
```

This installs the `prithvi_to_tiff` plugin.

### Start a vLLM serving instance

Start a vLLM serving instance that loads the `prithvi_to_tiff` plugin and the Prithvi model for flood detection.

```bash
vllm serve \
    --model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
    --model-impl terratorch \
    --task embed --trust-remote-code \
    --skip-tokenizer-init --enforce-eager \
    --io-processor-plugin prithvi_to_tiff
```

Once the serving instance is fully up and running, it is ready to serve requests with the selected plugin. 
The below log entries confirm that your vLLM instance is up and running and that it is listening on port `8000`.

```bash
INFO: Starting vLLM API server 0 on http://0.0.0.0:8000
...
...
INFO: Started server process [409128]
INFO: Waiting for application startup.
INFO: Application startup complete.
```

### Send requests to the model
The below python script sends a request to the vLLM `pooling` endpoint with a specific JSON payload where the `model` and `softmax` arguments are pre-defined, while the `data` field is defined by the user and depends on the plugin in use. 
>[!NOTE] 
>Setting the `softmax` field to `False` is required to ensure the plugin receives the raw model output.
In this case we send the input image to vLLM as a URL and we request the response to be a geotiff image in base64 encoding. 
The script decodes the image and writes it to disk as a tiff (geotiff) file.

```python
import base64
import os
import requests

def main():
  image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"
  server_endpoint = "http://localhost:8000/pooling"

  request_payload = {
      "data": {
          "data": image_url,
          "data_format": "url",
          "image_format": "tiff",
          "out_data_format": "b64_json",
      },
      "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
      "softmax": False,
  }

  ret = requests.post(server_endpoint, json=request_payload)

  if ret.status_code == 200:
    response = ret.json()

    decoded_image = base64.b64decode(response["data"]["data"])

    out_path = os.path.join(os.getcwd(), "online_prediction.tiff")

    with open(out_path, "wb") as f:
        f.write(decoded_image)
  else:
    print(f"Response status_code: {ret.status_code}")
    print(f"Response reason:{ret.reason}")


if __name__ == "__main__":
    main()
```

Below is an example of the input and the expected output. 
The input image (left) is a satellite picture of Valencia, Spain during the 2024 flood. 
The output image (right) shows the areas predicted as flooded (in white) by the Prithvi model.

<p align="center">
<picture>
<img src="/assets/figures/beyond-text/prithvi-prediction.png" width="100%">
</picture>
</p>

## What’s Next

This is just the beginning. 
We plan to expand IO Processor plugins across more Terratorch models and modalities and beyond, making installation seamless.
Longer-term, we envision IO Processors powered vision-language systems, structured reasoning agents, and multi-modal pipelines, all served from the same vLLM stack. We're also excited to see how the community uses IO Processors to push the boundaries of what’s possible with vLLM. 

**Contributions, feedback, and ideas are always welcome!**

To get started, check out the IO Processor [documentation](https://docs.vllm.ai/en/latest/design/io_processor_plugins.html) and explore the [examples](https://github.com/vllm-project/vllm/tree/main/examples). 
More information on IBM's Terratorch is available [here](https://github.com/IBM/terratorch).

## Acknowledgement
We would like to thank the members of the vLLM community who helped improving our contribution. In particular, we would like to thank [Maximilien Philippe Marie de Bayser](https://github.com/maxdebayser) (IBM Research Brazil) for his contributions to the IO Processor plugins framework, and [Cyrus Leung](https://github.com/DarkLight1337) (HKUST) for his support in shaping up the overall concept of extending vLLM beyond text generation.
