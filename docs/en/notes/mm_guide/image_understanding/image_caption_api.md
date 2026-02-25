---
title: Image Caption Generation Pipeline (API Version) 
icon: mdi:image-edit 
createTime: 2026/01/24 16:37:37 
permalink: /en/mm_guide/image_caption_api_pipeline/

---
## 1. Overview

**Image Caption Generation Pipeline (API Version)** is designed to leverage advanced Vision-Language Models (VLM) to automatically generate high-quality, accurate, and informative textual descriptions for large-scale image datasets. By calling APIs compatible with the OpenAI format, this pipeline can quickly process images and generate structured annotation data. It is an ideal choice for building multimodal pre-training datasets, image retrieval systems, and accessibility features.

We support the following application scenarios:

* **Multimodal Dataset Annotation**: Batch generate precise text descriptions for large-scale image libraries.
* **Image Content Understanding**: Automatically extract key objects, scenes, and text information from images.
* **Search and Retrieval Optimization**: Enhance image searchability through textual descriptions.

---

## 2. Quick Start

### Step 1: Configure API Key

Set the API Key environment variable in your script:

```python
import os
os.environ["DF_API_KEY"] = "your_api_key_here"

```

### Step 2: Environment Preparation

Create a work directory and initialize:

```bash
mkdir run_caption_pipeline
cd run_caption_pipeline
dataflowmm init

```

### Step 3: Download Example Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### Step 4: Core Parameter Configuration

Configure the API information in the generated `api_pipelines/image_caption.py` script:

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1", # Replace with your API address
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-5-nano-2025-08-07",
    max_workers=10,
    timeout=1800
)

```

### Step 5: Run the Pipeline

```bash
python api_pipelines/image_caption.py

```

---

## 3. Data Flow and Logic Description

### 1. **Input Data Structure**

The pipeline receives standard JSON/JSONL formats containing image paths and prompts:

```json
[
    {
        "image": ["./example_data/image_caption/person.png"],
        "conversation": [
            {
                "from": "human",
                "value": "Generate detailed captions based on image content."
            }
        ]
    }
]

```

### 2. **Core Operator: PromptedVQAGenerator**

In this process, we use `PromptedVQAGenerator` as the core operator. It transforms the VLM into a specialized image caption generator via a System Prompt.

* **System Prompt**: "You are a image caption generator. Your task is to generate a concise and informative caption for the given image content."
* **Concurrency Control**: Supports multi-threaded concurrent requests via the `max_workers` parameter, significantly improving processing efficiency for large-scale data.
* **Error Handling**: Built-in timeout and retry mechanisms ensure API call stability under high loads.

### 3. **Output Data Example**

After processing, the `caption` field is added directly to the data object:

```json
[
  {
    "image": ["./example_data/image_caption/person.png"],
    "conversation": [...],
    "caption": "Promotional poster for Nightmare Alley in grayscale, showing a man in a formal tuxedo with a white bow tie. The cast names run down the left side (Bradley Cooper, Cate Blanchett, Toni Collette, Willem Dafoe, and more), and the gold title Nightmare Alley appears near the bottom left with release text and Regal branding."
  }
]

```

---

## 4. Complete Pipeline Code

You can directly use or modify the following Python code to implement custom image captioning tasks.

```python
import os

# Set API Key environment variable
os.environ["DF_API_KEY"] = "sk-xxx"

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator


class ImageCaptionPipeline:
    """
    Complete batch image caption generation with a single command.
    """

    def __init__(self, llm_serving: LLMServingABC = None):

        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_caption/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="caption",
            cache_type="json",
        )

        # ---------- 2. Serving ----------
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1", # Any API platform compatible with OpenAI format
            key_name_of_api_key="DF_API_KEY", # Set the API key for the corresponding platform in the environment variable or line 4
            model_name="gpt-5-nano-2025-08-07",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator ----------
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt= "You are a image caption generator. Your task is to generate a concise and informative caption for the given image content."
        )

    # ------------------------------------------------------------------ #
    def forward(self):
        input_image_key = "image"
        output_answer_key = "caption"

        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key=input_image_key,
            output_answer_key=output_answer_key,
        )

# ---------------------------- CLI Entry ------------------------------- #
if __name__ == "__main__":
    pipe = ImageCaptionPipeline()
    pipe.forward()

```