---
title: Image Caption Generation Pipeline (API Version) 
icon: mdi:image-edit 
createTime: 2026/01/24 16:37:37 
permalink: /en/mm_guide/image_caption_api_pipeline/
---

## 1. Overview

The **Image Caption Generation Pipeline (API Version)** is designed to leverage advanced Vision-Language Models (VLM) to automatically generate high-quality, accurate, and informative text descriptions for large-scale image datasets. By calling APIs compatible with the OpenAI format, this pipeline rapidly processes images and generates structured annotation data. It is an ideal choice for building multimodal pre-training datasets, image retrieval systems, and accessibility features.

We support the following application scenarios:

* **Multimodal Dataset Annotation**: Batch generate precise text descriptions for massive image libraries.
* **Image Content Understanding**: Automatically extract key objects, scenes, and textual information from images.
* **Search & Retrieval Optimization**: Enhance image searchability through rich textual descriptions.

---

## 2. Quick Start

### Step 1: Configure API Key

Set the API Key environment variable in your script:

```python
import os
os.environ["DF_API_KEY"] = "your_api_key_here"

```

### Step 2: Prepare the Environment

Create a working directory and initialize:

```bash
mkdir run_caption_pipeline
cd run_caption_pipeline
dataflowmm init

```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir data

```

### Step 4: Configure Core Parameters

Configure the API information in the generated `api_pipelines/image_caption.py` script:

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1", # Replace with your API endpoint
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-5-nano-2025-08-07",
    max_workers=10,
    timeout=1800
)

```

### Step 5: Run the Pipeline

```bash
python api_pipelines/image_caption.py --images_file data/image_caption/sample_data.json

```

---

## 3. Data Flow & Logic

### 1. **Input Data Structure**

The pipeline accepts standard JSON/JSONL formats containing image paths and prompts:

```json
[
    {
        "image": ["./data/image_caption/person.png"],
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

In this workflow, we use `PromptedVQAGenerator` as the core operator. It transforms the VLM into a specialized image captioning engine via a system prompt.

* **System Prompt**: "You are an image caption generator. Your task is to generate a concise and informative caption for the given image content."
* **Concurrency Control**: Supports multi-threaded concurrent requests via the `max_workers` parameter, significantly improving processing efficiency for large datasets.
* **Fault Tolerance**: Built-in timeout and retry mechanisms ensure stability of API calls under high load.

### 3. **Output Data Example**

Once processing is complete, the `caption` field is appended directly to the data object:

```json
[
  {
    "image": ["./data/image_caption/person.png"],
    "conversation": [...],
    "caption": "Promotional poster for Nightmare Alley in grayscale, showing a man in a formal tuxedo with a white bow tie. The cast names run down the left side (Bradley Cooper, Cate Blanchett, Toni Collette, Willem Dafoe, and more), and the gold title Nightmare Alley appears near the bottom left with release text and Regal branding."
  }
]

```

---

## 4. Full Pipeline Code

You can directly use or modify the following Python code to implement your custom image captioning task.

```python
import os
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator

# Set API Key environment variable
os.environ["DF_API_KEY"] = "sk-xxx"

class ImageCaptionPipeline:
    """
    Batch image caption generation with a single command.
    """

    def __init__(
        self,
        first_entry_file: str,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "caption",
        cache_type: str = "json",
    ):
        # ---------- 1. Storage: Manage data reading and checkpoints ----------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # ---------- 2. Serving: Configure API Service ----------
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1", 
            key_name_of_api_key="DF_API_KEY",
            model_name="gpt-5-nano-2025-08-07",
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator: Define Generation Logic ----------
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are an image caption generator. Your task is to generate a concise and informative caption for the given image content."
        )

    def forward(self):
        # Run the pipeline
        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key="image",
            output_answer_key="caption",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image caption generation with DataFlow")
    parser.add_argument("--images_file", default="data/image_caption/sample_data.json")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="caption")
    parser.add_argument("--cache_type", default="json")

    args = parser.parse_args()

    pipe = ImageCaptionPipeline(
        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
    )
    pipe.forward()

```
