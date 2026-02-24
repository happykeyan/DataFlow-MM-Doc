---
title: Image VQA Generation Pipeline (API Version) 
icon: mdi:comment-question 
createTime: 2026/02/10 23:35:00 
permalink: /en/mm_guide/image_vqa_api_pipeline/
---

## 1. Overview

**Image VQA Generation Pipeline (API Version)** focuses on automatically constructing high-quality **Question-Answer (QA) Pairs** directly from image content. Leveraging high-performance VLM APIs, this pipeline generates human-like questions and accurate answers based on the visual features of an image. This is highly valuable for training multimodal dialogue models, evaluating visual understanding capabilities, and building industry-specific VQA datasets (e.g., medical, security, e-commerce).

We support the following application scenarios:

* **Instruction Fine-tuning Data Synthesis**: Generate diverse questioning styles to enhance model interaction capabilities.
* **Visual Understanding Evaluation**: Produce judgment, descriptive, or reasoning-based QAs targeting specific image details.
* **Automated Annotation**: Replace manual labor for large-scale image QA annotation, reducing data production costs.

---

## 2. Quick Start

### Step 1: Configure API Key

Ensure your environment variables include the API access rights:

```python
import os
os.environ["DF_API_KEY"] = "sk-your-key-here"

```

### Step 2: Initialize Environment

```bash
# Create and enter the workspace
mkdir run_vqa_dataflow
cd run_vqa_dataflow

# Initialize DataFlow-MM configuration
dataflowmm init

```

### Step 3: Download Example Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### Step 4: Configure Running Script

In `api_pipelines/image_vqa.py`, you can customize the VLM model name and API information:

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1", # Supports any OpenAI-compatible interface
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-5-nano-2025-08-07",
    max_workers=10
)

```

### Step 5: Execute the Pipeline

```bash
python api_pipelines/image_vqa.py 

```

---

## 3. Data Flow and Logic Description

### 1. **Input Data Format**

The input file must contain the image path and a prompt to guide the VQA generation:

```json
[
    {
        "image": ["./example_data/image_vqa/person.png"],
        "conversation": [
            {
                "from": "human",
                "value": "Generate complete QA pairs based on image content and caption."
            }
        ]
    }
]

```

### 2. **Core Operator: PromptedVQAGenerator**

This operator serves as the engine for generating QA pairs:

* **Role Definition**: Through the `system_prompt`, the model is set as an "image question-answer generator," guiding it to output standard QA formats.
* **Multi-turn Support**: It can combine historical context or specific instructions in the `conversation` field to refine the focus of question generation.
* **High Throughput Processing**: Utilizes `max_workers` to implement parallel calls, suitable for processing data at a scale of tens of thousands of images or more.

### 3. **Output Result Example**

The generated VQA results are stored as text in the `vqa` field, typically containing multiple Q&A sets:

```json
[
  {
    "image": ["./example_data/image_vqa/person.png"],
    "vqa": "- Q: What is the title of the movie shown on the poster?\n  A: Nightmare Alley\n\n- Q: What color is the film’s title text?\n  A: Gold"
  }
]

```

---

## 4. Complete Pipeline Code

```python
import os

# Set API Key environment variable
os.environ["DF_API_KEY"] = "sk-xxx"

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator


class ImageVQAPipeline:
    """
    Generate batch VQA for images with a single command.
    """

    def __init__(self, llm_serving: LLMServingABC = None):

        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_vqa/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="qa",
            cache_type="json",
        )

        # ---------- 2. Serving ----------
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1", # Any API platform compatible with OpenAI format
            key_name_of_api_key="DF_API_KEY", # Set the API key in environment variable or line 4
            model_name="gpt-5-nano-2025-08-07",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator ----------
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt= "You are a image question-answer generator. Your task is to generate a question-answer pair for the given image content."
        )

    # ------------------------------------------------------------------ #
    def forward(self):
        input_image_key = "image"
        output_answer_key = "vqa"

        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key=input_image_key,
            output_answer_key=output_answer_key,
        )

# ---------------------------- CLI Entry ------------------------------- #
if __name__ == "__main__":
    pipe = ImageVQAPipeline()
    pipe.forward()

```