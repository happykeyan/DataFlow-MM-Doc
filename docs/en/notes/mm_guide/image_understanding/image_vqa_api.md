---
title: Image VQA Generation Pipeline (API Version) 
icon: mdi:comment-question 
createTime: 2026/02/10 23:35:00 
permalink: /en/mm_guide/image_vqa_api_pipeline/
---

## 1. Overview

The **Image VQA Generation Pipeline (API Version)** focuses on automatically constructing high-quality **Question-Answer Pairs** directly from image content. By leveraging high-performance VLM APIs, the pipeline generates questions and accurate answers that align with human logic based on visual features. This is highly valuable for training multimodal dialogue models, evaluating visual understanding capabilities, and building domain-specific VQA datasets (e.g., medical, security, e-commerce).

We support the following application scenarios:

* **Instruction Tuning Data Synthesis**: Generate diverse questioning styles to enhance model interaction capabilities.
* **Visual Understanding Evaluation**: Create judgment, descriptive, or reasoning-based Q&A focused on image details.
* **Automated Annotation**: Replace manual labor for large-scale image Q&A labeling, reducing data production costs.

---

## 2. Quick Start

### Step 1: Configure API Key

Ensure your environment variables are set with API access permissions:

```python
import os
os.environ["DF_API_KEY"] = "sk-your-key-here"

```

### Step 2: Initialize Environment

```bash
# Create and enter the working directory
mkdir run_vqa_dataflow
cd run_vqa_dataflow

# Initialize DataFlow-MM configuration
dataflowmm init

```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir data

```

### Step 4: Configure the Script

In the generated `api_pipelines/image_vqa.py`, you can customize the VLM model name and API information:

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
python api_pipelines/image_vqa.py --images_file data/image_vqa/sample_data.json

```

---

## 3. Data Flow & Logic

### 1. **Input Data Format**

The input file must contain the image path and a prompt to trigger VQA generation:

```json
[
    {
        "image": ["./data/image_vqa/person.png"],
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

This operator is the core engine for generating Q&A pairs:

* **Role Definition**: Through the `system_prompt` set as "image question-answer generator", the model is guided to output standard Q&A formats.
* **Multi-turn Support**: Capable of combining historical context or specific instructions in the `conversation` field to optimize the focus of generated questions.
* **High-Throughput Processing**: Utilizes `max_workers` for parallel calls, suitable for processing image datasets at scales of  entries.

### 3. **Output Example**

Generated VQA results are stored as text in the `vqa` field, typically containing multiple Q&A sets:

```json
[
  {
    "image": ["./data/image_vqa/person.png"],
    "vqa": "- Q: What is the title of the movie shown on the poster?\n  A: Nightmare Alley\n\n- Q: What color is the film’s title text?\n  A: Gold"
  }
]

```

---

## 4. Full Pipeline Code

```python
import os
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator

# Configure API Environment
os.environ["DF_API_KEY"] = "sk-xxx"

class ImageVQAPipeline:
    """
    One-click batch image VQA generation pipeline
    """

    def __init__(
        self,
        first_entry_file: str,
        cache_path: str = "./cache_local_vqa",
        file_name_prefix: str = "vqa_task",
        cache_type: str = "json",
    ):
        # 1. Initialize Storage: Supports checkpoints and multi-format export
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # 2. Configure VLM API Service
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="gpt-5-nano-2025-08-07",
            max_workers=10
        )

        # 3. Initialize VQA Operator
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are an image question-answer generator. Your task is to generate a question-answer pair for the given image content."
        )

    def forward(self):
        # Execute inference task
        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key="image",
            output_answer_key="vqa",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VQA generation")
    parser.add_argument("--images_file", default="data/image_vqa/sample_data.json")
    args = parser.parse_args()

    pipe = ImageVQAPipeline(first_entry_file=args.images_file)
    pipe.forward()

```