---
title: Image Caption 图像描述生成流水线（API版） 
icon: mdi:image-edit 
createTime: 2026/01/24 16:37:37
permalink: /zh/mm_guide/image_caption_api_pipeline/
---

## 1. 概述

**Image Caption 图像描述生成流水线（API版）** 旨在利用先进的视觉语言模型（VLM）为大规模图像数据集自动生成高质量、准确且富有信息量的文本描述。该流水线通过调用兼容 OpenAI 格式的 API，能够快速处理图像并生成结构化的标注数据，是构建多模态预训练数据集、图像检索系统和无障碍辅助功能的理想选择。

我们支持以下应用场景：

* **多模态数据集标注**：为大规模图像库批量生成精准的文本描述。
* **图像内容理解**：自动提取图像中的关键对象、场景和文字信息。
* **搜索与检索优化**：通过文本描述增强图像的可搜索性。

---

## 2. 快速开始

### 第一步：配置 API Key

在脚本中设置 API Key 环境变量：

```python
import os
os.environ["DF_API_KEY"] = "your_api_key_here"

```

### 第二步：环境准备

创建工作目录并初始化：

```bash
mkdir run_caption_pipeline
cd run_caption_pipeline
dataflowmm init

```

### 第三步：下载示例数据

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### 第四步：核心参数配置

在生成的 `api_pipelines/image_caption.py` 脚本中配置 API 信息：

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1", # 替换为您的 API 地址
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-5-nano-2025-08-07",
    max_workers=10,
    timeout=1800
)

```

### 第五步：运行流水线

```bash
python api_pipelines/image_caption.py

```

---

## 3. 数据流与逻辑说明

### 1. **输入数据结构**

流水线接收标准的 JSON/JSONL 格式，包含图像路径和提示词：

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

### 2. **核心算子：PromptedVQAGenerator**

在该流程中，我们使用 `PromptedVQAGenerator` 作为核心算子。它通过系统提示词（System Prompt）将 VLM 转化为一个专门的图像描述生成器。

* **系统提示词**："You are a image caption generator. Your task is to generate a concise and informative caption for the given image content."
* **并发控制**：通过 `max_workers` 参数支持多线程并发请求，显著提升大规模数据的处理效率。
* **容错处理**：内置超时与重试机制，确保 API 调用在高负载下的稳定性。

### 3. **输出数据示例**

处理完成后，`caption` 字段将直接添加至数据对象中：

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

## 4. 流水线完整代码

您可以直接使用或修改以下 Python 代码来实现自定义的图像描述任务。

```python
import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-xxx"

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator


class ImageCaptionPipeline:
    """
    一行命令即可完成图片批量 Caption 生成。
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

# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    pipe = ImageCaptionPipeline()
    pipe.forward()

```