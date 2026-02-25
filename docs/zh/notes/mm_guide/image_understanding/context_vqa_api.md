---
title: ContextVQA 多模态问答数据生成流水线（API版） 
icon: mdi:image-text 
createTime: 2026/01/24 16:37:37 
permalink: /zh/mm_guide/contextvqa_api_pipeline/
---
## 1. 概述

**ContextVQA 多模态问答数据生成流水线（API版）旨在从图像出发，自动生成具备外部知识上下文的视觉问答（Context-based VQA）数据**。该流水线通过 API 形式的视觉语言模型（VLM）生成 Wikipedia 风格文章及问答对，并将其解析为结构化数据，便于构建知识型 VQA 与多模态 RAG 数据集。

我们支持以下应用场景：

* **知识型 VQA 数据合成**：构建需要外部知识推理的问答数据集。
* **多模态 RAG 数据构建**：生成用于检索增强生成（RAG）训练的高质量数据。
* **视觉推理训练**：生成问题指向图像、但答案需从文本上下文推理的数据。

流水线的主要流程包括：

1. **数据加载**：读取包含图像路径的数据文件。
2. **上下文与问答生成**：利用 VLM API 基于图像生成 Wikipedia 风格文章及原始问答对。
3. **数据清洗与结构化**：解析原始文本，提取结构化的 `{context, qas}` 格式。

---

## 2. 快速开始

### 第一步：配置 API Key

在脚本中设置 API Key 环境变量：

```python
import os
os.environ["DF_API_KEY"] = "sk-xxx"

```

### 第二步：创建新的 DataFlow 工作文件夹

```bash
mkdir run_dataflow
cd run_dataflow

```

### 第三步：初始化 DataFlow-MM

```bash
dataflowmm init

```

这时你会看到：

```bash
api_pipelines/image_contextvqa.py

```

### 第四步：下载示例数据

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### 第五步：配置参数

在 `image_contextvqa.py` 中配置 API 服务和输入数据路径（无需 `argparse`，直接在代码中修改默认路径）：

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1", # 任意兼容openai 格式的api平台
    key_name_of_api_key="DF_API_KEY", # 对应的api key，在第一步中设置
    model_name="gpt-5-nano-2025-08-07",
    image_io=None,
    send_request_stream=False,
    max_workers=10,
    timeout=1800
)

```

```python
self.storage = FileStorage(
    first_entry_file_name="./example_data/image_contextvqa/sample_data.json",
    cache_path="./cache_local",
    file_name_prefix="context_vqa",
    cache_type="json",
)

```

### 第六步：一键运行

```bash
python api_pipelines/image_contextvqa.py

```

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据主要包含以下字段：

* **image**：图像文件路径（本地路径或 URL）。
* **id**（可选）：数据的唯一标识符。
* **conversation**（可选）：对话格式文本，用于补充生成上下文。

数据通过 `FileStorage` 进行管理，支持断点续传。

**输入数据示例**：

```json
[
    {
        "image": ["./example_data/image_contextvqa/person.png"],
        "conversation": [
            {
                "from": "human",
                "value": "Write a Wikipedia article related to this image without directly referring to the image. Then write question answer pairs. The question answer pairs should satisfy the following criteria.\n1: The question should refer to the image.\n2: The question should avoid mentioning the name of the object in the image.\n3: The question should be answered by reasoning over the Wikipedia article.\n4: The question should sound natural and concise.\n5: The answer should be extracted from the Wikipedia article.\n6: The answer should not be any objects in the image.\n7: The answer should be a single word or phrase and list all correct answers separated by commas.\n8: The answer should not contain 'and', 'or', rather you can split them into multiple answers."
            }
        ]
    }
]
```

### 2. **核心算子逻辑**

该流水线通过串联两个核心算子来完成任务：

#### A. **PromptedVQAGenerator（上下文生成）**

该算子负责调用 VLM API，根据 Prompt 模板生成原始文本。

**功能：**

* 基于图像生成一段 Wikipedia 风格的科普文章。
* 基于文章生成问答对。
* **Prompt 约束**：问题指向图像但避免直接提及物体名称；答案来自文章内容且非图像中的物体；答案简练。

**模型服务配置**：

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1",
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-5-nano-2025-08-07",
    max_workers=10,
    timeout=1800
)

```

**算子运行**：

```python
self.vqa_generator.run(
    storage=self.storage.step(),
    input_conversation_key="conversation",
    input_image_key="image",
    output_answer_key="vqa"
)

```

#### B. **WikiQARefiner（结果解析）**

该算子负责将 VLM 生成的非结构化文本清洗并转换为标准格式。

**功能：**

* 清洗 Markdown 格式和多余的空白字符。
* 分离文章内容（Context）和问答对（QAs）。

**算子运行**：

```python
self.refiner.run(
    storage=self.storage.step(),
    input_key="vqa",
    output_key="context_vqa"
)

```

### 3. **输出数据**

最终，流水线生成的输出数据将包含以下内容：

* **image**：原始图像路径。
* **vqa**：VLM 生成的原始文本（中间结果）。
* **context_vqa**：结构化的最终结果，包含 `context`（文章）和 `qas`（问答列表）。

**输出数据示例**：

```json
[
  {
    "image":[
      "./example_data/image_contextvqa/person.png"
    ],
    "conversation":[
      {
        "from":"human",
        "value":"Write a Wikipedia article related to this image..."
      }
    ],
    "context_vqa":{
      "context":"**Wikipedia Article:** *Nightmare Alley* is a 2021 American psychological thriller film directed by Guillermo del Toro...",
      "qas":[
        {
          "question":"What genre does this film belong to?",
          "answer":"Psychological thriller"
        }
      ]
    }
  }
]

```

---

## 4. 流水线示例

以下是完整的 `ContextVQAPipeline` 代码实现。

```python
import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-xxx"

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator
from dataflow.operators.core_vision import WikiQARefiner


class ContextVQAPipeline:
    """
    一行命令即可完成图片批量 ContextVQA Caption 生成。
    """

    def __init__(self, llm_serving: LLMServingABC = None):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_contextvqa/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="context_vqa",
            cache_type="json",
        )

        # ---------- 2. Serving ----------
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1", # Any API platform compatible with OpenAI format
            key_name_of_api_key="DF_API_KEY", # Set the API key in environment variable
            model_name="gpt-5-nano-2025-08-07",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator ----------
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt= "You are a helpful assistant."
        )

        self.refiner = WikiQARefiner()

    # ------------------------------------------------------------------ #
    def forward(self):
        input_image_key = "image"
        output_answer_key = "vqa"
        output_wiki_key = "context_vqa"

        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key=input_image_key,
            output_answer_key=output_answer_key,
        )

        self.refiner.run(
            storage=self.storage.step(),
            input_key=output_answer_key,
            output_key=output_wiki_key
        )

# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    pipe = ContextVQAPipeline()
    pipe.forward()

```