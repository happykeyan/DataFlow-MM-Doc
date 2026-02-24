---
title: ContextVQA 多模态问答数据生成流水线 
icon: mdi:image-text 
createTime: 2026/01/24 16:37:37 
permalink: /zh/mm_guide/contextvqa_pipeline/

---
## 1. 概述

**ContextVQA 多模态问答数据生成流水线**旨在从图像出发，自动生成**具备外部知识上下文的视觉问答（Context-based VQA）数据**。该流水线利用视觉语言模型（VLM）生成与图像相关的 Wikipedia 风格文章及问答对，并将其解析为结构化数据。

我们支持以下应用场景：

* **知识型 VQA 数据合成**：构建需要外部知识推理的问答数据集。
* **多模态 RAG 数据构建**：生成用于检索增强生成（RAG）训练的高质量数据。
* **视觉推理训练**：生成问题指向图像、但答案需从文本上下文推理的数据。

流水线的主要流程包括：

1. **数据加载**：读取包含图像路径的数据文件。
2. **上下文与问答生成**：利用本地部署的 VLM 基于图像生成 Wikipedia 风格文章及原始问答对。
3. **数据清洗与结构化**：解析原始文本，提取结构化的 `{context, qas}` 格式。

---

## 2. 快速开始

### 第一步：创建新的 DataFlow 工作文件夹

```bash
mkdir run_dataflow_mm
cd run_dataflow_mm

```

### 第二步：初始化 DataFlow-MM

```bash
dataflow init

```

这时你会看到：

```bash
gpu_pipelines/context_vqa.py  

```

### 第三步：下载示例数据

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### 第四步：配置模型与数据路径

在 `context_vqa.py` 中直接修改类初始化参数（不再通过命令行参数传递）：

```python
# 模型服务配置
self.serving = LocalModelVLMServing_vllm(
    hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    hf_cache_dir="~/.cache/huggingface",
    hf_local_dir="./ckpt",
    vllm_tensor_parallel_size=1,
    vllm_max_tokens=512,
)

# 数据存储配置
self.storage = FileStorage(
    first_entry_file_name="./example_data/image_contextvqa/sample_data.json",
    cache_path="./cache_local",
    file_name_prefix="context_vqa",
    cache_type="json",
)

```

### 第五步：一键运行

```bash
python gpu_pipelines/context_vqa.py

```

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据通过 `FileStorage` 进行管理，支持断点续传。

**输入数据示例 (`sample_data.json`)**：

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

#### A. **FixPromptedVQAGenerator（上下文生成）**

该算子负责调用本地 VLM 模型，根据内置的 Wikipedia 风格 Prompt 模板生成原始文本。

**算子运行**：

```python
self.vqa_generator.run(
    storage=self.storage.step(),
    input_conversation_key="conversation",
    input_image_key=input_image_key,
    output_answer_key=output_answer_key,
)

```

#### B. **WikiQARefiner（结果解析）**

该算子负责将 VLM 生成的非结构化文本清洗并转换为标准格式，分离文章内容（Context）和问答对（QAs）。

**算子运行**：

```python
self.refiner.run(
    storage=self.storage.step(),
    input_key="vqa",          # 输入上一涉的原始文本
    output_key="context_vqa"  # 输出最终结构化数据
)

```

### 3. **输出数据**

最终生成的结构化数据包含 `context`（文章）和 `qas`（问答列表）。

**输出数据示例**：

```json
{
    "id": 1,
    "image": ["./example_data/image_contextvqa/person.png"],
    "context_vqa": {
        "context": "Nightmare Alley is a 2021 American psychological thriller film...",
        "qas": [
            {
                "question": "What genre does this film belong to?",
                "answer": "Psychological thriller"
            }
        ]
    }
}

```

---

## 4. 流水线示例

以下是完整的 `ContextVQAPipeline` 代码实现。

```python
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import PromptedVQAGenerator, WikiQARefiner


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
        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
            hf_cache_dir="~/.cache/huggingface",
            hf_local_dir="./ckpt",
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
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
            output_answer_key=output_answer_key
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