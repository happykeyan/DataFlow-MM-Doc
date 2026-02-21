---
title: Image VQA 视觉问答数据生成流水线（API版） 
icon: mdi:comment-question 
createTime: 2026/02/10 23:35:00 
permalink: /zh/mm_guide/image_vqa_api_pipeline/
---

## 1. 概述

**Image VQA 视觉问答数据生成流水线（API版）** 专注于从图像内容出发，自动构建高质量的**问答对（Question-Answer Pairs）**。该流水线利用高性能 VLM API，根据图像的视觉特征生成符合人类逻辑的提问与准确回答。这对于训练多模态对话模型、评估模型视觉理解能力以及构建行业特定（如医疗、安防、电商）的 VQA 数据集具有重要价值。

我们支持以下应用场景：

* **指令微调数据合成**：生成多样化的提问方式以增强模型的交互能力。
* **视觉理解评估**：针对图像细节生成判断、描述或推理型问答。
* **自动化标注**：替代人工进行大规模图像问答标注，降低数据生产成本。

---

## 2. 快速开始

### 第一步：配置 API Key

确保您的环境变量中已设置 API 访问权限：

```python
import os
os.environ["DF_API_KEY"] = "sk-your-key-here"

```

### 第二步：初始化环境

```bash
# 创建并进入工作目录
mkdir run_vqa_dataflow
cd run_vqa_dataflow

# 初始化 DataFlow-MM 配置
dataflowmm init

```

### 第三步：下载示例数据

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir data

```

### 第四步：配置运行脚本

在 `api_pipelines/image_vqa.py` 中，您可以自定义 VLM 的模型名称和 API 信息：

```python
self.vlm_serving = APIVLMServing_openai(
    api_url="http://172.96.141.132:3001/v1", # 支持任意 OpenAI 兼容接口
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-5-nano-2025-08-07",
    max_workers=10
)

```

### 第五步：执行流水线

```bash
python api_pipelines/image_vqa.py --images_file data/image_vqa/sample_data.json

```

---

## 3. 数据流与逻辑说明

### 1. **输入数据格式**

输入文件需包含图像路径及触发 VQA 生成的提示引导语：

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

### 2. **核心算子：PromptedVQAGenerator**

此算子是生成问答对的核心引擎：

* **角色定义**：通过 `system_prompt` 设置为 "image question-answer generator"，引导模型输出标准的问答格式。
* **多轮支持**：能够结合 `conversation` 字段中的历史上下文或特定指令来优化问题生成的侧重点。
* **高吞吐处理**：利用 `max_workers` 实现并行调用，适合处理万级以上的图像数据。

### 3. **输出结果示例**

生成的 VQA 结果将以文本形式存储在 `vqa` 字段中，通常包含多个 Q&A 组合：

```json
[
  {
    "image": ["./data/image_vqa/person.png"],
    "vqa": "- Q: What is the title of the movie shown on the poster?\n  A: Nightmare Alley\n\n- Q: What color is the film’s title text?\n  A: Gold"
  }
]

```

---

## 4. 流水线完整代码

```python
import os
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator

# 配置 API 环境
os.environ["DF_API_KEY"] = "sk-xxx"

class ImageVQAPipeline:
    """
    一键式图片批量 VQA 生成流水线
    """

    def __init__(
        self,
        first_entry_file: str,
        cache_path: str = "./cache_local_vqa",
        file_name_prefix: str = "vqa_task",
        cache_type: str = "json",
    ):
        # 1. 初始化存储：支持断点续传与多格式导出
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # 2. 配置 VLM API 服务
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="gpt-5-nano-2025-08-07",
            max_workers=10
        )

        # 3. 初始化 VQA 算子
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a image question-answer generator. Your task is to generate a question-answer pair for the given image content."
        )

    def forward(self):
        # 执行推理任务
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