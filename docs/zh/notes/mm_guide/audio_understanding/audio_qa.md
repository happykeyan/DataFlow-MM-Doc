---
title: 音频问答生成流水线
createTime: 2025/07/15 21:33:01
icon: material-symbols-light:autoplay
permalink: /zh/mm_guide/2gjc47qb/
---


## 音频问答生成流水线

## 1. 概述

**音频问答生成流水线**利用视觉-语言模型（VLM）的音频处理能力，自动为音频内容生成高质量的问答对，适用于音频理解、多模态数据集构建和音频内容分析任务。

我们支持以下应用场景：

- 音频内容自动问答生成
- 多模态音频训练数据集构建
- 音频理解与分析任务

流水线的主要流程包括：

1. **数据加载**：读取音频文件路径和对话格式数据。
2. **音频理解**：利用 VLM 模型分析音频内容。
3. **问答生成**：基于音频内容和对话上下文生成文本回答。

---

## 2. 快速开始

### 第一步：创建新的 DataFlow 工作文件夹
```bash
mkdir run_dataflow_mm
cd run_dataflow_mm
```

### 第二步：初始化 DataFlow-MM
```bash
dataflowmm init
```
这时你会看到：
```bash
run_dataflow_mm/gpu_pipelines/audio_aqa_pipeline.py
```

### 第三步：配置模型路径和参数

在 `audio_aqa_pipeline.py` 中配置模型参数：

```python
# VLM 模型配置
self.serving = LocalModelVLMServing_vllm(
    hf_model_name_or_path="Qwen/Qwen2-Audio-7B-Instruct",  # 修改为你的模型路径
    hf_cache_dir='./dataflow_cache',
    vllm_tensor_parallel_size=2,
    vllm_temperature=0.3,
    vllm_top_p=0.9,
    vllm_gpu_memory_utilization=0.9
)

# 问答生成器配置
self.prompted_generator = PromptedAQAGenerator(
    vlm_serving=self.serving,
    system_prompt="You are a helpful assistant."
)
```

### 第四步：一键运行
```bash
python gpu_pipelines/audio_aqa_pipeline.py
```

此外，你可以根据自己的需求调整参数运行。接下来，我们会介绍在 Pipeline 中使用到的 `PromptedAQAGenerator` 算子以及如何进行参数配置。

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据包括以下字段：

* **audio**：音频文件路径列表，如 `["path/to/audio.wav"]`
* **conversation**：对话格式数据，如 `[{"from": "human", "value": "描述这个音频的内容"}]`

这些输入数据存储在 `jsonl` 文件中，并通过 `FileStorage` 对象进行管理和读取。默认数据路径为：

```python
self.storage = FileStorage(
    first_entry_file_name="../example_data/audio_aqa_pipeline/sample_data.jsonl",
    cache_path="./cache",
    file_name_prefix="audio_aqa_pipeline",
    cache_type="jsonl",
)
```

**输入数据示例**（`sample_data.jsonl`）：

```json
{"audio": ["../example_data/audio_aqa_pipeline/test_1.wav"], "conversation": [{"from": "human", "value": "Transcribe the audio into Chinese." }]}
{"audio": ["../example_data/audio_aqa_pipeline/test_2.wav"], "conversation": [{"from": "human", "value": "Describe the sound in this audio clip." }]}
```

### 2. **音频问答生成（PromptedAQAGenerator）**

流程的核心步骤是使用**提示式音频问答生成器**（`PromptedAQAGenerator`）为每个音频生成问答对。

**功能：**

* 利用模型分析音频内容并生成回答
* 使用预定义的 system prompt 来引导模型生成高质量回答
* 支持自定义对话格式和问答风格
* 可配置模型参数（温度、top_p 等）

**输入**：音频文件路径和对话格式数据  
**输出**：生成的音频问答文本

**模型服务配置**：

```python
self.serving = LocalModelVLMServing_vllm(
    hf_model_name_or_path="Qwen/Qwen2-Audio-7B-Instruct",
    hf_cache_dir='./dataflow_cache',
    vllm_tensor_parallel_size=2,           # 双卡设为 2，单卡可设为 1
    vllm_temperature=0.3,                  # 生成温度，控制随机性
    vllm_top_p=0.9,                        # Top-p 采样参数
    vllm_gpu_memory_utilization=0.9        # GPU 显存利用率
)
```

**问答生成器配置**：

```python
self.prompted_generator = PromptedAQAGenerator(
    vlm_serving=self.serving,
    system_prompt="You are a helpful assistant."  # 系统提示词
)
```

**算子运行**：

```python
self.prompted_generator.run(
    storage=self.storage.step(),
    input_audio_key="audio",                 # 输入音频字段
    input_conversation_key="conversation",   # 输入对话字段
    output_answer_key="answer"               # 输出回答字段
)
```

### 3. **输出数据**

最终，流水线生成的输出数据将包含以下内容：

* **audio**：原始音频路径
* **conversation**：原始对话数据
* **answer**：生成的音频问答文本

## 4. 流水线示例

以下给出示例流水线，展示如何使用 PromptedAQAGenerator 进行音频问答生成。

```python
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    PromptedAQAGenerator,
)
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.prompts.audio import AudioCaptionGeneratorPrompt

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/audio_aqa_pipeline/sample_data.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_aqa_pipeline",
            cache_type="jsonl",
        )

        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2-Audio-7B-Instruct",
            hf_cache_dir="./dataflow_cache",
            vllm_tensor_parallel_size=2,
            vllm_temperature=0.3,
            vllm_top_p=0.9,
            vllm_gpu_memory_utilization=0.9
        )
        self.prompted_generator = PromptedAQAGenerator(
            vlm_serving=self.serving,
            system_prompt="You are a helpful assistant."
        )

    def forward(self):
        self.prompted_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            output_answer_key="answer",
        )

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()
```