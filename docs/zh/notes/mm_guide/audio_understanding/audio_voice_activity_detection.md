---
title: 语音活动检测与切分流水线
createTime: 2025/07/15 21:32:36
icon: material-symbols-light:interpreter-mode
permalink: /zh/mm_guide/dl0jhc6u/
---

## 语音活动检测与切分流水线

## 1. 概述
 
**语音活动检测流水线**使用 Silero VAD 模型和音频时间戳分块生成器，自动检测音频中的语音活动区间并生成对应的时间戳，适用于语音识别预处理、音频分段和多模态数据集构建任务。
 
我们支持以下应用场景：
 
- 音频语音活动检测与分段
- 语音识别预处理
- 多模态音频训练数据集构建
- 音频内容分析与理解
 
流水线的主要流程包括：
 
1. **数据加载**：从 sample_data.jsonl 读取音频文件路径
2. **语音检测**：使用 Silero VAD 模型检测语音活动区间
3. **时间戳生成**：生成语音活动的时间戳并保存结果
 
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
run_dataflow_mm/gpu_pipelines/audio_voice_activity_detection_pipeline.py  
```
 
### 第三步：配置模型路径和参数

在 `audio_voice_activity_detection_pipeline.py` 中配置语音检测模型参数：
 
```python
# Silero VAD 模型配置
self.silero_vad_generator = SileroVADGenerator(
    repo_or_dir="snakers4/silero-vad",
    source="github",
    device=['cuda:0'],                      # GPU 设备
    num_workers=1,
    threshold=0.5,                          # 语音检测阈值
    sampling_rate=16000,                    # 采样率
    max_speech_duration_s=30.0,            # 最大语音时长
    min_silence_duration_s=0.1,            # 最小静音时长
    speech_pad_s=0.03,                     # 语音填充时长
    return_seconds=True,                     # 返回秒级时间戳
)
 
# 时间戳分块生成器配置
self.timestamp_chunk_row_generator = TimestampChunkRowGenerator(
    dst_folder="./cache",
    timestamp_unit="second",
    mode="split",
    max_audio_duration=30.0,               # 最大音频时长
    hop_size_samples=512,                  # 跳步大小
    sampling_rate=16000,                   # 采样率
    num_workers=1,
)
```
 
### 第四步：一键运行
```bash
python gpu_pipelines/audio_voice_activity_detection_pipeline.py
```
 
---

## 3. 数据流与流水线逻辑
 
### 1. **输入数据**
 
该流程的输入数据包括以下字段：
 
* **audio**：音频文件路径列表，如 `["path/to/audio.wav"]`
 
这些输入数据存储在 `jsonl` 文件中，并通过 `FileStorage` 对象进行管理和读取。默认数据路径为：
 
```python
self.storage = FileStorage(
    first_entry_file_name="../example_data/audio_voice_activity_detection_pipeline/sample_data.jsonl",
    cache_path="./cache",
    file_name_prefix="audio_voice_activity_detection_pipeline",
    cache_type="jsonl",
)
```
 
**输入数据示例**（`sample_data.jsonl`）：
 
```jsonl
{"audio": ["../example_data/audio_voice_activity_detection_pipeline/test.wav"], "conversation": [{"from": "human", "value": "" }]}
```
 
### 2. **语音活动检测与分块**
 
流程的核心步骤包括：
 
**语音活动检测**：
 
```python
self.silero_vad_generator.run(
    storage=self.storage.step(),
    input_audio_key='audio',
    output_answer_key='timestamps',
)
```
 
**时间戳分块生成**：
 
```python
self.timestamp_chunk_row_generator.run(
    storage=self.storage.step(),
    input_audio_key="audio",
    input_timestamps_key="timestamps",
)
```

### 3. **输出数据**
 
最终，流水线生成的输出数据将包含以下内容：
 
* **audio**：原始音频路径
* **timestamps**：语音活动检测的时间戳结果
 
---
 
## 4. 流水线示例
 
以下给出示例流水线，展示如何使用 SileroVADGenerator 和 TimestampChunkRowGenerator 进行语音活动检测。

```python
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    SileroVADGenerator,
    TimestampChunkRowGenerator,
)
from dataflow.serving import LocalModelVLMServing_vllm, APIVLMServing_openai

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/audio_voice_activity_detection_pipeline/sample_data.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_voice_activity_detection_pipeline",
            cache_type="jsonl",
        )

        self.silero_vad_generator = SileroVADGenerator(
            repo_or_dir="snakers4/silero-vad",
            source="github",
            device=['cuda:0'],
            num_workers=1,
            threshold=0.5,
            sampling_rate=16000,
            max_speech_duration_s=30.0,
            min_silence_duration_s=0.1,
            speech_pad_s=0.03,
            return_seconds=True,
        )
        self.timestamp_chunk_row_generator = TimestampChunkRowGenerator(
            dst_folder="./cache",
            timestamp_unit="second",
            mode="split",
            max_audio_duration=30.0,
            hop_size_samples=512,
            sampling_rate=16000,
            num_workers=1,
        )
    
    def forward(self):
        self.silero_vad_generator.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            output_answer_key='timestamps',
        )
        self.silero_vad_generator.close()

        self.timestamp_chunk_row_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_timestamps_key="timestamps",
        )
        self.timestamp_chunk_row_generator.close()

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()
```











## 合成数据示例
```json
{"audio":["cache\/test_1.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":1}
{"audio":["cache\/test_2.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":2}
{"audio":["cache\/test_3.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":3}
{"audio":["cache\/test_4.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":4}
{"audio":["cache\/test_5.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":5}
{"audio":["cache\/test_6.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":6}
{"audio":["cache\/test_7.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":7}
{"audio":["cache\/test_8.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":8}
{"audio":["cache\/test_9.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":9}
{"audio":["cache\/test_10.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":10}
{"audio":["cache\/test_11.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":11}
{"audio":["cache\/test_12.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":12}
{"audio":["cache\/test_13.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":13}
{"audio":["cache\/test_14.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":14}
{"audio":["cache\/test_15.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":15}
{"audio":["cache\/test_16.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":16}
{"audio":["cache\/test_17.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":17}
{"audio":["cache\/test_18.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":18}
{"audio":["cache\/test_19.wav"],"original_audio_path":"..\/example_data\/audio_voice_activity_detection_pipeline\/test.wav","sequence_num":19}

```
