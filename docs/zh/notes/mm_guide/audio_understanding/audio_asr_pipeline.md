---
title: 语音识别与清洗流水线
icon: material-symbols:speech-to-text
createTime: 2025/11/17 13:38:14
permalink: /zh/mm_guide/brqv2ysg/
---

# 语音识别与清洗流水线

## 1. 概述
 
**音频语音识别流水线**使用 Whisper 模型和 CTC 强制对齐评估器，自动将音频内容转录为文本，适用于语音识别、音频转录、多模态数据集构建和语音理解任务。
 
我们支持以下应用场景：
 
- 音频内容自动转录
- 语音识别与文本生成
- 多模态音频训练数据集构建
- 语音内容分析与理解
 
流水线的主要流程包括：
 
1. **数据加载**：从 sample_data.jsonl 读取音频文件路径和对话数据
2. **语音转录**：使用 openai/whisper-large-v3 模型进行语音识别
3. **文本处理**：通过文本标准化器处理转录结果
4. **质量评估**：使用 CTC 强制对齐评估器评估转录质量
 
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
run_dataflow_mm/gpu_pipelines/audio_asr_pipeline.py  
```

### 第三步：配置模型路径和参数
 
在 `audio_asr_pipeline.py` 中配置语音识别模型参数：
 
```python
# Whisper 模型服务配置
self.serving = LocalModelVLMServing_vllm(
    hf_model_name_or_path="openai/whisper-large-v3",  # 修改为你的模型路径
    hf_cache_dir='./dataflow_cache',
    vllm_tensor_parallel_size=2,
    vllm_temperature=0.3,
    vllm_top_p=0.9,
    vllm_max_model_len=448,
    vllm_gpu_memory_utilization=0.9
)
 
# Whisper 转录提示配置
self.prompted_generator = PromptedAQAGenerator(
    vlm_serving=self.serving,
    system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="english", task="transcribe", with_timestamps=False)
)
 
# 文本标准化器配置
self.text_normalizer = TextNormalizer(
    language="en",
    remove_puncs=True,
)
 
# CTC 强制对齐评估器配置
self.evaluator = CTCForcedAlignmentSampleEvaluator(
    model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
    device=["cuda:3"],
    num_workers=1,
    sampling_rate=16000,
    language="en",
    micro_batch_size=16,
    chinese_to_pinyin=False,
    romanize=True,
)
```
 
### 第四步：一键运行
```bash
python gpu_pipelines/audio_asr_pipeline.py
```
 
---

## 3. 数据流与流水线逻辑
 
### 1. **输入数据**
 
该流程的输入数据包括以下字段：
 
* **audio**：音频文件路径列表，如 `["path/to/audio.wav"]`
* **conversation**：对话格式数据，如 `[{"from": "human", "value": "Transcribe the audio into Chinese."}]`
 
这些输入数据存储在 `jsonl` 文件中，并通过 `FileStorage` 对象进行管理和读取。默认数据路径为：
 
```python
self.storage = FileStorage(
    first_entry_file_name="../example_data/audio_asr_pipeline/sample_data.jsonl",
    cache_path="./cache",
    file_name_prefix="audio_asr_pipeline",
    cache_type="jsonl",
)
```
 
 ### 2. **语音识别与处理**
 
流程的核心步骤包括：
 
**语音转录**：
 
```python
self.prompted_generator.run(
    storage=self.storage.step(),
    input_audio_key="audio",
    input_conversation_key="conversation",
    output_answer_key="transcript"
)
```
 
**文本标准化**：
 
```python
self.text_normalizer.run(
    storage=self.storage.step(),
    input_text_key="transcript",
)
```
 
**质量评估**：
 
```python
self.evaluator.run(
    storage=self.storage.step(),
    input_audio_key="audio",
    input_conversation_key="transcript",
    output_answer_key="forced_alignment_results",
)
```
 
### 3. **输出数据**
 
最终，流水线生成的输出数据将包含以下内容：
 
* **audio**：原始音频路径
* **conversation**：原始对话数据
* **transcript**：生成的转录文本
* **forced_alignment_results**：强制对齐评估结果
 
---

## 4. 流水线示例
 
以下给出示例流水线，展示如何使用 Whisper 模型和 CTC 评估器进行语音识别。

```python
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    PromptedAQAGenerator,
    TextNormalizer,
    CTCForcedAlignmentFilter,
    CTCForcedAlignmentSampleEvaluator,
)
from dataflow.serving import LocalModelVLMServing_vllm, APIVLMServing_openai
from dataflow.prompts.audio import WhisperTranscriptionPrompt

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/audio_asr_pipeline/sample_data.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_asr_pipeline",
            cache_type="jsonl",
        )

        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="openai/whisper-large-v3",
            hf_cache_dir="./dataflow_cache",
            vllm_tensor_parallel_size=2,
            vllm_temperature=0.3,
            vllm_top_p=0.9,
            vllm_max_model_len=448,
            vllm_gpu_memory_utilization=0.9
        )

        # self.serving = APIVLMServing_openai(
        #     api_url="http://127.0.0.1:8091/v1",
        #     max_workers=3,
        #     model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        # )

        # 对于whisper模型, 使用WhisperTranscriptionPrompt生成prompt
        self.prompted_generator = PromptedAQAGenerator(
            vlm_serving=self.serving,
            system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="english", task="transcribe", with_timestamps=False)
        )

        self.text_normalizer = TextNormalizer(
            language="en",
            remove_puncs=True,
        )

        # self.filter = CTCForcedAlignmentFilter(
        #     model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
        #     device=["cuda:3"],
        #     num_workers=1,
        #     sampling_rate=16000,
        #     language="en",
        #     micro_batch_size=16,
        #     chinese_to_pinyin=False,
        #     threshold=0.1,
        #     threshold_mode="min",
        #     romanize=True,
        # )


        self.evaluator = CTCForcedAlignmentSampleEvaluator(
            model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
            device=["cuda:3"],
            num_workers=1,
            sampling_rate=16000,
            language="en",
            micro_batch_size=16,
            chinese_to_pinyin=False,
            romanize=True,
        )
        
    def forward(self):
        self.prompted_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="transcript"
        )

        self.text_normalizer.run(
            storage=self.storage.step(),
            input_text_key="transcript",
        )

        # self.filter.run(
        #     storage=self.storage.step(),
        #     input_audio_key="audio",
        #     input_conversation_key="transcript",
        # )
        # self.filter.close()

        self.evaluator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="transcript",
            output_answer_key="forced_alignment_results",
        )

        self.evaluator.close()

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()
```

## 5. 合成数据示例
```json
{"audio":["..\/example_data\/audio_asr_pipeline\/test.wav"],"conversation":[{"from":"human","value":""}],"transcript":"and says how do i get to dublin and the answer that comes back is well i would not start from here sonny that is to say much of political philosophy develops theories that take no account of where we actually are and how the theories that people argue about in the journals and in the literature actually could be implemented in the world if it","forced_alignment_results":{"alignment":[{"start":0.063,"end":0.147,"text":"and","score":0.9554548212},{"start":0.273,"end":0.462,"text":"says","score":0.9719064832},{"start":0.609,"end":0.735,"text":"how","score":0.9212982873},{"start":0.798,"end":0.84,"text":"do","score":0.9939799858},{"start":1.029,"end":1.029,"text":"i","score":null},{"start":1.113,"end":1.26,"text":"get","score":0.9985639263},{"start":1.365,"end":1.428,"text":"to","score":0.9945560943},{"start":1.554,"end":1.974,"text":"dublin","score":0.9609149893},{"start":2.856,"end":2.94,"text":"and","score":0.9309501759},{"start":2.982,"end":3.045,"text":"the","score":0.7141059392},{"start":3.192,"end":3.465,"text":"answer","score":0.5938632981},{"start":3.507,"end":3.633,"text":"that","score":0.9633214426},{"start":3.717,"end":4.011,"text":"comes","score":0.9843271526},{"start":4.116,"end":4.389,"text":"back","score":0.9842618417},{"start":4.515,"end":4.662,"text":"is","score":0.9815290374},{"start":5.25,"end":5.376,"text":"well","score":0.047969851},{"start":5.502,"end":5.502,"text":"i","score":null},{"start":5.544,"end":5.67,"text":"would","score":0.8428627272},{"start":5.754,"end":5.817,"text":"not","score":0.123845133},{"start":5.88,"end":6.153,"text":"start","score":0.9789600127},{"start":6.216,"end":6.363,"text":"from","score":0.9000720539},{"start":6.468,"end":6.657,"text":"here","score":0.9283110266},{"start":6.783,"end":7.035,"text":"sonny","score":0.8839239278},{"start":9.807,"end":9.975,"text":"that","score":0.7547208776},{"start":10.038,"end":10.122,"text":"is","score":0.8797863669},{"start":10.185,"end":10.248,"text":"to","score":0.8244834454},{"start":10.353,"end":10.542,"text":"say","score":0.9471999446},{"start":11.025,"end":11.34,"text":"much","score":0.9940719048},{"start":11.634,"end":11.802,"text":"of","score":0.9950778359},{"start":11.991,"end":12.621,"text":"political","score":0.9989232361},{"start":12.81,"end":13.629,"text":"philosophy","score":0.9465096714},{"start":14.217,"end":14.805,"text":"develops","score":0.9432990222},{"start":15.057,"end":15.666,"text":"theories","score":0.9267864129},{"start":17.136,"end":17.304,"text":"that","score":0.8086037475},{"start":17.43,"end":17.682,"text":"take","score":0.9565847912},{"start":17.829,"end":17.913,"text":"no","score":0.956001711},{"start":18.081,"end":18.648,"text":"account","score":0.9546385136},{"start":19.425,"end":19.656,"text":"of","score":0.8420175488},{"start":21.42,"end":21.567,"text":"where","score":0.7551332315},{"start":21.63,"end":21.693,"text":"we","score":0.9166198867},{"start":21.903,"end":22.323,"text":"actually","score":0.9312994611},{"start":22.512,"end":22.701,"text":"are","score":0.9616599245},{"start":22.89,"end":22.974,"text":"and","score":0.4025359219},{"start":23.079,"end":23.31,"text":"how","score":0.9633893459},{"start":23.436,"end":23.499,"text":"the","score":0.7716538814},{"start":23.625,"end":24.045,"text":"theories","score":0.9761697651},{"start":24.15,"end":24.36,"text":"that","score":0.9068021914},{"start":24.486,"end":24.78,"text":"people","score":0.9219708612},{"start":24.948,"end":25.2,"text":"argue","score":0.9620480049},{"start":25.242,"end":25.515,"text":"about","score":0.9651158228},{"start":25.641,"end":25.704,"text":"in","score":0.9931364561},{"start":25.767,"end":25.83,"text":"the","score":0.8166649179},{"start":25.956,"end":26.439,"text":"journals","score":0.9695284503},{"start":26.544,"end":26.607,"text":"and","score":0.9435737354},{"start":26.67,"end":26.712,"text":"in","score":0.778872343},{"start":26.754,"end":26.796,"text":"the","score":0.8787819404},{"start":26.88,"end":27.384,"text":"literature","score":0.928246194},{"start":27.804,"end":28.077,"text":"actually","score":0.9179609355},{"start":28.119,"end":28.266,"text":"could","score":0.8717020111},{"start":28.329,"end":28.392,"text":"be","score":0.9910494216},{"start":28.602,"end":29.169,"text":"implemented","score":0.9847475907},{"start":29.232,"end":29.274,"text":"in","score":0.9814222521},{"start":29.337,"end":29.379,"text":"the","score":0.8807633297},{"start":29.442,"end":29.736,"text":"world","score":0.9051810523},{"start":30.156,"end":30.24,"text":"if","score":0.7553217096},{"start":30.45,"end":30.471,"text":"it","score":0.0156467184}],"error":null}}
```