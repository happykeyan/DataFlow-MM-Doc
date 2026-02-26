---
title: Visual-Only MCQ Pipeline
createTime: 2026/01/11 22:13:45
icon: mdi:image-text
permalink: /en/mm_guide/image_visual_only_mcq_pipeline/
---
## 1. Overview

The **Visual-Only MCQ Pipeline** is a core component of the CapRL (Caption Reinforcement Learning) framework. Its goal is to generate a set of high-quality Multiple Choice Questions (MCQs) that satisfy **strict visual dependency**: the model must "see" the image to answer correctly; answering based on text alone (guessing or common sense) is not possible.

This pipeline uses a **Generate-Parse-Verify** three-step method, leveraging **Option Rotation** and **Blind Tests** to rigorously filter out hallucinations or overly simple questions. The generated questions serve as a robust reward signal for Reinforcement Learning.

The main process includes:

1. **MCQ Generation**: VLM generates raw QA pairs based on the image.
2. **Structured Parsing**: Using regex logic to parse text into standard question/option structures.
3. **Visual Dependency Verification**:
* **Rotation Test**: Shuffling options multiple times to eliminate positional bias.
* **Dual Filtering**: Requiring high "Visual Accuracy" and low "Text-only Accuracy".



---

## 2. Quick Start

### Step 1: Create Working Directory

```bash
mkdir run_vis_mcq
cd run_vis_mcq

```

### Step 2: Prepare Script

Save the code in the "Pipeline Example" section below as `visual_mcq_pipeline.py`.

### Step 3: Download Example Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### Step 4: Run

```bash
python visual_mcq_pipeline.py \
  --model_path "/path/to/Qwen2.5-VL-3B-Instruct" \
  --input_file "data/captions.jsonl" \
  --rotate_num 4 \
  --pass_vis 1.0 \
  --pass_txt 0.25

```

---

## 3. Data Flow & Logic

### 1. **Input Data**

Input only requires the image path:

* **image**: Path to the image file.

**Input Data Example**:

```json
{
    "image": "./images/sample_01.jpg"
}

```

### 2. **Core Operator Logic**

This pipeline chains three key operators:

#### A. **FixPromptedVQAGenerator (Raw Generation)**

* **Function**: Uses CapRL predefined Prompt templates (`SYS_PROMPT_MCQ` / `USER_PROMPT_MCQ`) to generate 5 MCQs at once.
* **Output**: Unstructured text block containing multiple `#### Question` and options.

#### B. **FunctionalRefiner (Regex Parsing)**

* **Logic Function**: `parse_mcq_text_logic`
* **Function**: Extracts questions, options (A-F), and correct answers from raw text using regex.
* **Output**: Structured MCQ list (`parsed_mcq_list`).

#### C. **VisualDependencyRefiner (Dependency Verification)**

This is the core filter. It performs N inferences (N = `rotate_num`) for each question:

1. **Option Rotation**: Randomly shuffles options (e.g., moving answer from A to C) to prevent the model from cheating by "always picking A".
2. **Visual Pass**: Input Image + Question. Records the model's accuracy.
3. **Textual Pass**: Input Question only (no image). Records the model's blind guessing accuracy.
4. **Filtering Criteria**:
* Keep the question IF AND ONLY IF: `Visual_Acc >= pass_visual_min` **AND** `Textual_Acc <= pass_textual_max`.
* *Example*: If a question can be answered correctly without the image (high text accuracy), it tests common sense rather than vision, so it is **discarded**.



### 3. **Output Data**

The output data (`final_mcqs`) contains only questions that passed rigorous verification. These questions possess high quality and visual relevance.

**Output Data Example**:

```json
{
    "image": "./images/sample_01.jpg",
    "final_mcqs": [
        {
            "question": "What is the color of the car on the far left?\n - A) Red\n - B) Blue...",
            "answer": "A",
            "stats": {
                "visual_acc": 1.0,  # 4/4 correct with image
                "text_acc": 0.0     # 0/4 correct without image
            }
        }
    ]
}

```

---

## 4. Pipeline Example

Below is the complete `VisualOnlyMCQPipeline` code implementation.

```python
import argparse
import re
from typing import List, Dict, Any
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import FixPromptedVQAGenerator, VisualDependencyRefiner
from dataflow.operators.core_text import FunctionalRefiner
from dataflow.prompts.image import ImageCaprlPrompt

# 正则解析逻辑
_Q_BLOCK_SPLIT = re.compile(r"^####\s*\d+\.\s*\*\*(.*?)\*\*\s*$", re.M)
_OPT_LINE_RE = re.compile(r"^\s*-\s*([A-F])\)\s*(.+?)\s*$")
_ANS_LINE_RE = re.compile(r"^\s*\*\*Answer:\*\*\s*([A-F])\)\s*(.+?)\s*$", re.I)

def parse_mcq_text_logic(mcq_text: str, expected: int = 5) -> List[Dict[str, Any]]:
    """将 VLM 生成的原始文本解析为结构化字典列表"""
    if not mcq_text or not isinstance(mcq_text, str): return []
    
    indices = [m.start() for m in _Q_BLOCK_SPLIT.finditer(mcq_text)]
    if not indices: return []
    indices.append(len(mcq_text))
    blocks = [mcq_text[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1)]
    
    parsed = []
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        q_title_m = _Q_BLOCK_SPLIT.search(block)
        if not q_title_m: continue
        
        q_title = q_title_m.group(1).strip()
        options = {}
        ans_letter, ans_text = None, None
        
        for ln in lines:
            m_opt = _OPT_LINE_RE.match(ln)
            if m_opt:
                options[m_opt.group(1)] = m_opt.group(2).strip()
                continue
            m_ans = _ANS_LINE_RE.match(ln)
            if m_ans:
                ans_letter = m_ans.group(1).upper()
                ans_text = m_ans.group(2).strip()
                break
        
        if options and ans_letter and ans_letter in options:
            q_lines = [q_title]
            for lbl in ["A", "B", "C", "D", "E", "F"]:
                if lbl in options:
                    q_lines.append(f"   - {lbl}) {options[lbl]}")
            
            parsed.append({
                "question": "\n".join(q_lines),
                "question_title": q_title,
                "options": options,
                "answer": ans_letter,
                "answer_text": ans_text
            })
            
    if expected > 0:
        parsed = parsed[:expected]
        
    uniq = []
    seen = set()
    for it in parsed:
        key = (it["question_title"], it["answer"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq


class VisualOnlyMCQPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        first_entry_file: str,
        cache_path: str = "./cache_mcq",
        file_name_prefix: str = "vis_mcq",
        # Config
        rotate_num: int = 4,
        pass_visual_min: float = 1.0,
        pass_textual_max: float = 0.25,
        add_none_above: bool = True,
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_mcqs",
        # VLLM
        device: str = "cuda",
        vllm_max_tokens: int = 2048
    ):
        # 1. 初始化存储
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )
        
        # 2. 初始化 VLM 服务
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.1,  # 低温度以保证格式稳定
            vllm_max_tokens=vllm_max_tokens
        )
        
        # Keys 配置
        self.keys = {
            "img": input_image_key,
            "raw_text": "raw_mcq_text",
            "parsed_list": "parsed_mcq_list",
            "final": output_key
        }
        
        # 加载 Prompt 库
        self.prompts_db = ImageCaprlPrompt().build_prompt()

        # ================== 算子初始化 ==================
        
        # 算子 1: 生成原始 MCQ 文本
        self.op_gen_raw = FixPromptedVQAGenerator(
            serving=self.serving,
            system_prompt=self.prompts_db["SYS_PROMPT_MCQ"],
            user_prompt=self.prompts_db["USER_PROMPT_MCQ"]
        )
        
        # 算子 2: 解析文本为结构化数据
        self.op_parse = FunctionalRefiner(func=parse_mcq_text_logic)
        
        # 算子 3: 视觉依赖性验证 (核心过滤)
        # 包含旋转 (Rotation) 和 无图检测 (Text-only check)
        self.op_verify = VisualDependencyRefiner(
            serving=self.serving,
            instruction_template=self.prompts_db["ANSWER_INSTRUCTION"],
            rotate_num=rotate_num,
            pass_visual_min=pass_visual_min,
            pass_textual_max=pass_textual_max,
            add_none_above_visual=add_none_above
        )

    def forward(self):
        print(">>> [Pipeline] Step 1: Generating Raw MCQs (FixPrompted)...")
        self.op_gen_raw.run(
            self.storage.step(),
            input_image_key=self.keys["img"],
            output_answer_key=self.keys["raw_text"]
        )
        
        print(">>> [Pipeline] Step 2: Parsing MCQs...")
        self.op_parse.run(
            self.storage.step(),
            output_key=self.keys["parsed_list"],
            mcq_text=self.keys["raw_text"], 
            expected=5
        )
        
        print(">>> [Pipeline] Step 3: Verifying Visual Dependency (Rotation Check)...")
        self.op_verify.run(
            self.storage.step(),
            input_list_key=self.keys["parsed_list"],
            input_image_key=self.keys["img"],
            output_key=self.keys["final"]
        )
        
        print(f">>> [Pipeline] Done. Results in: {self.keys['final']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--rotate_num", type=int, default=4)
    parser.add_argument("--pass_vis", type=float, default=1.0)
    parser.add_argument("--pass_txt", type=float, default=0.25)
    
    args = parser.parse_args()
    
    pipe = VisualOnlyMCQPipeline(
        model_path=args.model_path,
        first_entry_file=args.input_file,
        rotate_num=args.rotate_num,
        pass_visual_min=args.pass_vis,
        pass_textual_max=args.pass_txt
    )
    pipe.forward()

```