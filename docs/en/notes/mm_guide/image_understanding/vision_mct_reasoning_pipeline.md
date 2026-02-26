---
title: Vision MCTS Reasoning Pipeline
icon: mdi:image-text
createTime: 2026/01/11 21:59:59
permalink: /en/mm_guide/vision_mct_reasoning_pipeline/
---

## 1. Overview

The **Vision MCTS Reasoning Pipeline** is designed to construct high-quality **Process Supervision Data** for multimodal large models. This pipeline handles two types of data sources: existing Monte Carlo Tree Search (MCTS) trajectory data, or direct generation of new reasoning chains using a VLM.

This pipeline is a core tool for **Grounded-RL** and **SFT Data Construction**, converting complex tree-search processes into a linearized `<think>...</think><answer>...</answer>` format that models can learn from.

We support the following application scenarios:

* **MCTS Data Extraction**: Converting high-value paths (Rollouts) from search trees into linear training data.
* **Hybrid Data Construction**: Automatically falling back to VLM-based CoT generation for samples without search trees.
* **Spatial Reasoning Enhancement**: Supporting the generation of spatial reasoning chains containing explicit coordinates (Bounding Boxes).

The main process of the pipeline includes:

1. **MCTS Tree Parsing**: Parsing the search tree structure in the input data to extract successful reasoning paths.
2. **Visual Reasoning Generation (Fallback)**: Using a VLM to regenerate reasoning chains for samples where the tree structure is missing or parsing fails.
3. **Data Standardization**: Outputting reasoning chain data in a unified format.

---

## 2. Quick Start

### Step 1: Create a Working Directory

```bash
mkdir run_mcts_reasoning
cd run_mcts_reasoning

```

### Step 2: Prepare the Script

Save the code in the "Pipeline Example" section below as `vision_mcts_pipeline.py`.

### Step 3: Download Example Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### Step 4: Run

```bash
python vision_mcts_pipeline.py \
  --model_path "/path/to/Qwen2.5-VL-3B-Instruct" \
  --input_file "data/mcts_trajectories.jsonl" \
  --prompt_type "spatial"

```

---

## 3. Data Flow & Logic

### 1. **Input Data**

Input data typically comes from MCTS search logs or unlabelled image-text pairs:

* **image**: Path to the image.
* **question**: Visual question.
* **tree** (optional): JSON structure of the MCTS search tree, containing node values, visit counts, and actions.

**Input Data Example**:

```json
{
    "image": "./images/puzzle.jpg",
    "question": "What is the next step to solve this?",
    "tree": { "root": { "children": [...], "value": 1.0, "text": "Step 1..." } }
}

```

### 2. **Core Operator Logic**

The pipeline employs an **"Extract First, Fallback to Generate"** hybrid strategy:

#### A. **MCTSTreeRefiner**

This operator is responsible for processing the `tree` field. It traverses the tree structure and filters for the best paths from root to leaf based on node Q-values.

* **Input**: `tree` object.
* **Functionality**: Linearizes tree paths, filtering out low-value or incomplete search branches.
* **Output**: List of extracted reasoning chains (`mcts_chains`).

#### B. **VisualReasoningGenerator**

This operator is the "Generation Engine" of the pipeline. It takes the extraction results from the previous step as input.

* **Mechanism**: Checks `input_existing_chains_key` (i.e., `mcts_chains`).
* If MCTS parsing was successful (chains exist), it reuses them directly without running inference (saving compute).
* If MCTS chains are empty (tree missing or parsing failed), it calls the VLM to generate reasoning chains from scratch based on the `prompt_type`.


* **Prompt Type**: Supports modes like `spatial` (spatial coordinate reasoning), `logical` (logical reasoning), etc.

### 3. **Output Data**

The final output data (`final_reasoning_chains`) will contain high-quality Chain-of-Thought data ready for SFT training.

**Output Example**:

```json
{
    "image": "./images/puzzle.jpg",
    "final_reasoning_chains": [
        "<think>First, locate the red block at [100, 200]. To solve the puzzle, it needs to move right...</think><answer>Move Red Block</answer>"
    ]
}

```

---

## 4. Pipeline Example

Below is the complete `VisionMCTSReasoningPipeline` code implementation.
```python
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

# 引入原子算子
from dataflow.operators.core_text import MCTSTreeRefiner
from dataflow.operators.core_vision import VisualReasoningGenerator

class VisionMCTSReasoningPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        # Storage
        first_entry_file: str,
        cache_path: str = "./cache_mcts",
        file_name_prefix: str = "mcts_reason",
        # Config
        prompt_type: str = "spatial",
        max_samples_per_file: int = 10000,
        # Keys
        input_question_key: str = "question",
        input_image_key: str = "image",
        input_tree_key: str = "tree",
        output_key: str = "final_reasoning_chains",
        # VLLM
        vllm_max_tokens: int = 1024
    ):
        # 1. 存储初始化
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )
        
        # 2. 模型服务
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_max_tokens=vllm_max_tokens
        )
        
        self.keys = {
            "q": input_question_key,
            "img": input_image_key,
            "tree": input_tree_key,
            "mcts_chains": "mcts_extracted_chains", # 中间结果
            "final": output_key
        }

        # ================== Operators ==================
        
        # 算子 1: MCTS Tree -> Chains (提取器)
        # 负责将树结构扁平化为线性链
        self.op_mcts_refine = MCTSTreeRefiner(
            max_chains_per_sample=max_samples_per_file
        )
        
        # 算子 2: VLM -> Chains (生成器/Fallback)
        # 如果 MCTS 提取失败，则使用 VLM 生成；如果成功，则跳过
        self.op_vlm_gen = VisualReasoningGenerator(
            serving=self.serving,
            prompt_type=prompt_type
        )

    def forward(self):
        print(">>> [Pipeline] Step 1: Extracting Chains from MCTS Trees...")
        self.op_mcts_refine.run(
            self.storage.step(),
            input_tree_key=self.keys["tree"],
            output_key=self.keys["mcts_chains"]
        )
        
        print(">>> [Pipeline] Step 2: Generating Chains via VLM (Fallback)...")
        # 注意：input_existing_chains_key 实现了混合/回退逻辑
        self.op_vlm_gen.run(
            self.storage.step(),
            input_question_key=self.keys["q"],
            input_image_key=self.keys["img"],
            input_existing_chains_key=self.keys["mcts_chains"],
            output_key=self.keys["final"]
        )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="dataflow/example/image_to_text_pipeline/mct_reasoning.jsonl")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--prompt_type", default="spatial")
    args = parser.parse_args()
    
    pipe = VisionMCTSReasoningPipeline(
        model_path=args.model_path,
        first_entry_file=args.input_file,
        prompt_type=args.prompt_type
    )
    pipe.forward()

```