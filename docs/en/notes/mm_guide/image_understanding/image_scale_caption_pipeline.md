---
title: ScaleCap High-Density Captioning Pipeline
createTime: 2026/01/11 22:08:57
icon: mdi:image-text
permalink: /en/mm_guide/image_scale_caption_pipeline/
---
## 1. Overview

The **ScaleCap High-Density Captioning Pipeline** implements an advanced **"Generate-Verify-Expand-Fuse"** paradigm for image captioning. This pipeline is designed to generate **extremely high information density** captions with **minimal hallucinations**, making it ideal for scenarios requiring deep understanding of image details.

Based on the paper *ScaleCap: Inference-Time Scalable Image Captioning via Dual-Modality Debiasing*, this method progressively mines object and position details through multi-turn dialogue and visual self-verification (Visual Grounding), filtering out hallucinations along the way.

We support the following application scenarios:

* **High-Quality Multimodal Dataset Construction**: Generating training data that is more detailed and accurate than standard captions.
* **Fine-Grained Image Retrieval**: Providing index text rich in detail.
* **Accessibility/Blind Assistance**: Generating "What You See Is What You Get" (WYSIWYG) detailed narrations.

The main process of the pipeline includes:

1. **Initial Caption Generation**: VLM generates a baseline description.
2. **Visual Debiasing**: Splitting the description into sentences and verifying each sentence against visual evidence (Visual Grounding).
3. **Detail Expansion**: Generating follow-up questions about object attributes and positions based on verified "Golden Sentences".
4. **Answering & Re-verification**: VLM answers the questions and performs another round of visual grounding to filter incorrect details.
5. **Final Fusion**: Merging all verified information into a coherent, long description.

---

## 2. Quick Start

### Step 1: Create a Working Directory

```bash
mkdir run_scalecap
cd run_scalecap

```

### Step 2: Prepare the Script

Save the code in the "Pipeline Example" section below as `scalecap_pipeline.py`.

### Step 3: Download Example Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir example_data

```

### Step 4: Run

```bash
python scalecap_pipeline.py \
  --model_path "/path/to/Qwen2.5-VL-3B-Instruct" \
  --input_jsonl "data/images.jsonl" \
  --output_key "final_caption"

```

---

## 3. Data Flow & Logic

### 1. **Input Data**

The input data requires only the image path:

* **image**: Path to the image file.

**Input Data Example**:

```json
{
    "image": "./images/complex_scene.jpg"
}

```

### 2. **Core Operator Logic**

This pipeline is a complex orchestration of multiple atomic operators:

#### A. **Initial Generation (PromptedVQAGenerator)**

* **Function**: Generates a preliminary description (`init_caption`) of the image using a basic prompt.

#### B. **Visual Debiasing (VisualGroundingRefiner)**

* **Function**: The core anti-hallucination mechanism of ScaleCap.
* **Logic**:
1. Uses `split_sentences` to break the draft into single sentences.
2. Asks the VLM: "Given the image, is the description '{text}' directly supported by visual evidence?".
3. Keeps only sentences where the answer is "Yes", forming **"Golden Sentences"**.



#### C. **Question Generation & Parsing (PromptTemplatedQAGenerator)**

* **Function**: Generates targeted follow-up questions based on Golden Sentences using LLM capabilities.
* **Logic**: The model generates text like "Describe more details about the [Object]", which is then automatically expanded into **Object Detail** and **Positional Relation** questions via `parse_questions_logic`.

#### D. **Batch Answering & Refiltering (BatchVQAGenerator & Refiner)**

* **Function**: Mining deep image information.
* **Logic**:
1. Uses `BatchVQAGenerator` to have the VLM answer all generated questions in a batch.
2. Uses `VisualGroundingRefiner` again to check if these new details are accurate.
3. Retains reliable details (`final_details`).



#### E. **Final Fusion (PromptTemplatedQAGenerator)**

* **Function**: Rewrites the "Golden Sentences" and "Verified Details" into a fluent text.
* **Output**: `final_caption`.

### 3. **Output Data**

The output data records the entire pipeline process, facilitating debugging and analysis:

* **init_caption**: Raw generated draft.
* **golden_sentences**: List of sentences that passed the first check.
* **q_list**: List of generated follow-up questions.
* **final_details**: Detailed answers that passed the second check.
* **final_caption**: The final high-density description.

**Output Data Example**:

```json
{
    "image": "./images/complex_scene.jpg",
    "init_caption": "A dog sitting on a bench.",
    "golden_sentences": ["A dog is sitting on a wooden bench."],
    "q_list": ["Describe more details about the dog.", "Describe position of the bench."],
    "final_details": ["The dog is a Golden Retriever with a red collar.", "The bench is located in a park."],
    "final_caption": "A Golden Retriever with a red collar is sitting on a wooden bench located in a park..."
}

```

---

## 4. Pipeline Example

Below is the complete `ImageScaleCaptionPipeline` code implementation.

```python
import re
import argparse
from typing import Callable, Any, List

from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.prompts.prompt_template import NamedPlaceholderPromptTemplate
from dataflow.prompts.image import ImageScaleCaptionPrompt
from dataflow.operators.core_vision import PromptedVQAGenerator, BatchVQAGenerator, VisualGroundingRefiner
from dataflow.operators.core_text import PromptTemplatedQAGenerator, FunctionalRefiner

class ImageScaleCaptionPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        device: str = "cuda",
        # Storage params
        first_entry_file: str = "images.jsonl",
        cache_path: str = "./cache_scalecap",
        file_name_prefix: str = "scalecap",
        cache_type: str = "jsonl",
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_caption",
        # VLLM Config
        vllm_tensor_parallel_size: int = 1,
        vllm_temperature: float = 0.7,
        vllm_top_p: float = 0.9,
        vllm_max_tokens: int = 512,
    ):
        # 1. Storage
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # 2. Serving
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature,
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
        )

        # 3. Prompts
        self.prompts_db = ImageScaleCaptionPrompt().build_prompt()

        # 4. Keys
        self.input_image_key = input_image_key
        self.output_key = output_key

        # ================== Operator Initialization ==================

        # --- Step A: Generate Init Caption ---
        self.refine_const_prompt = FunctionalRefiner(func=lambda: self.prompts_db["VLM_PROMPT_1"])
        self.gen_init_caption = PromptedVQAGenerator(
            serving=self.serving,
            system_prompt="You are a helpful assistant."
        )

        # --- Step B: Refine Golden Sentences ---
        self.refine_split = FunctionalRefiner(func=split_sentences)
        # 视觉自检 (保留 Yes 的句子)
        self.refine_golden = VisualGroundingRefiner(
            serving=self.serving,
            prompt_template="Given the image, is the description '{text}' directly supported by visual evidence? Answer strictly yes or no."
        )

        # --- Step C: Generate Questions ---
        self.refine_join = FunctionalRefiner(func=join_list)
        tpl_q = NamedPlaceholderPromptTemplate(
            template=self.prompts_db["LLM_PROMPT_1"], 
            join_list_with="\n"
        )
        self.gen_questions_text = PromptTemplatedQAGenerator(
            serving=self.serving,
            prompt_template=tpl_q
        )
        self.refine_parse_qs = FunctionalRefiner(func=parse_questions_logic)

        # --- Step D: Generate Answers ---
        self.gen_answers = BatchVQAGenerator(serving=self.serving)
        self.refine_answers = VisualGroundingRefiner(
            serving=self.serving,
            prompt_template="Given the image, is the statement '{text}' grounded in the image and not generic? Answer strictly yes or no."
        )

        # --- Step E: Integrate Final Caption ---
        tpl_final = NamedPlaceholderPromptTemplate(
            template=self.prompts_db["LLM_PROMPT_4"], 
            join_list_with="\n"
        )
        self.gen_final_caption = PromptTemplatedQAGenerator(
            serving=self.serving,
            prompt_template=tpl_final
        )

    def forward(self):
        print(">>> [Pipeline] Step 0: Preparing Prompts...")
        self.refine_const_prompt.run(
            self.storage.step(), 
            output_key="init_prompt"
        )

        print(">>> [Pipeline] Step 1: Generating Initial Caption...")
        self.gen_init_caption.run(
            self.storage.step(),
            input_prompt_key="init_prompt",
            input_image_key=self.input_image_key,
            output_answer_key="init_caption"
        )

        print(">>> [Pipeline] Step 2: Refining Golden Sentences...")
        self.refine_split.run(
            self.storage.step(), 
            output_key="sentences", 
            text="init_caption"
        )
        self.refine_golden.run(
            self.storage.step(), 
            input_list_key="sentences", 
            input_image_key=self.input_image_key, 
            output_key="golden_sentences"
        )

        print(">>> [Pipeline] Step 3: Generating Details Questions...")
        self.refine_join.run(
            self.storage.step(), 
            output_key="golden_str", 
            data="golden_sentences"
        )
        self.gen_questions_text.run(
            self.storage.step(), 
            output_answer_key="raw_q_text", 
            sentence="golden_str"
        )
        self.refine_parse_qs.run(
            self.storage.step(), 
            output_key="q_list", 
            text="raw_q_text"
        )

        print(">>> [Pipeline] Step 4: Generating & Filtering Answers...")
        self.gen_answers.run(
            self.storage.step(), 
            input_prompts_key="q_list", 
            input_image_key=self.input_image_key, 
            output_key="raw_answers"
        )
        self.refine_answers.run(
            self.storage.step(), 
            input_list_key="raw_answers", 
            input_image_key=self.input_image_key, 
            output_key="final_details"
        )

        print(">>> [Pipeline] Step 5: Integrating Final Caption...")
        self.refine_join.run(
            self.storage.step(), 
            output_key="details_str", 
            data="final_details"
        )
        self.gen_final_caption.run(
            self.storage.step(),
            output_answer_key=self.output_key,
            context="golden_str",
            object_info="details_str",
            position_info="details_str"
        )

        print(f">>> [Pipeline] All Done. Result saved to: {self.storage.cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScaleCap Dense Captioning Pipeline")
    
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--input_jsonl", default="./dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl")
    parser.add_argument("--cache_path", default="./cache_scalecap_results")
    parser.add_argument("--file_name_prefix", default="scalecap")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--output_key", default="final_caption")

    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()

    pipe = ImageScaleCaptionPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,
        first_entry_file=args.input_jsonl,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        input_image_key=args.input_image_key,
        output_key=args.output_key,
        vllm_tensor_parallel_size=args.tp,
        vllm_max_tokens=args.max_tokens
    )
    
    pipe.forward()

```
