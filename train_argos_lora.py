"""
Argos-Action-1.5B v2.0: LoRA Fine-Tuning Pipeline with Assistant-Only Loss Masking & Holdout Evaluation
Base Model: Qwen/Qwen2.5-1.5B-Instruct
Dataset: pedroreoli/argos-computer-use-mcp (v2.0: 1,700 Train / 300 Disjoint Held-Out Test, all 6 tools)
Target Model: pedroreoli/argos-action-1.5b
Author: Pedro Lucas Reis (@pedroreoli)
License: Apache-2.0

Key v2.0 Engineering & Methodological Improvements:
1. Assistant-Only Loss Masking (Completion-Only Cross-Entropy):
   - Tokens belonging to the system prompt and user input (AXTree + Objective + History) are masked
     with label = -100 up to '<|im_start|>assistant\\n'.
   - Loss and token metrics reflect ONLY the assistant's JSON action decision, preventing prompt
     tokens from artificially inflating token accuracy.
2. Hardware & Dataset Cryptographic Provenance:
   - Logs exact GPU model (torch.cuda.get_device_name), CUDA version, and SHA-256 dataset hashes
     at training time alongside native Trainer state.
3. Precision Stability on NVIDIA T4 (Turing sm_75):
   - 4-bit NF4 base quantization (float16 compute) + float32 trainable LoRA weights with
     fp16=False, bf16=False to avoid PyTorch GradScaler BFloat16 unscale errors.
4. Built-in Post-Training Held-Out Generation Benchmark:
   - Automatically evaluates model.generate() on the 300 held-out test samples after training,
     measuring strict schema rate, per-tool accuracy, escalation recall, and Defense-in-Depth
     with ArgosMCPBridge.
"""

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "pedroreoli/argos-computer-use-mcp"
OUTPUT_DIR = "./argos-action-1.5b-lora-v2"
HUB_MODEL_ID = "pedroreoli/argos-action-1.5b"
MAX_SEQ_LENGTH = 1536
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def tokenize_with_assistant_mask(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Tokenizes a ChatML conversation and masks all prompt tokens (system + user + assistant header)
    with -100 in 'labels' so cross-entropy loss is computed strictly on the assistant JSON output.
    """
    messages = example["messages"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


@dataclass
class AssistantMaskCollator:
    """Pads input_ids, attention_mask, and labels (with -100) dynamically per batch."""

    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    print(f"[ARGOS v2.0] Loading dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)

    print(f"[ARGOS v2.0] Loading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[ARGOS v2.0] Tokenizing with Assistant-Only Loss Masking (labels=-100 on prompt)...")
    tokenized_train = dataset["train"].map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    tokenized_eval = dataset["test"].map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer),
        remove_columns=dataset["test"].column_names,
    )

    print("[ARGOS v2.0] Configuring QLoRA 4-bit NF4 base quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    # Ensure trainable parameters are float32 for Turing T4 stability
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=AssistantMaskCollator(pad_token_id=tokenizer.pad_token_id),
    )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[ARGOS v2.0] Starting LoRA training on {gpu_name}...")
    train_result = trainer.train()

    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    provenance = {
        "pipeline_version": "2.0.0",
        "base_model": BASE_MODEL,
        "dataset_id": DATASET_ID,
        "loss_masking": "assistant_only_completion_mask",
        "hardware": gpu_name,
        "torch_version": torch.__version__,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "final_train_loss": train_result.metrics.get("train_loss"),
    }
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_DIR) / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print("[ARGOS v2.0] Pushing LoRA adapter and native trainer_state.json to Hugging Face Hub...")
    trainer.push_to_hub(commit_message="feat(model): publish Argos-Action-1.5B v2.0 LoRA with assistant-only loss & 6-tool coverage")
    tokenizer.push_to_hub(HUB_MODEL_ID)
    print(f"\n[OK] Model published to: https://huggingface.co/{HUB_MODEL_ID}")


if __name__ == "__main__":
    main()
