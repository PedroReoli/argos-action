"""
Argos-Action-1.5B: LoRA Fine-Tuning Pipeline
Base Model: Qwen/Qwen2.5-1.5B-Instruct
Dataset: pedroreoli/argos-computer-use-mcp
Target Model: pedroreoli/argos-action-1.5b
Author: Pedro Lucas Reis (@pedroreoli)
License: Apache-2.0

Engineering Notes:
- Base model is quantized on-the-fly to 4-bit NormalFloat (NF4) via BitsAndBytes.
- LoRA adapters are injected into all 7 attention and MLP projection layers (r=16, alpha=32).
- Precision stability for NVIDIA T4 (Compute Capability 7.5):
  Trainable parameters are cast to float32 with fp16=False, bf16=False in Trainer to
  bypass CUDA GradScaler unscale limitations on Turing architectures.
"""

import os
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
try:
    from transformers import TrainingArguments
except ImportError:
    from trl import SFTConfig as TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configurações
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "pedroreoli/argos-computer-use-mcp"
OUTPUT_DIR = "./argos-action-1.5b-lora"
HUB_MODEL_ID = "pedroreoli/argos-action-1.5b"

def main():
    print(f"[ARGOS] Carregando dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)
    
    print(f"[ARGOS] Carregando tokenizer para {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[ARGOS] Configurando quantização 4-bit (QLoRA NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print(f"[ARGOS] Carregando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 2. Configuração de LoRA (Parameter-Efficient Fine-Tuning)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    # Garantir que todos os tensores float/trainable estejam em float32 (evita conflito com grad scaler em CUDA T4)
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
            
    model.print_trainable_parameters()
    
    # 3. Formatação dos Prompts no padrão ChatML
    def format_prompts(batch):
        formatted_texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        return {"text": formatted_texts}
        
    formatted_dataset = dataset.map(format_prompts, batched=True)
    
    # 4. Hiperparâmetros de Treinamento
    try:
        from trl import SFTConfig
    except ImportError:
        from transformers import TrainingArguments as SFTConfig

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=1536,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID
    )
    
    # 5. Trainer
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset["train"],
            eval_dataset=formatted_dataset["test"],
            peft_config=None,
            processing_class=tokenizer,
            args=training_args
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset["train"],
            eval_dataset=formatted_dataset["test"],
            peft_config=None,
            dataset_text_field="text",
            max_seq_length=1536,
            tokenizer=tokenizer,
            args=training_args
        )
    
    print("[ARGOS] Iniciando treinamento LoRA...")
    train_result = trainer.train()
    
    # Salvar métricas e estado do treinamento localmente
    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    print("[ARGOS] Treinamento concluído! Enviando adaptadores para o Hugging Face Hub...")
    trainer.push_to_hub(commit_message="feat(model): publish Argos-Action-1.5B LoRA policy")
    tokenizer.push_to_hub(HUB_MODEL_ID)
    
    print(f"\n[OK] MODELO PUBLICADO COM SUCESSO: https://huggingface.co/{HUB_MODEL_ID}")

if __name__ == "__main__":
    main()
