"""
Argos-Action-1.5B: LoRA Fine-Tuning Script
Base Model: Qwen/Qwen2.5-1.5B-Instruct
Dataset: pedroreoli/argos-computer-use-mcp
Target Repo: pedroreoli/argos-action-1.5b
Author: Pedro Lucas Reis (@pedroreoli)
"""

import os
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
        
    print(f"[ARGOS] Configurando quantização 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print(f"[ARGOS] Carregando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
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
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
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
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset["train"],
            eval_dataset=formatted_dataset["test"],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=1536,
            tokenizer=tokenizer,
            args=training_args
        )
    
    print("[ARGOS] Iniciando treinamento LoRA...")
    trainer.train()
    
    print("[ARGOS] Treinamento concluído! Enviando adaptadores para o Hugging Face Hub...")
    trainer.push_to_hub(commit_message="feat(model): publish Argos-Action-1.5B LoRA adapter")
    tokenizer.push_to_hub(HUB_MODEL_ID)
    
    print(f"\n[OK] MODELO PUBLICADO COM SUCESSO: https://huggingface.co/{HUB_MODEL_ID}")

if __name__ == "__main__":
    main()
