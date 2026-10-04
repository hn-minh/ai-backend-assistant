import argparse
import time
import torch
import wandb
from pathlib import Path
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, DatasetDict
import transformers
from transformers import set_seed, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from src.code_suggester.utils.utils import load_config, format_dataset

def main():
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description="Unsloth Configurable SFT Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    wandb.init(
        project=config["project_name"],
        config=config
    )
    
    active_config = wandb.config

    set_seed(active_config["seed"])

    # SUMMARY LOG
    start_time = time.time()
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name} | Max Memory: {max_memory} GB")

    # BASE MODEL
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=active_config["model"]["base_model"],
        max_seq_length=active_config["model"]["max_seq_length"],
        dtype=active_config["model"]["dtype"],
        load_in_4bit=active_config["model"]["load_in_4bit"],
    )

    # LoRA CONFIG
    model = FastLanguageModel.get_peft_model(
        model,
        r=active_config["lora"]["r"],
        target_modules=active_config["lora"]["target_modules"],
        lora_alpha=active_config["lora"]["r"],
        lora_dropout=active_config["lora"]["lora_dropout"],
        bias=active_config["lora"]["bias"],
        use_gradient_checkpointing=active_config["lora"]["use_gradient_checkpointing"],
        random_state=active_config["seed"],
        use_rslora=active_config["lora"]["use_rslora"],
        loftq_config=active_config["lora"]["loftq_config"],
    )

    model.config.use_cache = False
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # DATASET
    raw_dataset = load_dataset("json", data_files=active_config["data"]["dataset_name"])
    
    train_test = raw_dataset["train"].train_test_split(test_size=0.2, seed=active_config["seed"])
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=active_config["seed"])
    
    dataset = DatasetDict({
        "train": train_test["train"],
        "val": val_test["train"],
        "test": val_test["test"]
    })
    
    formatted_dataset = format_dataset(
        dataset, 
        tokenizer, 
        system_prompt=active_config["data"]["system_prompt"]
    )

    OUTPUT_DIR = Path(active_config["training"]["output_dir"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # SFT CONFIG
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        disable_tqdm=True,
        num_train_epochs=active_config["training"]["num_train_epochs"],
        per_device_train_batch_size=active_config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=active_config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=active_config["training"]["gradient_accumulation_steps"],
        learning_rate=float(active_config["training"]["learning_rate"]),
        weight_decay=active_config["training"]["weight_decay"],
        warmup_steps=active_config["training"]["warmup_steps"],
        fp16=active_config["training"]["fp16"],
        bf16=active_config["training"]["bf16"],
        logging_steps=active_config["training"]["logging_steps"],
        eval_strategy=active_config["training"]["evaluation_strategy"],
        eval_steps=active_config["training"]["eval_steps"],
        save_strategy=active_config["training"]["save_strategy"],
        save_steps=active_config["training"]["save_steps"],
        load_best_model_at_end=active_config["training"].get("load_best_model_at_end", False),
        optim=active_config["training"]["optim"],
        report_to="wandb",
        dataset_text_field="text",
        max_seq_length=active_config["model"]["max_seq_length"],
        dataset_num_proc=2
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["val"],
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # TRAIN
    print("Starting training...")
    trainer.train()

    end_time = time.time()
    training_duration_minutes = (end_time - start_time) / 60
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    memory_pct = round((used_memory / max_memory) * 100, 3)

    wandb.run.summary["system/training_duration_minutes"] = training_duration_minutes
    wandb.run.summary["system/peak_gpu_memory_gb"] = used_memory
    wandb.run.summary["system/peak_gpu_memory_percent"] = memory_pct

    print(f"Training completed in {training_duration_minutes:.2f} minutes.")
    print(f"Peak GPU Memory used: {used_memory} GB ({memory_pct}%)")

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    wandb.finish()

if __name__ == "__main__":
    main()
