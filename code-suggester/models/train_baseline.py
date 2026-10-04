import argparse
import time
import torch
import wandb
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import set_seed
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

from src.code_suggester.utils.utils import load_config, format_dataset

def main():
    parser = argparse.ArgumentParser(description="Unsloth Configurable SFT Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    set_seed(config["seed"])

    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config
    )

    start_time = time.time()
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name} | Max Memory: {max_memory} GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["base_model"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=config["model"]["dtype"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
        use_rslora=config["lora"]["use_rslora"],
        loftq_config=config["lora"]["loftq_config"],
    )
    model.config.use_cache = False

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    raw_dataset = load_dataset("json", data_files=config["data"]["dataset_name"])
    
    train_test = raw_dataset["train"].train_test_split(test_size=0.2, seed=config["seed"])
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=config["seed"])
    
    dataset = DatasetDict({
        "train": train_test["train"],
        "val": val_test["train"],
        "test": val_test["test"]
    })
    
    formatted_dataset = format_dataset(
        dataset, 
        tokenizer, 
        system_prompt=config["data"]["system_prompt"]
    )

    training_args = config["training"]
    OUTPUT_DIR = Path(training_args["output_dir"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=training_args["num_train_epochs"],
        per_device_train_batch_size=training_args["per_device_train_batch_size"],
        per_device_eval_batch_size=training_args["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
        learning_rate=float(training_args["learning_rate"]),
        weight_decay=training_args["weight_decay"],
        warmup_steps=training_args["warmup_steps"],
        fp16=training_args["fp16"],
        bf16=training_args["bf16"],
        logging_steps=training_args["logging_steps"],
        eval_strategy=training_args["evaluation_strategy"],
        eval_steps=training_args["eval_steps"],
        save_strategy=training_args["save_strategy"],
        save_steps=training_args["save_steps"],
        load_best_model_at_end=training_args.get("load_best_model_at_end", False),
        optim=training_args["optim"],
        report_to="wandb",
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        dataset_num_proc=2
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["val"],
        args=sft_config,
    )

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
