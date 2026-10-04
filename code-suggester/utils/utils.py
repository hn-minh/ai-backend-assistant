import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_dataset(dataset, tokenizer, system_prompt: str):
    def formatting_prompts_func(examples):
        batch_messages = examples["messages"]
        texts = []
        for messages in batch_messages:
            chat_template_messages = [{"role": "system", "content": system_prompt}]
            chat_template_messages.extend(messages)
            
            text = tokenizer.apply_chat_template(
                chat_template_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}
    
    return dataset.map(formatting_prompts_func, batched=True, num_proc=2)


