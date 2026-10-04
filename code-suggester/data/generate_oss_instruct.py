import asyncio
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from configs.env import settings

OPENAI_API_KEY = settings.OPENAI_API_KEY
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SEMAPHORE = asyncio.Semaphore(5)

SYSTEM_PROMPT = """You are a practical Python backend developer asking an AI coding assistant for help. 
Your task is to reverse-engineer the prompt you would have typed to get the provided Python code as the answer.

Guidelines for generating the prompt:
1. **Be Natural & Conversational:** Write like a real human developer. DO NOT use robotic, overly prescriptive templates like "Create a function that takes 3 parameters and returns a dictionary."
2. **Focus on Intent (The "What" and "Why"):** Describe the feature, bug fix, or business logic you need to implement, rather than translating the code line-by-line. (e.g., "Write a FastAPI endpoint to process user webhooks and save them to Postgres" is much better than "Create a function that accepts a Request object and a DB session...").
3. **Vary the Detail Level:** Real developers write different types of prompts. Sometimes write short, high-level requests; other times provide a bit more context about the stack (e.g., SQLAlchemy, Pydantic, Redis).
4. **No Spoilers:** DO NOT include the actual code, the exact function name, or internal variable names in your prompt.
5. **Strict Output:** Output ONLY the user instruction text, without any conversational filler, intro, or formatting block."""

async def generate_instruction(func_code: str) -> str:
    async with SEMAPHORE:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a user prompt for this code:\n\n```python\n{func_code}\n```"}
                ],
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return None

async def process_dataset(input_path: Path, output_path: Path):
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
        
    async def process_record(record):
        func_code = record['text']
        instruction = await generate_instruction(func_code)
        
        if instruction:
            return {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": func_code}
                ]
            }
        return None

    tasks = [process_record(r) for r in records]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating Prompts")
    
    valid_dataset = [res for res in results if res is not None]

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n Done! Saved {len(valid_dataset)} Instruction-Response to {output_path}")

if __name__ == "__main__":
    input_file = Path("data/raw/ai_backend_dataset.jsonl")
    output_file = Path("data/processed/magicoder_instruct_dataset.jsonl")
    
    asyncio.run(process_dataset(input_file, output_file))

