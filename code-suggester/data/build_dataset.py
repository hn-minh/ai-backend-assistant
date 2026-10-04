import ast
import json
import re
import random
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

NUM_PERMUTATIONS = 128
LSH_THRESHOLD = 0.85
MAX_RECORDS = 2000
MIN_CHARS = 50
MAX_CHARS = 3000

def get_tokens(text):
    return set(re.findall(r'\b\w+\b', text.lower()))

def extract_functions_from_code(source_code):
    extracted = []
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if len(node.body) > 3:
                    func_code = ast.unparse(node)
                    if MIN_CHARS <= len(func_code) <= MAX_CHARS and func_code.isascii():
                        extracted.append(func_code)
    except SyntaxError:
        pass
    return extracted

def process_and_dedup(file_paths, output_jsonl):
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERMUTATIONS)
    dataset = []
    
    for path in tqdm(file_paths, desc="Processing files"):
        try:
            content = Path(path).read_text(encoding='utf-8')
        except Exception:
            continue

        for func_code in extract_functions_from_code(content):
            tokens = get_tokens(func_code)
            if not tokens:
                continue
                
            m = MinHash(num_perm=NUM_PERMUTATIONS)
            for token in tokens:
                m.update(token.encode('utf-8'))
            
            if not lsh.query(m):
                lsh.insert(str(len(dataset)), m)
                dataset.append(func_code)

    if len(dataset) > MAX_RECORDS:
        dataset = random.sample(dataset, MAX_RECORDS)

    with open(output_jsonl, 'w', encoding='utf-8') as out_file:
        for func_code in dataset:
            out_file.write(json.dumps({"text": func_code}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    source_dir = Path("data/raw/git_repos") 
    all_py_files = list(source_dir.rglob("*.py"))
    
    output_file = Path("data/raw/ai_backend_dataset.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    process_and_dedup(all_py_files, output_file)
