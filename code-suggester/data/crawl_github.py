import os
import requests
import zipfile
import io
from time import sleep
from configs.env import settings

GITHUB_TOKEN = settings.GITHUB_TOKEN
OUTPUT_DIR = "data/raw/git_repos"

TARGET_REPOS = [
    "fastapi/full-stack-fastapi-template",
    "GokuMohandas/Made-With-ML",
    "LAION-AI/Open-Assistant",
    "betaacid/FastAPI-Reference-App",
    "the-momentum/python-ai-kit",
    "Kludex/fastapi-tips",
    "koldakov/futuramaapi",
    "Sanjeev-Thiyagarajan/fastapi-course",
    "benavlabs/FastAPI-boilerplate",
    "benavlabs/fastcrud",
    "JakubPluta/gymhero",
    "douglaschalegre/fastapi-template",
    "AmirHbro/fastapi-blog-sqlalchemy-v2"
]

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_repo_info(full_name):
    print(f"Fetching info cho {full_name}...")
    url = f"https://api.github.com/repos/{full_name}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"API Error: {response.json().get('message')}")
        return None
        
    data = response.json()
    return {
        "name": data.get("name"),
        "full_name": data.get("full_name"),
        "default_branch": data.get("default_branch", "main"),
        "stars": data.get("stargazers_count", 0)
    }

def download_and_extract_repo(repo_info, output_dir):
    full_name = repo_info['full_name']
    branch = repo_info['default_branch']
    
    zip_url = f"https://api.github.com/repos/{full_name}/zipball/{branch}"
    print(f"Downloading {full_name} ({repo_info['stars']} stars, branch: {branch})...")
    
    response = requests.get(zip_url, headers=HEADERS)
    if response.status_code != 200:
        print(f"  -> Cannot download: {full_name}")
        return

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            py_files = [f for f in z.namelist() if f.endswith('.py')]
            
            repo_folder_name = full_name.replace("/", "_")
            repo_out_path = os.path.join(output_dir, repo_folder_name)
            os.makedirs(repo_out_path, exist_ok=True)
            
            for file_path in py_files:
                if "test" in file_path.lower():
                    continue
                    
                filename = os.path.basename(file_path)
                if not filename:
                    continue
                    
                source = z.read(file_path)
                with open(os.path.join(repo_out_path, filename), "wb") as f:
                    f.write(source)
                    
    except zipfile.BadZipFile:
        print(f"Zip file error: {full_name}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for repo_name in TARGET_REPOS:
        repo_info = get_repo_info(repo_name)
        if repo_info:
            download_and_extract_repo(repo_info, OUTPUT_DIR)
        sleep(2)

    print("\nDone")
