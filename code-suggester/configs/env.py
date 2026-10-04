import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    
settings = Config()
