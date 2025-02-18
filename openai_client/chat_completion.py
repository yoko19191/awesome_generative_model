from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

_ = load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

response = client.chat.completions.create(
    model = "o3-mini",
    messages=[{"role": "user", "content": "天空为什么是蓝色的？"}]
)

print(response)