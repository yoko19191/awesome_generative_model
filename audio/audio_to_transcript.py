from dotenv import load_dotenv, find_dotenv
import requests
import os

_ = load_dotenv(find_dotenv())

url = "https://api.siliconflow.cn/v1/audio/voice/list"

headers = {
    "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}" # 从https://cloud.siliconflow.cn/account/ak获取
}
response = requests.get(url, headers=headers)

print(response.status_code)
print(response.json) # 打印响应内容（如果是JSON格式）


