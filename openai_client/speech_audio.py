from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

speech_file_path = Path(__file__).parent / "siliconcloud-generated-speech.mp3"

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"), # 从 https://cloud.siliconflow.cn/account/ak 获取
    base_url=os.getenv("SILICONFLOW_BASE_URL")
)

with client.audio.speech.with_streaming_response.create(
  model="FunAudioLLM/CosyVoice2-0.5B", # 支持 fishaudio / GPT-SoVITS / CosyVoice2-0.5B 系列模型
  voice="FunAudioLLM/CosyVoice2-0.5B:claire", # 系统预置音色
  # 用户输入信息
  input="你能用高兴的情感说吗？<|endofprompt|>今天真是太开心了，马上要放假了！I'm so happy, Spring Festival is coming!",
  response_format="mp3" # 支持 mp3, wav, pcm, opus 格式
) as response:
    response.stream_to_file(speech_file_path)