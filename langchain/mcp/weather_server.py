from typing import List
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv, find_dotenv
import os
import requests
from datetime import datetime

_ = load_dotenv(find_dotenv())


mcp = FastMCP("Weather")

@mcp.tool()
def get_weather(location: str):
    """example response data
    """
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'appid': weather_api_key,
        'q': location,
        'units': 'metric',
        #'lang': 'zh_cn'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        # extra weather data
        weather_desc = weather_data['weather'][0]['description']
        temp = weather_data['main']['temp']
        feel_temp = weather_data['main']['feels_like']
        mini_temp = weather_data['main']['temp_min']
        max_temp = weather_data['main']['temp_max']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        cloud_cover = weather_data['clouds']['all']
        pressure = weather_data['main']['pressure']
        visibility = weather_data['visibility']
        sunrise = datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M')
        sunset = datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M')
        return {
            "basic": {
                "description": weather_desc, # 天气描述（如 "scattered clouds"）
                "temp": temp, # 当前温度（摄氏度）
                "temp_min": mini_temp, # 当日最低温
                "temp_max": max_temp, # # 当日最高温
                "unit": "celsius" # 单位：摄氏度
            },
            "advanced": {
                "feels_like": feel_temp, # 体感温度
                "humidity": humidity, # 湿度（百分比）
                "wind_speed": wind_speed, # 风速（米/秒）
                "cloud_cover": cloud_cover, # 云量（百分比）
                "pressure": pressure, # 云量（百分比）
                "visibility": visibility, # 能见度（米）
                "sunrise": sunrise, # 日出时间（格式 "HH:MM"）
                "sunset": sunset # # 日落时间（格式 "HH:MM"）
            }
        }
    except requests.exceptions.RequestException as e:
        return f"Cound not retrieve weather information for {location} due to {e}"

if __name__ == "__main__":
    mcp.run()