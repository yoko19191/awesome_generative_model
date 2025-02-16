from dotenv import load_dotenv, find_dotenv
import os
import requests
import json
from datetime import datetime

_ = load_dotenv(find_dotenv())

TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the basic and advanced weather in a given city in english. priority at basic weather information if user don't have any specific requirement",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The english name of the city to get weather information for."
                    }
                },
                "required": ["city"]
            }
        }
    }
]



def get_weather(city: str):
    """example response data
    {'coord': 
        {'lon': 120.1614, 'lat': 30.2937},
        'weather': [
            {'id': 802, 
            'main': 'Clouds', 
            'description': 'scattered clouds', 
            'icon': '03d'}
            ], 
            'base': 'stations',
            'main': {'temp': 13.95,
                    'feels_like': 12.37, 
                    'temp_min': 13.95, 
                    'temp_max': 13.95,
                    'pressure': 1022, 
                    'humidity': 37,
                    'sea_level': 1022,
                    'grnd_level': 1019}, 
            'visibility': 10000, 
            'wind': {'speed': 3.03,
                     'deg': 296,
                     'gust': 5.81}, 
            'clouds': {'all': 40},
            'dt': 1739680212,
            'sys': {'type': 1, 'id': 9651, 'country': 'CN', 'sunrise': 1739659145, 'sunset': 1739699284}, 'timezone': 28800, 'id': 1808926, 'name': 'Hangzhou', 'cod': 200}
    """
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'appid': weather_api_key,
        'q': city,
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
        return f"Cound not retrieve weather information for {city} due to {e}"
    
    
    


    
    
