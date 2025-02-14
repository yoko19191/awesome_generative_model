from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os 
import requests
from datetime import datetime
import json
_ = load_dotenv(find_dotenv())

def get_weather(city: str):
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'appid': weather_api_key,
        'q': city,
        'units': 'metric'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        pressure = data['main']['pressure']
        visibility = data['visibility']
        sunrise = datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M')
        sunset = datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M')
        
        return (f"Weather in {city}:\n"
                f"• Condition: {weather_desc}\n"
                f"• Temperature: {temp}°C (feels like {feels_like}°C)\n"
                f"• Humidity: {humidity}%\n"
                f"• Wind Speed: {wind_speed} m/s\n"
                f"• Pressure: {pressure} hPa\n"
                f"• Visibility: {visibility/1000} km\n"
                f"• Sunrise: {sunrise}\n"
                f"• Sunset: {sunset}")
    else:
        return f"Could not retrieve weather information for {city}."

# Define OpenAI function calling tools

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city to get weather information for."}
                },
                "required": ["city"]
            }
        }
    }
]


def chat_with_function_calling():
    # Initialize OpenAI client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    # Initialize conversation history
    messages = []
    available_functions = {"get_weather": get_weather}

    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Assistant: Goodbye!")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.9,
                tools=tools,
                tool_choice="auto"
            )
            response_message = response.choices[0].message

            # Check if function call is required
            if response_message.tool_calls:
                # Handle function calls
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        print(f"\n[Function Call] {function_name}")
                        print(f"Input: {function_args}")
                    except json.JSONDecodeError:
                        print("\nError: Invalid JSON in function arguments")
                        continue
                    
                    # Execute function
                    if function_name in available_functions:
                        function_response = available_functions[function_name](**function_args)
                        print(f"Output: {function_response}\n")
                        
                        # Add function call and result to messages
                        messages.append(response_message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": str(function_response)
                        })

                # Get final response after function call
                second_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7
                )
                assistant_response = second_response.choices[0].message.content
            else:
                # Normal response without function call
                assistant_response = response_message.content
                messages.append(response_message)

            print("\nAssistant:", assistant_response)

        except Exception as e:
            print(f"\nError: {str(e)}")
            messages.append({"role": "assistant", "content": "I encountered an error. Please try again."})


if __name__ == "__main__":
    chat_with_function_calling()
