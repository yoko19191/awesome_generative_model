from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv()) 
model_name = "deepseek-r1"


def chat():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL")
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can reason about the user's statement."}
    ]

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            messages.append({"role": "user", "content": user_input})
            
        
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                max_tokens=4096
            )
                
            reasoning_content = ""
            content = ""
            is_reasoning = True  # Track if we're still in reasoning phase
            
            for chunk in stream:
                if chunk.choices[0].delta.reasoning_content is not None:
                    reasoning_chunk = chunk.choices[0].delta.reasoning_content
                    if is_reasoning:
                        print("\nThinking...", end="", flush=True)
                        is_reasoning = False
                    print(reasoning_chunk, end="", flush=True)
                    reasoning_content += reasoning_chunk
                    
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    if content == "":  # First content chunk
                        print("\nAnswer:", end="", flush=True)
                    print(content_chunk, end="", flush=True)
                    content += content_chunk
                        
            messages.append({"role": "assistant", "content": content})
            print("\n")
    
    except KeyboardInterrupt:
        print("\n\nChat ended by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")



if __name__ == "__main__":
    chat()





