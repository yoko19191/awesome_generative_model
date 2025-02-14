from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model_name = "deepseek-chat"


print(f"Base URL: {base_url}")

def main():
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # Store conversation history
    messages = []
    
    print("Chat with AI (type 'quit' to exit)")
    print("-" * 50)
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for quit command
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
                
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Print "AI: " before the response starts streaming
            print("\nAI: ", end="", flush=True)
            
            # Create streaming chat completion
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            
            # Collect the full response while streaming
            full_response = ""
            
            # Stream the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            # Add assistant's message to history
            messages.append({"role": "assistant", "content": full_response})
            print()  # New line after response
            
    except KeyboardInterrupt:
        print("\n\nChat ended by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
