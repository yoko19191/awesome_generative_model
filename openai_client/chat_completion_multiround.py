from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Configure OpenAI client
model_name = "gpt-4o-mini"

def chat():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    # Store conversation history
    messages = []
    
    print("Chat with AI (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for quit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Get response from OpenAI
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,  # Controls randomness: lower is more deterministic
                max_tokens=150,  # Maximum number of tokens in the response
                top_p=1.0,  # Nucleus sampling: consider the top_p probability mass
                frequency_penalty=0.0,  # Penalize new tokens based on their frequency
                presence_penalty=0.0,  # Penalize new tokens based on their presence
                stop=["\n"]  # Stop sequence(s) to end the generation
            )
        
            # Print full response details
            print("\n" + "="*50)
            print("Response Details:")
            print("-"*20)
            print(f"Model: {response.model}")
            print(f"Created: {response.created}")
            print(f"Response ID: {response.id}")
            print(f"Prompt Tokens: {response.usage.prompt_tokens}")
            print(f"Completion Tokens: {response.usage.completion_tokens}")
            print(f"Total Tokens: {response.usage.total_tokens}")
            print("-"*20)
            
            # Extract assistant's reply
            assistant_reply = response.choices[0].message.content
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_reply})
            
            # Print assistant's reply with separator
            print("\nAssistant Response:")
            print("-"*20)
            print(assistant_reply)
            print("="*50)
            
        except Exception as e:
            print(f"Base URL: {client.base_url}")
            print(f"\nError: {str(e)}")
            messages.pop()  # Remove the failed message from history

if __name__ == "__main__":
    chat()
