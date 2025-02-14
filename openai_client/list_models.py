import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Configure OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

print(f"Base URL: {client.base_url}")


def list_models():
    try:
        # Get list of available models
        # Fix the API endpoint URL by adding a forward slash
        models = client.models.list()
        
        print("\nAvailable Models:")
        print("-" * 50)
        
        # Print each model's ID and creation date
        for model in models:
            print(f"ID: {model.id}")
            print(f"Created: {model.created}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        # Print the base URL for debugging

if __name__ == "__main__":
    list_models()
