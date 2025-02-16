from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from typing import List, Dict, Optional, Union, Tuple

# Load environment variables
_ = load_dotenv(find_dotenv())

class LLMClient:
    def __init__(self, 
                 provider: str = "openai",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        """Initialize LLM client with configuration."""
        providers = {
            "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "gpt-4o-mini"),
            "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "deepseek-chat"),
            "dashscope": ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "qwen-max"),
            "zhipu": ("ZHIPU_API_KEY", "ZHIPU_BASE_URL", "glm-4-flash"),
            "siliconflow": ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL", "gpt-4o-mini"),
            "ollama": ("OLLAMA_API_KEY", "OLLAMA_BASE_URL", "qwen2.5:latest"),
        }
        
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
            
        api_key_env, base_url_env, default_model = providers[provider]
        
        self.client = OpenAI(
            api_key=api_key or os.getenv(api_key_env),
            base_url=base_url or os.getenv(base_url_env)
        )
        self.model = model or default_model
        
    def list_models(self):
        """List available models from the provider."""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []
        
    def get_completion(self, 
                      prompt: str,
                      temperature: float = 0.7,
                      max_tokens: int = 500,
                      tools: Optional[List[Dict]] = None,
                      tool_choice: Optional[str] = None) -> Union[str, Tuple[str, str]]:
        """Get a simple completion response for a single prompt."""
        try:
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
                
            response = self.client.chat.completions.create(**params)
            if hasattr(response.choices[0], 'reasoning_content'):
                return response.choices[0].reasoning_content, response.choices[0].message.content
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in get_completion: {str(e)}")
            return ""

    def get_chat_completion(self,
                          messages: List[Dict[str, str]],
                          temperature: float = 0.7,
                          max_tokens: int = 500,
                          tools: Optional[List[Dict]] = None,
                          tool_choice: Optional[str] = None) -> Union[str, Tuple[str, str]]:
        """Get completion for a chat conversation."""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
                
            response = self.client.chat.completions.create(**params)
            if hasattr(response.choices[0], 'reasoning_content'):
                return response.choices[0].reasoning_content, response.choices[0].message.content
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in get_chat_completion: {str(e)}")
            return ""

    def stream_chat_completion(self,
                             messages: List[Dict[str, str]],
                             temperature: float = 0.7,
                             max_tokens: int = 500,
                             tools: Optional[List[Dict]] = None,
                             tool_choice: Optional[str] = None):
        """Stream chat completion responses."""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
                
            response = self.client.chat.completions.create(**params)
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
                    yield ('reasoning', chunk.choices[0].delta.reasoning_content)
                if chunk.choices[0].delta.content is not None:
                    yield ('content', chunk.choices[0].delta.content)
        except Exception as e:
            print(f"Error in stream_chat_completion: {str(e)}")
            yield ""
            
            
    def get_chat_completion_in_format(self,
                                    messages: List[Dict[str, str]],
                                    response_format: Dict[str, str] | str = "json",
                                    temperature: float = 0.9,
                                    max_tokens: int = 500) -> str:
        """Get chat completion with specified response format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            response_format: Either "json" string or a format dict like {"type": "json_object"}
            temperature: Sampling temperature between 0 and 1
            max_tokens: Maximum tokens to generate
            
        Returns:
            The model's response as a string
        """
        try:
            # Convert string format to dict format if needed
            if isinstance(response_format, str) and response_format == "json":
                response_format = {"type": "json_object"}
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in get_chat_completion_in_format: {str(e)}")
            return ""
        
    def get_chat_completion_multiple(self,
                                   messages: List[Dict[str, str]],
                                   num_completions: int = 3,
                                   temperature: float = 0.7,
                                   max_tokens: int = 500) -> List[str]:
        """Get multiple chat completion responses for the same messages."""
        try:
            responses = []
            for _ in range(num_completions):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                responses.append(response.choices[0].message.content)
            return responses
        except Exception as e:
            print(f"Error in get_chat_completion_multiple: {str(e)}")
            return []
        
    def get_completion_with_FIM(self,
                                 prompt: str,
                                 suffix: str,
                                 temperature: float = 0.7,
                                 max_tokens: int = 500) -> str:
        """Get completion with a FIM."""
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                suffix=suffix,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Error in get_completion_with_FIM: {str(e)}")
            return ""
        
    
    async def get_chat_completion_async(self,
                                      messages: List[Dict[str, str]],
                                      temperature: float = 0.7,
                                      max_tokens: int = 500) -> str:
        """Get chat completion response asynchronously."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in get_chat_completion_async: {str(e)}")
            return ""
        
    def get_embedding(self, text: str, model: Optional[str] = "text-embedding-3-small") -> List[float]:
        """Get embedding for a text input."""
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in get_embedding: {str(e)}")
            return []

    async def get_embedding_async(self, text: str, model: Optional[str] = "text-embedding-3-small") -> List[float]:
        """Get embedding for a text input asynchronously."""
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in get_embedding_async: {str(e)}")
            return []

def create_llm_client(provider: str = "openai") -> LLMClient:
    """Factory function to create LLM client for different providers."""
    return LLMClient(provider=provider)
    
def get_model_list(self) -> List[str]:
    """Get list of available models."""
    try:
        models = self.client.models.list()
        model_ids = [model.id for model in models]
        return model_ids
        
    except Exception as e:
        print(f"Error getting model list: {str(e)}")
        return []

    
