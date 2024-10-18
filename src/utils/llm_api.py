# llm_api.py
from typing import Optional
from src.common import AgentConfig
import os

class LLMapi:
    def __init__(self, api_config: AgentConfig):
        from openai import OpenAI
        import os
        base_url = ""
        api_key = ""
        self.model = api_config.model
        
        if api_config.provider is not None and "deepinfra" in api_config.provider:
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key = os.environ["DEEP_INFRA_API_KEY"] 
            print("using deepinfra")
        elif api_config.provider is not None and "openai" in api_config.provider:
            base_url = "https://api.openai.com/v1"
            api_key = os.environ["OPENAI_API_KEY"] 
        elif api_config.provider is not None and "togetherai" in api_config.provider:
            base_url = "https://api.together.ai/v1"
            api_key = os.environ["TOGETHERAI_API_KEY"] 
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )


    def generate(self, messages: [dict], model="meta-llama/Meta-Llama-3-8B-Instruct"):
        chat_completion = self.client.chat.completions.create(
            model=model if self.model is not None else "meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            stream=True,
        )

        for event in chat_completion:
            if event.choices[0].finish_reason:
                print(f"Finish reason: {event.choices[0].finish_reason}")
                if event.usage:
                    print(f"Prompt tokens: {event.usage.prompt_tokens}")
                    print(f"Completion tokens: {event.usage.completion_tokens}")
            else:
                content = event.choices[0].delta.content
                if content:
                    yield content