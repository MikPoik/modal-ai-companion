# src/handlers/llm_handler.py
from typing import Generator, Dict, List
from src.models.schemas import AgentConfig
import os,json



class LLMHandler:
    def __init__(self):
        from openai import OpenAI
        import os
        self.client = None
        self._provider_configs = {
            "deepinfra": {
                "base_url": "https://api.deepinfra.com/v1/openai",
                "api_key_env": "DEEP_INFRA_API_KEY"
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY"
            },
            "togetherai": {
                "base_url": "https://api.together.ai/v1",
                "api_key_env": "TOGETHERAI_API_KEY"
            }
        }

    def initialize_client(self, provider: str):
        from openai import OpenAI
        from together import Together
        """Initialize OpenAI client with provider-specific configuration."""
        config = self._provider_configs.get(provider, self._provider_configs["openai"])
        #print("*** LLM CONFIG ***: ",config)
        if config["api_key_env"] not in os.environ:
            raise ValueError(f"Missing API key for provider {provider}")
        print("Initializing llm client with provider: ", provider)
        if provider.strip() == "togetherai":
            return Together(
                api_key=os.environ[config["api_key_env"]]
            )
        elif provider.strip() in ["deepinfra", "openai"]:
            # Create client options dict, excluding proxies if present
            client_options = {
                "base_url": config["base_url"],
                "api_key": os.environ[config["api_key_env"]]
            }
            
            # Filter out any environment-injected proxy settings
            if "http_proxy" in os.environ:
                print("Note: http_proxy environment variable detected but not used")
            if "https_proxy" in os.environ:
                print("Note: https_proxy environment variable detected but not used")
                
            return OpenAI(**client_options)

    def generate(self, 
                messages: List[Dict], 
                agent_config: AgentConfig,
                 temperature=None,
                 model=None,
                 provider=None,
                 stop_words=None,
                 max_tokens=None,
                 frequency_penalty=None,
                 presence_penalty=None,
                 repetition_penalty=None,
                 top_p=None,
                 top_k=None,
                 min_p=None) -> Generator[str, None, None]:
        """Generate text using the configured LLM provider."""
        if not agent_config.llm_config.provider:
            raise ValueError("LLM provider not specified in config")
        
        together_ai_models = ['meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo','NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO','meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo','mistralai/Mixtral-8x22B-Instruct-v0.1','Gryphe/MythoMax-L2-13b']
        openai_models = ['gpt4-o-mini','gpt4-o']
        deepinfra_models = ['NousResearch/Hermes-3-Llama-3.1-405B','Sao10K/L3.3-70B-Euryale-v2.3','Sao10K/L3.1-70B-Euryale-v2.2','mistralai/Mistral-Small-24B-Instruct-2501','nvidia/Llama-3.1-Nemotron-70B-Instruct','meta-llama/Llama-3.3-70B-Instruct-Turbo','meta-llama/Meta-Llama-3.1-405B-Instruct','meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo']
        
        extra_body = {}
        
        if agent_config.llm_config.model in together_ai_models:
            provider = "togetherai"
        if agent_config.llm_config.model in openai_models:
            provider = "openai"
        if agent_config.llm_config.model in deepinfra_models:
            provider = "deepinfra"

        cleaned_messages = []
        for msg in messages:
            if 'content' in msg:
                cleaned_messages.append({'role': msg['role'] ,'content': msg['content']})

        payload = {
            "model": model or agent_config.llm_config.model,
            "messages": cleaned_messages,
            "stream": True,
            "temperature": temperature or agent_config.llm_config.temperature,
            "max_tokens": max_tokens or agent_config.llm_config.max_tokens,
            "stop": stop_words if stop_words is not None else agent_config.llm_config.stop,
            "frequency_penalty": frequency_penalty or agent_config.llm_config.frequency_penalty,
            "presence_penalty": presence_penalty or agent_config.llm_config.presence_penalty,
            "top_p": top_p or agent_config.llm_config.top_p,
        }

        
        if provider == 'deepinfra':
            # Use None check instead of logical OR to handle zero values correctly
            extra_body['min_p'] = min_p if min_p is not None else agent_config.llm_config.min_p
            extra_body['repetition_penalty'] = repetition_penalty if repetition_penalty is not None else agent_config.llm_config.repetition_penalty
            payload['extra_body'] = extra_body
            
        if provider == 'togetherai':
            payload['min_p'] = min_p if min_p is not None else agent_config.llm_config.min_p
            payload['repetition_penalty'] = repetition_penalty if repetition_penalty is not None else agent_config.llm_config.repetition_penalty
            
        provider_name = provider or agent_config.llm_config.provider
        print(f"Initializing llm client with provider: {provider_name}")
        try:
            self.client = self.initialize_client(provider_name)
            print(f"Client initialized: {type(self.client).__name__}")
            
            # Deep debug of payload
            print("Payload keys:", list(payload.keys()))
            
            # Create a clean copy of the payload
            request_payload = payload.copy()
            
            # Check for and remove problematic parameters
            problem_params = ['proxies']
            for param in problem_params:
                if param in request_payload:
                    print(f"Removing '{param}' parameter as it's not supported by the client")
                    del request_payload[param]
            
            print(f"Calling chat.completions.create with provider: {provider_name}, model: {request_payload.get('model', 'unknown')}")
            response = self.client.chat.completions.create(**request_payload)
            for chunk in response:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_message = f"Error in run method: {type(e).__name__}: {str(e)}"
            print(error_message)
            
            # Print stack trace for debugging
            import traceback
            print("Stack trace:")
            traceback.print_exc()
            
            yield f"Error: {error_message}"

    def _get_provider_config(self, provider: str) -> tuple[str, str]:
        """Get provider configuration (base URL and API key)."""
        config = self._provider_configs.get(provider, self._provider_configs["openai"])
        #if config["api_key_env"] not in os.environ:
            #raise ValueError(f"Missing API key for provider {provider}")
        return config["base_url"], os.environ[config["api_key_env"]]