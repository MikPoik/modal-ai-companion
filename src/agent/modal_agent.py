import modal
from src.models.schemas import AgentConfig, Generation, LLMConfig, volume,app
from src.handlers.llm_handler import LLMHandler
from src.handlers.image_handler import ImageHandler
from src.handlers.index_handler import IndexHandler
from src.handlers.chat_handler import ChatHandler
from src.handlers.agent_config_handler import AgentConfigHandler
from src.services.file_service import FileService
from src.services.cache_service import CacheService
from typing import Generator, Optional


agent_image = (
modal.Image.debian_slim(python_version="3.10")
.pip_install(
    "openai==1.47",
    "pydantic==2.6.4",
    "requests",
    "shortuuid",
    "annoy"
)
)

with agent_image.imports():
    import json
    import os
    from typing import List, Optional
    import shortuuid
    import requests
    import pickle
    from annoy import AnnoyIndex
    import re
    import textwrap
    from openai import OpenAI

# Define secrets and mounts
gcp_hmac_secret = modal.Secret.from_name(
"gcp-secret",
required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"]
)

@app.cls(
timeout=60 * 2,
container_idle_timeout=60 * 15,
allow_concurrent_inputs=10,
image=agent_image,
secrets=[
    modal.Secret.from_name("gcp-secret"),
    modal.Secret.from_name("deep-infra-api-key"),
    modal.Secret.from_name("falai-apikey")
],
volumes={
    "/data": volume,
    "/bucket-mount": modal.CloudBucketMount(
        bucket_name="modal-agent-chat-test",
        bucket_endpoint_url="https://storage.googleapis.com",
        secret=gcp_hmac_secret
    ),
    "/cloud-images": modal.CloudBucketMount(
        bucket_name="coqui-samples",
        bucket_endpoint_url="https://storage.googleapis.com",
        secret=gcp_hmac_secret
    )
}
)
class ModalAgent:
    def __init__(self):
        print("Initializing Agent")
        # Initialize handlers
        self.llm_handler = LLMHandler()
        self.image_handler = ImageHandler()
        self.index_handler = IndexHandler()
        self.chat_handler = ChatHandler()
        self.config_manager = AgentConfigHandler()
    
        # Initialize services
        self.file_service = FileService()
        self.cache_service = CacheService()

    
    @modal.method()
    def get_or_create_agent_config(self, agent_config: AgentConfig, update_config: bool = False) -> AgentConfig:
        """Handle agent configuration management"""
        return self.config_manager.get_or_create_config(agent_config, update_config)
    
    @modal.method()
    async def generate_avatar(self, prompt: str,agent_config:AgentConfig) -> Optional[str]:
        """Generate avatar using image handler"""
        return self.image_handler.generate_avatar(prompt, agent_config)
    
    @modal.method(is_generator=True)
    def run(self, generation: Generation, agent_config: Optional[AgentConfig] = None) -> Generator[str, None, None]:
        """Main method to handle generation requests"""
        try:
            # Get or create agent configuration
            agent_config = self.get_or_create_agent_config.local(
                agent_config or AgentConfig(
                    context_id=generation.context_id,
                    agent_id=generation.agent_id,
                    workspace_id=generation.workspace_id,
                )
            )
            formatted_config = json.dumps(agent_config.model_dump(), indent=4)
            print(f"Agent config: {formatted_config}")
        
            # Get chat history and prepare messages
            messages = self.chat_handler.prepare_messages(
                generation.prompt,  # Pass the prompt string
                agent_config       # Pass the AgentConfig object, not chat history
            )
            print(f"Prepared messages: {messages}")

            llm_response = ""
            # Generate response using LLM
            for token in self.llm_handler.generate(messages, agent_config):
                llm_response += token
                yield token
                
            messages.append({
                    "role": "assistant",
                    "content": f"{llm_response}"
                })
            
            # Generate image if enabled
            if agent_config.enable_image_generation:
                is_image_request = self.image_handler.check_for_image_request(messages, agent_config)
                print("Is imaging request:", is_image_request)
                if is_image_request:
                    image_url = self.image_handler.request_image_generation(messages, agent_config)
                    if image_url:
                        yield image_url
                        
            # Save updated chat history
            self.chat_handler.save_chat_history(messages, agent_config)
    
        except Exception as e:
            print(f"Error in run method: {str(e)}")
            yield f"Error: {str(e)}"
    
