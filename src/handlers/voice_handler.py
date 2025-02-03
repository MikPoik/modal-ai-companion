
import os
import requests
from src.models.schemas import AgentConfig
from src.services.file_service import FileService

class VoiceHandler:
    def __init__(self):
        self.file_service = FileService('/cloud-images')
        self.api_url = "https://api.deepinfra.com/v1/inference/hexgrad/Kokoro-82M"
        self.api_key = os.environ.get("DEEP_INFRA_API_KEY")

    def generate_voice(self, text: str, agent_config: AgentConfig) -> str:
        """Generate voice from text using DeepInfra Kokoro model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        print(response.json())
        
        # Save audio to GCP bucket
        audio_data = response.content
        voice_url = self.file_service.save_binary_to_bucket(
            audio_data,
            agent_config,
            "voice",
            f"{agent_config.context_id}.wav"
        )
        
        return voice_url
