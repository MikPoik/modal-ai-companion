import modal
from pydantic import BaseModel, Field
from typing import List,Optional
import shortuuid

def generate_uuid():
    return shortuuid.uuid()
    
class BaseConfig(BaseModel):
    context_id: Optional[str] = Field(default_factory=generate_uuid)
    agent_id: Optional[str] = Field(default_factory=generate_uuid)
    workspace_id: Optional[str] = Field(default_factory=generate_uuid)
    model: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
    provider: Optional[str] = "deepinfra"
    update_config: bool = False
    
class LLMConfig(BaseModel):
    system_prompt: Optional[str] = ""
    max_tokens: int = 512
    context_size: int = 4096
    
class ImageConfig(BaseModel):
    image_model: Optional[str] = "https://civitai.com/api/download/models/312314"
    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "square_hd"
    num_inference_steps: Optional[int] = 10
    guidance_scale: Optional[float] = 4.0
    scheduler: Optional[str] = "DPM++ 2M SDE Karras"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[str] = "disfigured,deformed, poorly drawn, extra limbs, blurry:0.25"
    image_api_path: Optional[str] = "fal-ai/lora"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    
class AgentConfig(BaseConfig):
    llm_config: LLMConfig = LLMConfig()
    image_config: ImageConfig = ImageConfig()
    backstory: Optional[str] = """
    Luna's primary goal is to assist, inform, and inspire the humans she interacts with, always striving to make a positive impact in every conversation.
    Luna likes Pizza
    """

class Generation(BaseConfig):
    prompt: str
    system_prompt: Optional[str] = ""

    
app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data",create_if_missing=True)