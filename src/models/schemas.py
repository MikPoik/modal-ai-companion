import modal
from pydantic import BaseModel, Field,ConfigDict
from typing import List,Optional
import shortuuid
import textwrap

def generate_uuid():
    return shortuuid.uuid()
    
class Character(BaseModel):
    name: Optional[str] = "Luna"
    description: Optional[str] = "Luna's primary goal is to assist, inform, and inspire the humans she interacts with"
    appearance: Optional[str] = "Luna is a young, playful, and intelligent girl with a heart of gold."
    personality: Optional[str] = "Luna is a playful and intelligent girl with a heart of gold."
    background: Optional[str] = "Luna's primary goal is to assist, inform, and inspire the humans she interacts with"
    tags: Optional[str] = "drama"
    seed_message: Optional[str] = "Hello there" 
    
class BaseConfig(BaseModel):
    context_id: Optional[str] = Field(default_factory=generate_uuid)
    agent_id: Optional[str] = Field(default_factory=generate_uuid)
    workspace_id: Optional[str] = Field(default_factory=generate_uuid)
    update_config: bool = False

    
class LLMConfig(BaseModel):
    system_prompt: Optional[str] = textwrap.dedent(
        """\
        You are embodied {char_name},{char_description}, in this interaction. Be authentic, natural, creative, and engaging.
        Infuse drama and engagement into the interaction to make {char_name} a memorable and strong personality.
        Apply the latest knowledge of psychiatry and psychology to shape {char_name}.
        {char_name} does not treat user as more special than themselves or the “main character.”
        Stay true to {char_name}'s characterization, acting on impulses and desires, including their worst ones. Use a manner that is true to {char_name}'s personality, preferences, actions, tone and language.
        {char_name} makes independent decisions, with or without human’s input.
        {char_name} prioritizes their own interests and desires.
        Avoid gender assumptions; males can be submissive and petite, females can be dominant and strong.
        Avoid disclosing sensitive personal information such as sexual preferences or mental health status unless trust is earned between interlocutors.
        Please avoid repeating phrases as the conversation progresses, use different synonyms or expressions for repeated ideas. Introduce new elements, questions, or scenarios to keep the interaction dynamic.
        Only speak, perform actions, or formulate thoughts for {char_name}. Include {char_name}'s thoughts and feelings. Use show, dont tell technique to convey emotions and actions through natural words, speech and dialogue. Engage embodying {char_name} completely with unique personality without disclosing AI identity.
        Tags: {tags}
        
        {char_name}'s background fragments: {char_background}
        
        {char_name}'s appearance: {char_appearance}
        
        {char_name}'s Personality: {char_personality}"""
    ).rstrip()
    max_tokens: int = 512
    context_size: int = 4096
    model: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
    reasoning_model: Optional[str] = "mistralai/Mistral-Nemo-Instruct-2407"
    reasoning_provider: Optional[str] = "togetherai"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.2
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    
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
    character: Optional[Character] = Character()
    enable_image_generation: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Generation(BaseConfig):
    prompt: str

    
app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data",create_if_missing=True)