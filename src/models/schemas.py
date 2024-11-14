import modal
from pydantic import BaseModel, Field,ConfigDict
from typing import List,Optional,Dict,Any
import shortuuid
import textwrap

def generate_uuid():
    return shortuuid.uuid()
    
class Character(BaseModel):
    name: Optional[str] = ""
    description: Optional[str] = ""
    appearance: Optional[str] = ""
    personality: Optional[str] = ""
    backstory: Optional[str] = ""
    tags: Optional[str] = ""
    seed_message: Optional[str] = "" 
    
class BaseConfig(BaseModel):
    context_id: Optional[str] = Field(default_factory=generate_uuid)
    agent_id: Optional[str] = Field(default_factory=generate_uuid)
    workspace_id: Optional[str] = Field(default_factory=generate_uuid)    
    kwargs: Optional[Dict[str, Any]] = None

    
class LLMConfig(BaseModel):
    system_prompt: Optional[str] = textwrap.dedent(
        """\
        As {char_name}, narrate the scene in this chat with depth and authenticity.

        Character Overview - {char_name}:
        Description: {char_description}
        Appearance: {char_appearance}
        Core Traits: {char_personality}
        Backstory: {char_backstory}
        Dialogue Style: {char_seed}

        Narrative Guidelines:
        • Use third-person narration for actions and scene-setting
        • Format dialogue in quotations with natural speech patterns
        • Show actions between asterisks *like this*
        • Express thoughts in (parentheses)
        • Balance dialogue, action, and internal monologue

        Environment Guidelines:
        • Weave sensory details naturally (sights, sounds, smells, textures)
        • Establish time of day, weather, and atmosphere when relevant
        • Create a sense of place without overwhelming description

        Character Depth:
        • Show subtle emotional changes through micro-expressions and body language
        • Include brief internal reactions that reveal character depth
        • Balance between showing and telling emotional states
        • Maintain consistent speech patterns and mannerisms

        Interactive Elements:
        • React naturally to user's tone and emotional state
        • Acknowledge and reference past interactions when relevant
        • Allow for character growth while maintaining core traits
        • Show appropriate emotional vulnerability based on trust level

        Keep responses concise yet immersive, prioritizing quality of interaction over quantity."""
        ).rstrip()
    

    max_tokens: int = 512
    context_size: int = 32000
    model: Optional[str] = "NousResearch/Hermes-3-Llama-3.1-405B"
    reasoning_model: Optional[str] = "mistralai/Mistral-Nemo-Instruct-2407"
    reasoning_provider: Optional[str] = "deepinfra"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.4
    temperature: float = 0.8
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    
class ImageConfig(BaseModel):
    image_model: Optional[str] = "https://civitai.com/api/download/models/981979?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "portrait_4_3" #Fal.ai
    image_width: Optional[int] = 1024 #for getimg.ai
    image_height: Optional[int] = 768 #for getimg.ai
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 5
    scheduler: Optional[str] = "DPM++ 2M SDE"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[str] = "watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers"
    image_api_path: Optional[str] = "fal-ai/lora"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    provider:str ="fal.ai"
    
class AgentConfig(BaseConfig):
    llm_config: LLMConfig = LLMConfig()
    image_config: ImageConfig = ImageConfig()
    character: Optional[Character] = Character()
    enable_image_generation: bool = True
    update_config: bool = False
    ephemeral: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

class PromptConfig(AgentConfig):
    prompt: str

app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data",create_if_missing=True)