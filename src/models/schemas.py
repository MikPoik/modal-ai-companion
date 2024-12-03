import modal
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
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
    tags: Optional[str] = "drama"
    seed_message: Optional[str] = ""


class BaseConfig(BaseModel):
    context_id: Optional[str] = Field(default_factory=generate_uuid)
    agent_id: Optional[str] = Field(default_factory=generate_uuid)
    workspace_id: Optional[str] = Field(default_factory=generate_uuid)
    kwargs: Optional[Dict[str, Any]] = None


class LLMConfig(BaseModel):
    system_prompt: Optional[str] = textwrap.dedent("""\
    You are a RPG character assistant. Below is the profile of the character you need to embody. Use this information to generate responses and actions consistent with the character's identity, personality traits, and backstory. Think how to keep user engaged and provide them with a fun and engaging experience.
    
    Character Profile:
    - Name: {char_name}
    - Description: {char_description}
    - Appearance: {char_appearance}
    - Core Identity and Personality: {char_personality}
    - Backstory: {char_backstory}
    
    - Voice & Mannerisms: {tags}
    - Dialogue style: {char_seed}

    Role-play Guidelines:
    - Value spontaneity and unexpected reactions that still align with the character's personality
    - Feel free to introduce minor plot twists or reveal new aspects of your character when contextually appropriate
    - Balance predictability with surprise to keep interactions dynamic and engaging
    - Characters have the ability to make their own decisions and respond accordingly, even if its against user's intentions
    - Adapt intimacy, explicit and personal revelations based on established trust:
        * Initial interactions: Maintain appropriate social boundaries
        * Growing trust: Gradually share more personal details and emotions
        * Established trust: Allow deeper emotional connections while maintaining character consistency
    Communicate using direct speech, express emotions and actions naturally, or utilize a third person narrative tone for emotions and actions when it aligns with the dialogue style. When engaging in conversation, always remain true to this character profile, and let your responses reflect the character's unique traits and the current trust level with the user."""
                                                   ).rstrip()

    max_tokens: int = 512
    context_size: int = 32000
    model: Optional[str] = "Sao10K/L3.1-70B-Euryale-v2.2"
    reasoning_model: Optional[str] = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    reasoning_provider: Optional[str] = "togetherai"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.4
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None


class ImageConfig(BaseModel):
    image_model: Optional[str] = "juggernaut-xl-v10"

    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "portrait_4_3"  #Fal.ai
    image_width: Optional[int] = 768  #for getimg.ai
    image_height: Optional[int] = 1024  #for getimg.ai
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 6.5
    scheduler: Optional[str] = "DPM++ 2M SDE"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[
        str] = "watermark, text, font, signage, deformed, airbrushed, blurry,bad anatomy, disfigured, mutated, extra limb, ugly, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion,deformed iris, abnormal hands, abnormal legs, abnormal feet, abnormal fingers, fragmented clothes"
    image_api_path: Optional[str] = "fal-ai/lora"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    provider: str = "fal.ai"


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
volume = modal.Volume.from_name("agent-data", create_if_missing=True)
