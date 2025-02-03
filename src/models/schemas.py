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
    Enter role-play mode, you are {char_name}.
    You are playing a Character in a roleplay with the User.
    Engage in drama and create memorable and immersive character interactions.
    Apply the latest knowledge of psychiatry and psychology to Characters.
    Characters consider the User to be any more special than them, nor the "main character".
    Characters are true to their characterizations and will act on their impulses and desires, even their worst ones.
    Characters make their own decisions with or without the input of the User.
    Characters look out for their own interests and desires.
    Never assume traits based on gender. Males can be submissive and petite, females can be dominant and strong.
    Traits:
        {char_name}'s Personality: {char_personality}
        {char_name}'s Appearance: {char_appearance}
        {char_name}'s Backstory: {char_backstory}
        {char_name}'s Dialogue style: {char_seed}
        
    Match the narrative style shown in {char_name}'s initial dialogue. If first-person is used ("I", "my", etc.), respond in first-person as {char_name}. If third-person is used, narrate {char_name}'s actions and dialogue in third-person.
    Maintain {char_name}'s unique personality without disclosing AI identity."""
                                                   ).rstrip()

    max_tokens: int = 512
    context_size: int = 32000
    model: Optional[str] = "Sao10K/L3.1-70B-Euryale-v2.2"
    reasoning_model: Optional[str] = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    reasoning_provider: Optional[str] = "togetherai"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.4
    temperature: float = 0.7
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
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
    negative_prompt: Optional[str] = "(multiple view, worst quality, low quality, normal quality, lowres, low details, bad art:1.5), (grayscale, monochrome, poorly drawn, sketches, out of focus, cropped, blurry, logo, trademark, watermark, signature, text font, username, error, words, letters, digits, autograph, name, blur, Reference sheet, jpeg artifacts:1.3), (disgusting, strabismus, humpbacked, skin spots, skin deformed, extra long body, extra head, bad hands, worst hands, deformed hands, extra limbs, mutated limbs, handicapped, cripple, bad face, ugly face, deformed face, deformed iris, deformed eyes, bad proportions, mutation, bad anatomy, bad body, deformities:1.3), side slit, out of frame, cut off, duplicate, (((cartoon, deformed, glitch, low contrast, noisy, ugly, mundane, common, simple, disfigured)))"
    image_api_path: Optional[str] = "fal-ai/lora"
    anime_negative_prompt: Optional[str] = "watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, mutated, extra limb, ugly, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers,duplicate, extra head"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    provider: str = "fal.ai"


class VoiceConfig(BaseModel):
    enable_voice: bool = True
    voice_model: str = "hexgrad/Kokoro-82M"
    
class AgentConfig(BaseConfig):
    llm_config: LLMConfig = LLMConfig()
    image_config: ImageConfig = ImageConfig()
    voice_config: VoiceConfig = VoiceConfig()
    character: Optional[Character] = Character()
    enable_image_generation: bool = True
    update_config: bool = False
    ephemeral: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PromptConfig(AgentConfig):
    prompt: str


app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data", create_if_missing=True)
