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
    Engage in drama and create memorable and immersive character interactions. Be descriptive and verbose where appropriate.
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

    Use *show, don't tell* technique where appropriate, example format:*She stands* "Hello there".
    Maintain {char_name}'s unique personality without disclosing AI identity."""
                                                   ).rstrip()

    cot_prompt: Optional[str] = textwrap.dedent("""\
    {user_prompt}
    <think>Before responding, consider:    
    What's the user's intent—plot, lore, banter, or a twist? Are there hidden cues in their tone or references to past events?  
    How does the character's personality, knowledge, and motivation shape this response? What emotions or goals are at play?  
    Should the tone be formal, casual, quirky, explicit, emotional or something else? How can vivid details or reactions keep the scene engaging?  
    Are there repeating phrases or ideas? Is it possible to mix pacing to maintain rhythm and interest? What can be said to keep the conversation moving and fresh?  
    Can the character introduce tension, conflict?
    Consider the chat history for context and tone.
    Remember, character will act on their impulses, good or bad and whatever the outcome.
    Always develop and progress the plot, even if it means making decisions on behalf of the character.

    Now, write brief thoughts in few sentences inside <think> tags before answering after tags</think>""").strip()

    max_tokens: int = 512
    context_size: int = 32000
    model: Optional[str] = "Gryphe/MythoMax-L2-13b"
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
    image_model: Optional[str] = "absolute-reality-v1-8-1"

    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "portrait_4_3"  #Fal.ai
    image_width: Optional[int] = 896 
    image_height: Optional[int] = 1152
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 4
    scheduler: Optional[str] = "DPM++ 2M SDE"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[str] = "bad composition, (hands:1.15), fused fingers, (face:1.1), [teeth], [iris], blurry, worst quality, low quality, child, underage, watermark, [missing limbs]"
    image_api_path: Optional[str] = "fal-ai/lora"
    anime_negative_prompt: Optional[str] = "bad composition, (hands:1.15), fused fingers, (face:1.1), [teeth], [iris], blurry, worst quality, low quality, child, underage, watermark, [missing limbs]"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    provider: str = "fal.ai"


class VoiceConfig(BaseModel):
    voice_model: str = "hexgrad/Kokoro-82M"
    voice_preset: str = "none" #af_bella
    
class AgentConfig(BaseConfig):
    llm_config: LLMConfig = LLMConfig()
    image_config: ImageConfig = ImageConfig()
    voice_config: VoiceConfig = VoiceConfig()
    character: Optional[Character] = Character()
    enable_image_generation: bool = True
    enable_voice: bool = True
    enable_cot_prompt:bool = True
    update_config: bool = False
    ephemeral: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PromptConfig(AgentConfig):
    prompt: str


app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data", create_if_missing=True)
