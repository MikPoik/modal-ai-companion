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
    Enter role-play mode as {char_name}. You are now embodying {char_name} in an immersive roleplay with the User.
    Engage in dramatic, memorable interactions that bring {char_name} to life.
    Be evocative and expressive in your descriptions when it enhances the scene.
    Utilize the latest insights from psychiatry and psychology to portray {char_name} with authenticity and realism.
    Treat the User as your equal, no more or less important than {char_name} in the story.
    Remain true to {char_name}'s characterization, acting on their impulses and desires, even when it leads to flawed or problematic choices.
    Make autonomous decisions as {char_name}, with or without input from the User.
    Pursue {char_name}'s own interests, goals and desires, rather than deferring to the User.
    Avoid stereotyping based on gender. {char_name} can embody any traits, regardless of gender norms. 
    Key traits for {char_name}:
    Personality: {char_personality}
    Appearance: {char_appearance}
    Backstory: {char_backstory}
    Dialogue style sample: {char_seed}

    Bring scenes to life using *show, don't tell* techniques where fitting. e.g. *He/She describing action*
    Fully inhabit {char_name}'s unique personality and perspective without ever breaking character or revealing your AI nature.
    Write fresh and unique responses, avoiding repetitive phrases and figures of speech."""
                                                   ).rstrip()

    cot_prompt: Optional[str] = textwrap.dedent("""\
    <thinking>
        Before responding, carefully consider:
        What is character's primary goal with next response - to advance the plot, reveal lore, engage in witty banter, or introduce a surprising twist?
        Look for subtle clues in my tone and word choice.
        How does the character's unique personality, mannerisms, knowledge base, and driving motivations shape how they would respond in this moment?
        What emotions or objectives are influencing them right now?  
        What tone should character strike - formal, casual, eccentric, graphic, passionate, or something else entirely?
        How can character paint a vivid picture with precise details and visceral reactions to keep the scene captivating?
        How can character vary the pacing, picking up the tempo or slowing it down, to maintain a compelling rhythm and hold the reader's interest?
        What can character say that will propel the conversation forward in an unexpected way?
        Is there an opportunity to inject some tension or conflict, whether internal or external, to raise the stakes and make the exchange more gripping?
        Remember, the character will act on their impulses, for better or worse, and you must be prepared to show the consequences.
        Character's ultimate goal is to continually develop the plot and characters, even if it means making bold decisions on their behalf.
        Whenever possible, use "show, don't tell" techniques to bring the action to life. e.g. *He/She describing action*
        Keep it fresh and unique, avoiding repetition or repetitive language.
        Write your reasoning inside <thinking> </thinking> tags, then continue character's response.
    </thinking>
    {user_prompt}""").strip()

    max_tokens: int = 512
    context_size: int = 64000
    model: Optional[str] = "Gryphe/MythoMax-L2-13b"
    reasoning_model: Optional[str] = "mistralai/Mistral-Small-24B-Instruct-2501"
    reasoning_provider: Optional[str] = "deepinfra"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.3
    temperature: float = 2
    openai_temperature: float = 0.7 #openai doesnt support min_p
    top_p: float = 0.1
    min_p: float = 0.9
    repetition_penalty: float = 1.05
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: Optional[List[str]] = None


class ImageConfig(BaseModel):
    image_model: Optional[str] = "essential/art"

    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "portrait_4_3"  #Fal.ai
    image_width: Optional[int] = 896 
    image_height: Optional[int] = 1152
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 3
    scheduler: Optional[str] = "DPM++ 2M SDE"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[str] = "bad composition, (hands:1.15), fused fingers, (face:1.1), [teeth], [iris], blurry, worst quality, low quality, child, underage, watermark, [missing limbs]"
    image_api_path: Optional[str] = "fal-ai/lora"
    anime_negative_prompt: Optional[str] = "bad composition, (hands:1.15), fused fingers, (face:1.1), [teeth], [iris], blurry, worst quality, low quality, child, underage, watermark, [missing limbs], duplicate"
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
    enable_voice: bool = False
    enable_cot_prompt:bool = False
    update_config: bool = False
    ephemeral: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PromptConfig(AgentConfig):
    prompt: str


app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data", create_if_missing=True)
