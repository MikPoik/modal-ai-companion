# src/handlers/image_handler.py
from typing import Optional, List, Dict, Any,Literal
import os
import json
import time
import requests
from src.models.schemas import AgentConfig, PromptConfig
from src.handlers.llm_handler import LLMHandler
from src.services.file_service import FileService
import textwrap
import shortuuid
import base64

euler_a_models = [ "156375","286821","303526","378499","293564","384264" ]


class ImageHandler:
    def __init__(self):
        self.file_service = FileService('/cloud-images')
        self.base_url = "https://queue.fal.run"
        self.getimg_base_url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"
        self.api_key = os.environ["FALAI_API_KEY"]
        self.getimg_api_key = os.environ["GETIMGAI_API_KEY"]
        self.llm_handler = LLMHandler()

    def _generate_with_getimg(self, prompt: str, agent_config: AgentConfig) -> Optional[bytes]:
        """Generate image using GetImg API and return binary data"""
        headers = {
            "Authorization": f"Bearer {self.getimg_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": agent_config.image_config.image_model,
            "prompt": prompt,
            "negative_prompt": agent_config.image_config.negative_prompt,
            "width": agent_config.image_config.image_width,
            "height": agent_config.image_config.image_height,
            "steps": agent_config.image_config.num_inference_steps,
            "guidance": agent_config.image_config.guidance_scale,
            "scheduler": "dpmsolver++",
            "output_format": agent_config.image_config.image_format
        }

        try:
            response = requests.post(self.getimg_base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if "image" in result:
                # Decode base64 image to binary
                return base64.b64decode(result["image"])
            return None
        except Exception as e:
            print(f"Error generating image with GetImg: {str(e)}")
            return None
            
    def generate_image(self, prompt: str, agent_config: AgentConfig, folder: str = "images",preallocated_image_name:str = "") -> Optional[str]:
        """Generate an image and save it to the specified folder."""
        try:
            if (agent_config.image_config.provider == "getimg" or "https" not in agent_config.image_config.image_model)\
            and "flux-general-with-lora" not in agent_config.image_config.image_model \
            and "SG161222/Realistic_Vision_V6.0_B1_noVAE" not in agent_config.image_config.image_model \
            and "fal-ai/flux/dev" not in agent_config.image_config.image_model:
                #print(f"Generating image with GetImg")
                image_data = self._generate_with_getimg(prompt, agent_config)
                if image_data:
                    return self.file_service.save_binary_to_bucket(
                        image_data,
                        agent_config,
                        folder,
                        preallocated_image_name=preallocated_image_name
                    )
                return None
            else:
                # Existing FAL.ai implementation
                status_url = self._submit_job(prompt, agent_config)
                if not status_url:
                    print("Failed to submit image generation job")
                    return None
                image_url = self._check_status_and_download_image(status_url)
                if not image_url:
                    print("Failed to generate image")
                    return None
                return self.file_service.save_image_to_bucket(
                    image_url,
                    agent_config,
                    folder,
                    preallocated_image_name=preallocated_image_name
                )
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def generate_avatar(self, agent_config: PromptConfig,folder="") -> Optional[str]:
        """Generate an avatar image."""
        
        return self.generate_image(agent_config.prompt,agent_config,folder= "")
        
    def format_image_size(self,image_size):
        """Helper to format image size from string or dict"""
        
        if isinstance(image_size, str):
            try:
                # Check if string is JSON formatted
                parsed_size = json.loads(image_size)
                if isinstance(parsed_size, dict):
                    return parsed_size
                return image_size
            except json.JSONDecodeError:
                # If not valid JSON, return original string (like "portrait_4_3")
                return image_size
        return image_size
        
    def _submit_job(self, prompt: str, agent_config: AgentConfig) -> Optional[str]:
        """Submit image generation job to FAL AI."""
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }



        scheluder_config = agent_config.image_config.scheduler
        negative_prompt = agent_config.image_config.negative_prompt
        num_inference_steps = agent_config.image_config.num_inference_steps
        model_architecture = agent_config.image_config.image_model_architecture
        api_path = agent_config.image_config.image_api_path
        
        if any(euler_model in agent_config.image_config.image_model for euler_model in euler_a_models):
            scheluder_config = "Euler A"
            num_inference_steps = "30"
            negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

        if agent_config.image_config.image_model == "SG161222/Realistic_Vision_V6.0_B1_noVAE":
            agent_config.image_config.image_api_path = "fal-ai/realistic-vision"
            
        image_size = self.format_image_size(agent_config.image_config.image_size)
                

        # Prepare job details based on agent config
        job_details = {
            "prompt": prompt,
            "model_name": agent_config.image_config.image_model,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": agent_config.image_config.guidance_scale,
            "image_size": image_size,
            "scheduler": scheluder_config,
            "clip_skip": agent_config.image_config.clip_skip,
            "enable_safety_checker": agent_config.image_config.enable_safety_checker,
            "model_architecture":agent_config.image_config.image_model_architecture,
            "image_format": agent_config.image_config.image_format,
            "prompt_weighting": True,
            "num_images": 1,
        }
        
        formatted_job_details = json.dumps(job_details, indent=4)
        print(f"Submitting job with details: {formatted_job_details}")
        # Add optional loras if configured
        if agent_config.image_config.loras:
            job_details["loras"] = agent_config.image_config.loras
        
        try:
            response = requests.post(
                f"{self.base_url}/{agent_config.image_config.image_api_path}",
                headers=headers,
                json=job_details
            )
            response.raise_for_status()
            return response.json().get("status_url")

        except requests.exceptions.RequestException as e:
            print(f"Error submitting job: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding response: {str(e)}")
            return None

    def _check_status_and_download_image(self, status_url: str) -> Optional[str]:
        """Check job status and return image URL when complete."""
        headers = {"Authorization": f"Key {self.api_key}"}
        max_retries = 250
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(status_url, headers=headers)
                response.raise_for_status()
                status_data = response.json()

                status = status_data.get("status")
                if status == "COMPLETED":
                    result_response = requests.get(
                        status_data.get("response_url"),
                        headers=headers
                    )
                    result_response.raise_for_status()
                    return result_response.json()["images"][0]["url"]

                elif status == "FAILED":
                    print("Image generation task failed")
                    return None

                # Still processing, wait and retry
                time.sleep(0.4)
                retry_count += 1

            except requests.exceptions.RequestException as e:
                print(f"Error checking status: {str(e)}")
                return None
            except json.JSONDecodeError as e:
                print(f"Error decoding status response: {str(e)}")
                return None

        print("Timeout waiting for image generation")
        return None

    def check_for_image_request(self,messages: List[dict], agent_config: AgentConfig) -> tuple[bool, str, str]:
        """check for Image request"""
    
        local_messages = messages.copy()
        prompt = textwrap.dedent(f"""\
        ## Instruction
        Analyze if the conversation warrants visual generation based on these criteria:
        1. Direct Visual Elements:
           - Physical actions (showing, revealing, exposing)
           - Gestures and body language expressing visual cues
           - Changes in posture or stance
           - Clothing descriptions or adjustments
           - Facial expressions and emotional displays
        2. Character Presentation:
           - Flirtatious or seductive behavior
           - Self-presentation moments
           - Dramatic or theatrical actions
           - Emotional expressions through body language
           - Physical interactions or movements
        3. Scene-Setting Elements:
           - Detailed environmental descriptions
           - Lighting changes or atmospheric details
           - Character positioning or movement in space
        4. Traditional Image Triggers:
           - Direct requests ("show me", "send a picture")
           - Implied requests ("what do I look like?")
           - Photo-taking actions ("takes a selfie")
        Review the last exchange carefully:

        If the character is performing actions that create a strong visual scene, especially involving movement, poses, or emotional expression, respond with TRUE.
        Provide response as:
        {{
            "result": True/False,
            "reasoning": "Brief explanation focusing on visual elements",
            "visual_type": "action/request/implied/emotional",
            "visual_focus": "character/scene/interaction"
        }}""").rstrip()
        #print(prompt)
        local_messages.append({"role": "user", "content": prompt})    
    
        reasoning_response = ""
        for token in self.llm_handler.generate(local_messages,agent_config,
                                               temperature=0.2,
                                               model=agent_config.llm_config.reasoning_model,
                                               provider=agent_config.llm_config.reasoning_provider,
                                               stop_words=[',\n']
                                              ):
            reasoning_response += token
        #print(f"Reasoning response: {reasoning_response}")
        if "true" in reasoning_response.lower():
            preallocated_name, public_url = self.file_service.generate_preallocated_url(agent_config, "images")
            return True, preallocated_name, public_url
        return False, "", ""
    
    def request_image_generation(self,messages: List[dict],agent_config: AgentConfig,preallocated_image_name:str = ""):
        """request image generation"""
    
        def get_last_ten_messages_with_system(self, messages: List[dict]) -> List[dict]:
            # Preserve the first system message
            first_system_message = next((msg for msg in messages if msg.get('role') == 'system'), None)
    
            # Filter out only non-system messages from the list
            non_system_messages = [msg for msg in messages if msg.get('role') != 'system']
            
            # Take the last 10 messages
            last_ten_messages = non_system_messages[1:11]
    
            return last_ten_messages
    
        local_messages = messages.copy()
        prompt_prefix = ""
        prompt_suffix = ",masterpiece ,realistic,skin texture,ultra detailed,highres, RAW,8k,selfie, self shot,depth of field "
        if any(euler_model in agent_config.image_config.image_model for euler_model in euler_a_models):
            negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
            prompt_prefix = "masterpiece, best quality, very aesthetic, absurdres,"
            prompt_suffix = ""
            
            
        prompt = textwrap.dedent(
        f"""\
        ## Instruction
        Analyze the conversation context and generate precise image description keywords from last character message.
        
        Focus points:
        1. Character Identity: {agent_config.character.name}
        2. Base Appearance: {agent_config.character.appearance}
        3. Current Context: {agent_config.character.name}'s latest message and conversation flow
        
        Format response as a JSON object with the following categories:
        {{
            "ImageDescriptionKeywords": [
                "Subject:detailed physical traits", // e.g. "Subject:hair,eye color,facial features,... ,body features,.."
                "Expression:emotion", // e.g. "Expression:facial expression"
                "Attire:clothing", // e.g. "Attire:detailed clothes,coloring,accessories"
                "Pose:action", // e.g. "Pose:posture,movement,action"
                "Setting:environment", // e.g. "Setting:indoor,outdoor""
                "Lighting:condition", // e.g. "Lighting:style of lighting""
                "Atmosphere:mood", // e.g. "Atmosphere:emotional atmosphere of image"
                "Style:style execution", // e.g. "Style: {prompt_prefix}{prompt_suffix}"
            ]
        }}
        Guidelines:
        - Use specific, descriptive keywords
        - Maintain consistency with character's established traits
        - Ensure keywords flow naturally together
        - Focus on visual elements that can be rendered
        
        Return only the JSON object.""").rstrip()
        #print(prompt)
        image_description_response = ""
        local_messages.append({"role": "user", "content": prompt})

        for token in self.llm_handler.generate(local_messages,
                                               agent_config,
                                               temperature=0.4,
                                               model=agent_config.llm_config.reasoning_model,
                                               provider=agent_config.llm_config.reasoning_provider):
            image_description_response += token
    
        image_prompt = self.parse_image_description(image_description_response.replace("```json","").replace("```","").replace("_"," ").replace("close-up","").replace("closeup","").strip())
        #print(image_prompt)    
        image_url = self.generate_image(image_prompt,agent_config,preallocated_image_name=preallocated_image_name)
        #print(image_url)
        if image_url:
            return f"![{image_prompt}]({image_url})"
        return None
    
    def parse_image_description(self,image_description_response:str):
    
        def remove_duplicate_words(text):
            # Split the text into words
            words = text.split()
            # Use a dictionary to preserve order and remove duplicates
            unique_words = dict.fromkeys(words)
            # Join the unique words back into a string
            return " ".join(unique_words)
    
        try:
    
            image_description_data = json.loads(image_description_response.strip())
            
            if "ImageDescriptionKeywords" in image_description_data:
                # Extract the values and remove keys if present
                image_description_keywords = image_description_data["ImageDescriptionKeywords"]
                cleaned_keywords = [
                    keyword.split(":", 1)[1].strip() if ":" in keyword else keyword
                    for keyword in image_description_keywords
                ]
                image_description = ", ".join(cleaned_keywords)

    
                image_description = remove_duplicate_words(image_description)
                return image_description
        except json.JSONDecodeError:
            print(f"Failed to parse image description JSON. {image_description_response}")
            return image_description_response.strip()
            #return None
        except Exception:
            print(f"Malformed response. {image_description_response}")
            return image_description_response.strip()
        return image_description_response

    def format_messages_to_display(self,local_messages: list) -> str:
        formatted_messages = []
        for msg in local_messages:
            if msg['role'] != 'system':  # Skip system messages
                role = 'User' if msg['role'] == 'user' else 'Assistant'
                formatted_messages.append(f"{role}: {msg['content']}")
    
        return "\n".join(formatted_messages)
