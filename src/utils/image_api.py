# image_api.py
from typing import Optional, List, Dict, Any
import os
import json
import time
from src.common import AgentConfig

class ImageAPI:
    def __init__(self, api_config: AgentConfig):
        import requests
        
        self.config = api_config
        self.base_url = "https://queue.fal.run"
        self.api_key = os.environ["FALAI_API_KEY"]  # Assuming AgentConfig has an api_key field

    def submit_job(self, job_details: Dict[str, Any]) -> Optional[str]:
        headers = {
            "Authorization": f'Key {os.environ["FALAI_API_KEY"]}',
            "Content-Type": "application/json"
        }
        data = json.dumps(job_details)
        response = requests.post(f"{self.base_url}/{self.config.image_config.image_api_path}",
                                 headers=headers,
                                 data=data)
        try:
            response_json = response.json()
            status_url = response_json.get("status_url")
            return status_url
        except json.JSONDecodeError as e:
            lprint("JSON decode error:", e)
            return None

    def check_status_and_download_image(self, status_url: str) -> Optional[str]:
        headers = {"Authorization": f'Key {os.environ["FALAI_API_KEY"]}'}
        status = "IN_QUEUE"
        while status not in ("COMPLETED", "FAILED"):
            response = requests.get(status_url, headers=headers)
            response_json = response.json()
            time.sleep(0.4)
            status = response_json.get("status")
            if status == "COMPLETED":
                result_response = requests.get(
                    response_json.get("response_url"), headers=headers)
                return result_response.json()["images"][0]["url"]
            elif status == "FAILED":
                print("Image generation task failed.")
                return None

    def generate(self, prompt: str) -> List[str]:
        print(f"Generating image with prompt: {prompt}")
        
        start_time = time.time()

        arguments = {
            "model_name": self.config.image_config.image_model,
            "prompt": prompt,
            "negative_prompt": self.config.image_config.negative_prompt,
            "loras": self.config.image_config.loras,
            "image_size": self.config.image_config.image_size,
            "num_inference_steps": self.config.image_config.num_inference_steps,
            "guidance_scale": self.config.image_config.guidance_scale,
            "model_architecture":self.config.image_config.image_model_architecture,
            "scheduler": self.config.image_config.scheduler,
            "clip_skip": self.config.image_config.clip_skip,
            "image_format": self.config.image_config.image_format,
            "num_images": 1,
            "enable_safety_checker": self.config.image_config.enable_safety_checker
        }

        status_url = self.submit_job(arguments)
        final_urls = []

        if status_url:
            image_url = self.check_status_and_download_image(status_url)
            if image_url:
                final_urls.append(image_url)
                print(f"Generated image URLs: {final_urls}")


        end_time = time.time() - start_time
        formatted_time = "{:.2f} seconds".format(end_time)
        print(f"Image generation completed in {formatted_time}")
        

        return final_urls

