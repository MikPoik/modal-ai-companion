# src/services/file_service.py
import json
import pathlib
import requests
import shortuuid
from typing import Any, Optional
from src.models.schemas import AgentConfig

class FileService:
    def __init__(self, base_path: str = "/data"):
        import shortuuid
        self.base_path = pathlib.Path(base_path)
        self.image_base_path = pathlib.Path("/cloud-images")
        self.public_url_base = "https://storage.googleapis.com/coqui-samples"

    def get_path(self, workspace_id: str, filename: str) -> pathlib.Path:
        path = pathlib.Path(f"{self.base_path}/{workspace_id}/{filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, data: Any, workspace_id: str, filename: str):
        path = self.get_path(workspace_id, filename)
        print(f"Saving JSON to {path}")
        with path.open('w') as f:
            json.dump(data, f)

    def load_json(self, workspace_id: str, filename: str) -> Optional[dict]:
        path = self.get_path(workspace_id, filename)
        print(f"Loading JSON from {path}")
        if path.exists():
            with path.open('r') as f:
                return json.load(f)
        return None

    def save_image_to_bucket(self, image_url: str, agent_config: AgentConfig, sub_folder: str = "") -> str:
        """Save image to cloud bucket and return public URL."""
        filename = f"{shortuuid.uuid()}.png"

        if sub_folder and not sub_folder.endswith('/'):
            sub_folder += "/"

        image_path = pathlib.Path(f"{self.image_base_path}/{sub_folder}{agent_config.workspace_id}/{filename}")
        public_url = f"{self.public_url_base}/{sub_folder}{agent_config.workspace_id}/{filename}"

        # Create directory if needed
        image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving image to bucket {image_path}")
        # Download and save image
        response = requests.get(image_url)
        response.raise_for_status()
        image_path.write_bytes(response.content)

        return public_url

    
    def delete_file(self, workspace_id: str, filename: str) -> bool:
        """Delete a file from the workspace."""
        try:
            path = self.get_path(workspace_id, filename)
            if path.exists():
                print("Deleting file:", path)
                path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False