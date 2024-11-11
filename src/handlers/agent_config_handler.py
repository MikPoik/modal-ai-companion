# src/handlers/agent_config_handler.py
from typing import Optional, Dict,Union
from src.models.schemas import AgentConfig,PromptConfig
from src.handlers.index_handler import IndexHandler
from src.services.file_service import FileService
from src.services.cache_service import CacheService

class AgentConfigHandler:
    def __init__(self):
        self.file_service = FileService('/data')
        self.cache_service = CacheService()
        self.config_cache: Dict[str, AgentConfig] = {}
        self.index_handler = IndexHandler()

    def get_or_create_config(self, agent_config: Union[AgentConfig, PromptConfig], update_config: bool = False) -> Union[AgentConfig, PromptConfig]:
        """
        Get existing config or create new one if it doesn't exist
        """

        if not agent_config:
            print(f"No agent config provided, create defaults")
            if not agent_config.workspace_id:
                print(f"No workspace ID provided, create defaults")
            agent_config = AgentConfig()

        cache_key = f"{agent_config.workspace_id}_{agent_config.agent_id}"

        # Try to get from memory cache first
        if not update_config and cache_key in self.config_cache:
            cached_config = self.config_cache[cache_key]
            if isinstance(agent_config, PromptConfig):
                # Create a new PromptConfig from the cached base config
                base_dict = cached_config.model_dump()
                if 'prompt' in base_dict:
                    del base_dict['prompt']  # Remove prompt if it exists
                return PromptConfig(**base_dict, prompt=agent_config.prompt)
            return cached_config

        # Try to get from file cache
        config_path = f"{agent_config.agent_id}_config.json"
        existing_config = None
        if not update_config:
            existing_config = self.file_service.load_json(
                agent_config.workspace_id,
                config_path
            )
            if existing_config:
                existing_config = AgentConfig(**existing_config)
                self.config_cache[cache_key] = existing_config
                return existing_config
                
        # Create embedding index if background text exists and is long enough
        if (agent_config.character and 
            agent_config.character.backstory and 
            len(agent_config.character.backstory) > 1000):
            print("Creating embedding index for background text")
            success = self.index_handler.create_and_save_index(
                agent_config.character.backstory,
                agent_config,
                update_config
            )
            if not success:
                print("Warning: Failed to create embedding index")
                
        # Save new config
        self.file_service.save_json(
            agent_config.model_dump(),
            agent_config.workspace_id,
            config_path
        )

        # Update memory cache
        self.config_cache[cache_key] = agent_config
        if isinstance(agent_config, PromptConfig):
            base_dict = agent_config.dict()
            if 'prompt' in base_dict:
                del base_dict['prompt']  # Remove prompt from the dictionary
            return PromptConfig(**base_dict, prompt=agent_config.prompt)
        return agent_config

    def update_config(self, agent_config: AgentConfig) -> AgentConfig:
        """
        Update existing configuration
        """
        return self.get_or_create_config(agent_config, update_config=True)

    def get_config(self, workspace_id: str, agent_id: str) -> Optional[AgentConfig]:
        """
        Retrieve configuration by workspace and agent IDs
        """
        cache_key = f"{workspace_id}_{agent_id}"

        # Try memory cache first
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]

        # Try file cache
        config_data = self.file_service.load_json(
            workspace_id,
            f"{agent_id}_config.json"
        )

        if config_data:
            config = AgentConfig(**config_data)
            self.config_cache[cache_key] = config
            return config

        return None

    def delete_config(self, workspace_id: str, agent_id: str) -> bool:
        """
        Delete configuration
        """
        cache_key = f"{workspace_id}_{agent_id}"
        config_path = f"{agent_id}_config.json"

        # Remove from memory cache
        if cache_key in self.config_cache:
            del self.config_cache[cache_key]

        # Remove from file storage
        return self.file_service.delete_file(
            workspace_id,
            config_path
        )

    def clear_cache(self):
        """
        Clear the configuration cache
        """
        self.config_cache.clear()