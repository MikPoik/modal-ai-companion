# src/handlers/chat_handler.py
from typing import List, Dict, Optional
from src.models.schemas import AgentConfig
from src.services.file_service import FileService
from src.handlers.index_handler import IndexHandler

class ChatHandler:
    def __init__(self):
        import shortuuid
        self.file_service = FileService('/bucket-mount')
        self.index_handler = IndexHandler()
        
    def prepare_messages(self, prompt: str, agent_config: AgentConfig) -> List[Dict]:
        """Prepare messages for the LLM including system prompt and relevant background."""
        messages = []
        
        relevant_backstory = None
        # Get relevant background if available
        if (agent_config.character and 
            agent_config.character.background and 
            len(agent_config.character.background) > 1000):

            similar_chunks = self.index_handler.search(prompt, agent_config)
            if similar_chunks:
                relevant_backstory = "\n".join(similar_chunks)
                print("\nRelevant backstory: "+relevant_backstory)
                
        # Get the formatted system prompt
        system_prompt = self._format_system_prompt(agent_config,updated_backstory=relevant_backstory)
        
        messages.append({"role": "system", "content": system_prompt})
        # Get chat history (excluding system messages)
        history = self.get_chat_history(agent_config)  # This returns List[Dict]
        messages.extend([msg for msg in history if msg.get('role') != 'system'])

        # Add the user's prompt
        messages.append({"role": "user", "content": prompt})
        # Filter messages to fit context window
        return self.filter_messages_for_context(
            messages, 
            max_context_size=agent_config.llm_config.context_size
        )
        
    def _format_system_prompt(self, agent_config: AgentConfig,updated_backstory = None) -> Optional[str]:
        """Format system prompt with character details."""
        try:
            return agent_config.llm_config.system_prompt.format(
                char_name=agent_config.character.name,
                char_description=agent_config.character.description,
                char_appearance=agent_config.character.appearance,
                char_personality=agent_config.character.personality,
                char_background=updated_backstory or agent_config.character.background,
                tags=agent_config.character.tags,
                char_seed=agent_config.character.seed_message
            )
        except (AttributeError, KeyError) as e:
            print(f"Error formatting system prompt: {str(e)}")
            return agent_config.llm_config.system_prompt

    def get_chat_history(self, agent_config: AgentConfig) -> List[Dict]:
        """Load chat history from bucket."""
        filepath = f"/bucket-mount/{agent_config.workspace_id}/{agent_config.context_id}_chat.json"
        history = self.file_service.load_json(agent_config.workspace_id,f'{agent_config.context_id}_chat.json') or []
        return self._format_chat_history(history, agent_config)

    def save_chat_history(self, messages: List[Dict], agent_config: AgentConfig):
        import shortuuid
        """Save chat history to bucket."""
        filename = f"{agent_config.context_id}_chat.json"
        filepath = f"/bucket-mount/{agent_config.workspace_id}/{agent_config.context_id}_chat.json"
        self.file_service.save_json(messages,agent_config.workspace_id,filename)

    def _format_chat_history(self, history: List[Dict], agent_config: AgentConfig) -> List[Dict]:
        system_prompt = self._format_system_prompt(agent_config)
        return [{"role": "system", "content": system_prompt}] + history[-agent_config.llm_config.context_size:]

    def filter_messages_for_context(self, messages: List[Dict], max_context_size: int = 4096) -> List[Dict]:
        """
        Filter messages to fit within context size while preserving system message and chronological order.
        """
        def estimate_tokens(message: Dict) -> int:
            """Estimate tokens in a message using 4 chars/token ratio."""
            content = message.get('content', '')
            return len(content) // 4
        if not messages:
            return []
        # Extract system message
        system_message = next((msg for msg in messages if msg.get('role') == 'system'), None)
        filtered_messages = [system_message] if system_message else []
        # Get non-system messages
        conversation = [msg for msg in messages if msg.get('role') != 'system']
        # Calculate current token count starting with system message
        current_tokens = estimate_tokens(system_message) if system_message else 0
        # Add messages from newest to oldest until we hit token limit
        for message in reversed(conversation):
            message_tokens = estimate_tokens(message)
            if current_tokens + message_tokens <= max_context_size:
                filtered_messages.append(message)
                current_tokens += message_tokens
            else:
                break
        # Restore message order: system first, then chronological
        return ([msg for msg in filtered_messages if msg.get('role') == 'system'] + 
                list(reversed([msg for msg in filtered_messages if msg.get('role') != 'system'])))