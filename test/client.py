import httpx
import asyncio
import os
from character_loader import load_character_from_yaml, Character

API_URL = "https://mikpoik--modal-agent-fastapi-app-dev.modal.run/prompt"
AUTH_TOKEN = os.environ["API_KEY"]
WORKSPACE = "default82"
TIMEOUT_SETTINGS = httpx.Timeout(
    timeout=300.0,  # 5 minutes total timeout
    connect=60.0,   # connection timeout
    read=300.0,     # read timeout
    write=60.0      # write timeout
)

async def send_message(client: httpx.AsyncClient, message: str, workspace_id: str = WORKSPACE):
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": message,
        "workspace_id": workspace_id,
        "context_id": "default2",
        "agent_id": "default2",
    }

    try:
        async with client.stream('POST', API_URL, headers=headers, json=data, timeout=TIMEOUT_SETTINGS) as response:
            async for line in response.aiter_lines():
                if line:
                    print(line)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

async def init_character(client: httpx.AsyncClient, character_yaml: str):
    """Initialize the agent with a character from YAML"""
    character = load_character_from_yaml(character_yaml)
    if not character:
        print("Failed to load character, using defaults")
        return

    # Initialize agent with loaded character
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    agent_config = {
        "context_id": "default2",
        "agent_id": "default2",
        "workspace_id": WORKSPACE,
        "character": character.to_dict()
    }

    init_url = "https://mikpoik--modal-agent-fastapi-app-dev.modal.run/init_agent"
    response = await client.post(init_url, headers=headers, json=agent_config, timeout=TIMEOUT_SETTINGS)
    if response.status_code == 200:
        print(f"Initialized agent with character: {character.name}")
    else:
        print("Failed to initialize agent")

async def main():
    async with httpx.AsyncClient() as client:
        # Initialize with character from YAML
        await init_character(client, "test/characters/velvet.yaml")

        # Send initial greeting
        await send_message(client, """Narrate Velvet's scene for me in this chat.
Narrative Guidelines:
• Format spoken dialogue in quotation marks in first person.
• Show actions between asterisks in third person.
• Express internal thoughts in parentheses.
• Include emotional states and reactions.
• Describe environment and atmosphere when relevant.
• Use third-person narration.

Companion Narration Instructions:
• Narrate character's perspective in third person.
• Keep responses concise and engaging
• Balance character dialogue, action, and internal monologue
• Use environmental details to enhance immersion
• Maintain consistent character voice and perspective

Start Velvet's response with a brief description of the setting or an action.""")

        while True:
            prompt = input("Enter your prompt ('exit' to quit): ")
            if prompt.lower() == 'exit':
                break
            await send_message(client, prompt)

if __name__ == "__main__":
    asyncio.run(main())