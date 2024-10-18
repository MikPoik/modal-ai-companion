import httpx
import asyncio
import os

API_URL = "https://mikpoik--modal-agent-fastapi-app-dev.modal.run/prompt"
AUTH_TOKEN = os.environ["API_KEY"]

async def main():
    async with httpx.AsyncClient() as client:
        while True:
            prompt = input("Enter your prompt ('exit' to quit): ")

            if prompt.lower() == 'exit':
                break

            headers = {
                "Authorization": f"Bearer {AUTH_TOKEN}",
                "Content-Type": "application/json"
            }

            data = {
                "prompt": prompt
            }

            try:
                async with client.stream('POST', API_URL, headers=headers, json=data, timeout=60) as response:
                    async for line in response.aiter_lines():
                        if line:
                            print(line)

            except httpx.HTTPStatusError as exc:
                print(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())