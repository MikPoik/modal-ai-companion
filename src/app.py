from typing import Optional
import modal
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os

from src.utils.agent import ModalAgent
from src.common import app, Generation, AgentConfig, LLMConfig, volume

web_app = FastAPI()
modal_agent = ModalAgent()

image = modal.Image.debian_slim(python_version="3.10").pip_install(
   "openai==1.47",
   "pydantic==2.6.4",
   "fastapi==0.114.0",
   "shortuuid"
)


# Set up the HTTP bearer scheme
http_bearer = HTTPBearer(
    scheme_name="Bearer Token",
    description="Enter your API token",
)


# Authentication dependency
async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    if credentials.credentials != os.environ["auth_token"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
        )
    return credentials.credentials

@web_app.post("/init_agent")
async def init_agent(agent_config: AgentConfig, credentials: str = Depends(authenticate)):
    print("Initializing Agent")
    print("agent_config", agent_config)
    print("Update agent config")
    return modal_agent.get_or_create_agent_config.remote(agent_config, update_config=True)
    
@web_app.post("/prompt")
async def prompt(input_data: Generation, token: str = Depends(authenticate)):
    print(f"Received input: {input_data}")
    # Create AgentConfig from the input data
    agent_config = AgentConfig(
        context_id=input_data.context_id,
        agent_id=input_data.agent_id,
        workspace_id=input_data.workspace_id,
        model=input_data.model,
        provider=input_data.provider,
        update_config=input_data.update_config,  # This is now available from BaseConfig
        llm_config=LLMConfig(system_prompt=input_data.system_prompt)
    )
    print(f"Created AgentConfig: {agent_config}")
    print(f"POST /generate - received generation.prompt={input_data.prompt}")
    def stream_generator():
        try:
            print("Call generation")
            for token in modal_agent.run.remote_gen(input_data, agent_config, input_data.update_config):
                yield token
        except Exception as e:
            yield f"\ndata: Error: {str(e)}\n\n"
        yield "\ndata: [DONE]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
    

@app.function(
    timeout=60 * 2,
    container_idle_timeout=60 * 15,
    allow_concurrent_inputs=10,
    image=image,
    secrets=[modal.Secret.from_name("fast-api-secret")],
    volumes={"/data": volume}
)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    app.deploy("webapp")