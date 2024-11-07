from typing import Optional
import modal
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os

from starlette.responses import JSONResponse

from src.agent.modal_agent import ModalAgent
from src.models.schemas import app, Generation, AgentConfig, LLMConfig, volume

web_app = FastAPI()
modal_agent = ModalAgent()

image = modal.Image.debian_slim(python_version="3.10").pip_install(
   "pydantic==2.6.4",
   "fastapi==0.114.0",
    "requests",
   "shortuuid",
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

@web_app.post("/generate_avatar")
async def generate_avatar(
    generation:Generation,credentials: str = Depends(authenticate)):
    try:
        print(generation)
        agent_config = AgentConfig(
                prompt = generation.prompt,
                context_id=generation.context_id,
                agent_id=generation.agent_id,
                workspace_id=generation.workspace_id,
            )
        avatar_url = modal_agent.generate_avatar.remote(generation.prompt,agent_config)
        print("Avatar URL: ",avatar_url)
        return avatar_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/init_agent")
async def init_agent(agent_config: AgentConfig, credentials: str = Depends(authenticate)):
    print("Initializing Agent")
    #print("agent_config", agent_config)
    return modal_agent.get_or_create_agent_config.remote(agent_config, update_config=True)
    
@web_app.post("/get_chat_history")
async def get_chat_history(agent_config: AgentConfig, credentials: str = Depends(authenticate)):
    print("Getting Chat History")
    return modal_agent.get_chat_history.remote(agent_config)
                           
@web_app.post("/delete_chat_history")
async def delete_chat_history(agent_config: AgentConfig,credentials: str = Depends(authenticate)):
    print("Deleting chat history")
    return modal_agent.delete_chat_history.remote(agent_config)

@web_app.post("/delete_workspace")
async def delete_workspace(agent_config: AgentConfig,credentials: str = Depends(authenticate)):
    print("Deleting workspace")
    return modal_agent.delete_workspace.remote(agent_config)

@web_app.post("/delete_message_pairs")
async def delete_message_pairs(agent_config: AgentConfig,credentials: str = Depends(authenticate)):
    print("Deleting message pairs")
    
    try:
        result = modal_agent.delete_message_pairs.remote(agent_config, **agent_config.kwargs)
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@web_app.post("/prompt")
async def prompt(input_data: Generation, token: str = Depends(authenticate)):

    print(f"POST /generate")
    def stream_generator():
        try:
            for token in modal_agent.run.remote_gen(input_data, input_data.agent_config or None):
                yield token
        except Exception as e:
            yield f"\ndata: Error: {str(e)}\n\n"
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