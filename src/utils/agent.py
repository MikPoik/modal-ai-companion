import modal
import pathlib
from src.common import AgentConfig, Generation, LLMConfig, volume,app
from src.utils.llm_api import LLMapi
from src.utils.image_api import ImageAPI
from src.utils.embedding_api import EmbeddingAPI

agent_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "openai==1.47",
        "pydantic==2.6.4",
        "requests",
        "shortuuid",
        "annoy"
    )
)



with agent_image.imports():
    import json
    import os
    from typing import List, Optional
    import shortuuid
    import requests
    import pickle
    from annoy import AnnoyIndex
    import re
    
gcp_hmac_secret = modal.Secret.from_name(
    "gcp-secret",
    required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"]
)

@app.cls(
    timeout=60 * 2,
    container_idle_timeout=60 * 15,
    allow_concurrent_inputs=10,
    image=agent_image,
    secrets=[modal.Secret.from_name("gcp-secret"),modal.Secret.from_name("deep-infra-api-key"),modal.Secret.from_name("falai-apikey")],
    volumes={"/data": volume,
            "/bucket-mount": modal.CloudBucketMount(
                bucket_name="modal-agent-chat-test",
                bucket_endpoint_url="https://storage.googleapis.com",
                secret=gcp_hmac_secret
            ),
            "/cloud-images": modal.CloudBucketMount(
                bucket_name="coqui-samples",
                bucket_endpoint_url="https://storage.googleapis.com",
                secret=gcp_hmac_secret
            )}
)
class ModalAgent:
    
    def __init__(self):
        print("Initializing Agent")
        self.embedding_api = None

    
    @modal.method(is_generator=True)
    def run(self, prompt: Generation, agent_config: Optional[AgentConfig] = None, update_config: bool = False):
        print("Get or create Agent config")
        print("update_config", update_config)
        if agent_config is None:
            print("No agent config found, creating a new one")
            agent_config = AgentConfig(
                context_id=prompt.context_id,
                agent_id=prompt.agent_id,
                workspace_id=prompt.workspace_id,
                model=prompt.model,
                provider=prompt.provider,
                llm_config=LLMConfig(system_prompt=prompt.system_prompt)
            )
        agent_config = self.get_or_create_agent_config.local(agent_config, update_config)
        print(f"Agent config: {agent_config.model_dump()}")

        if self.embedding_api is None:
            self.embedding_api = EmbeddingAPI(agent_config)
        # Perform vector search
        similar_chunks = self.load_index_and_search(prompt.prompt, agent_config)
        if similar_chunks:
            relevant_backstory = "\n".join(similar_chunks)
            print(f"Relevant backstory: {relevant_backstory}")

        print(f"Generating with prompt: {prompt}")
        chat_history = self.load_chat_history(agent_config)
        if not any(entry.get('role') == 'system' for entry in chat_history):
            chat_history.append({"role": "system", "content": f"{agent_config.llm_config.system_prompt}"})
        chat_history.append({"role": "user", "content": prompt.prompt})
        self.save_chat_history(agent_config, chat_history)

        #image_api = ImageAPI(agent_config)
        #image_url = image_api.generate("Beautiful woman")

        #if image_url:
        #    self.save_image_to_bucket(image_url[0], agent_config)
        
        model = agent_config.model if agent_config.model is not None else "meta-llama/Meta-Llama-3-8B-Instruct"
        response_content = ""
        llm_api = LLMapi(agent_config)
        for token in llm_api.generate(chat_history, model):
            response_content += token
            yield token
        chat_history.append({"role": "assistant", "content": response_content})
        self.save_chat_history(agent_config, chat_history)
    
    @modal.method()
    def get_or_create_agent_config(self, agent_config: AgentConfig, update_config: bool = False) -> AgentConfig:
        cache_key = f"{agent_config.workspace_id}-{agent_config.agent_id}"
        agent_config_cache = modal.Dict.from_name(cache_key, create_if_missing=True)
        if cache_key in agent_config_cache and not update_config:
            print(f"Returning cached agent config for {cache_key}")
            return AgentConfig(**agent_config_cache[cache_key])
        config_path = pathlib.Path(f"/data/{agent_config.workspace_id}/{agent_config.agent_id}_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if config_path.exists() and not update_config:
                with config_path.open("r") as f:
                    print("Loading existing agent config from volume")
                    config_data = json.load(f)
                    return AgentConfig(**config_data)
            else:
                print("Using provided agent config")
                # Save to file
                with config_path.open("w") as f:
                    json.dump(agent_config.model_dump(), f, indent=2)
                volume.commit()
                # Save to cache
                agent_config_cache[cache_key] = agent_config.model_dump()
                if agent_config.backstory:
                    if self.embedding_api is None:
                        print("Initializing embedding_api")
                        self.embedding_api = EmbeddingAPI(agent_config)
                    self.create_and_save_index(agent_config.backstory, agent_config, update_config)
                return agent_config
        except Exception as e:
            print(f"Error in get_or_create_agent_config: {str(e)}")
            raise

    def load_chat_history(self, agent_config: AgentConfig) -> list:
        chat_history_path = pathlib.Path(f"/bucket-mount/{agent_config.workspace_id}/{agent_config.agent_id}_{agent_config.context_id}_chat_history.json")
        chat_history_path.parent.mkdir(parents=True, exist_ok=True)
        print("Loading chat history from path", chat_history_path)
        if chat_history_path.exists():
            with chat_history_path.open("r") as f:
                return json.load(f)
        return []

    def save_chat_history(self, agent_config: AgentConfig, chat_history: list):
        chat_history_path = pathlib.Path(f"/bucket-mount/{agent_config.workspace_id}/{agent_config.agent_id}_{agent_config.context_id}_chat_history.json")
        print("Saving chat history to path", chat_history_path)
        with chat_history_path.open("w") as f:
            json.dump(chat_history, f, indent=2)
            
    def load_index_and_search(self, query: str, agent_config: AgentConfig, n: int = 2) -> List[str]:
        index_path = pathlib.Path(f"/data/{agent_config.workspace_id}/{agent_config.agent_id}_index.ann")
        chunks_path = pathlib.Path(f"/data/{agent_config.workspace_id}/{agent_config.agent_id}_chunks.pkl")
        print(f"Loading index for {agent_config.agent_id}")

        if not index_path.exists() or not chunks_path.exists():
            print("Index or chunks not found")
            return []

        with chunks_path.open("rb") as f:
            chunks = pickle.load(f)
        
        search_k = -1
        query_embedding = self.embedding_api.generate_embedding(query)
        index = AnnoyIndex(len(query_embedding), 'angular')
        index.load(str(index_path))
        print(f"Searching with {query}")
        search_k=search_k
        similar_ids, distances = index.get_nns_by_vector(query_embedding, n, search_k=search_k,include_distances=True)
        return [chunks[i] for i in similar_ids]
        
    def create_and_save_index(self, backstory: str, agent_config: AgentConfig, update_config: bool):
        index_path = pathlib.Path(f"/data/{agent_config.workspace_id}/{agent_config.agent_id}_index.ann")
        chunks_path = pathlib.Path(f"/data/{agent_config.workspace_id}/{agent_config.agent_id}_chunks.pkl")

        # Check if index already exists and update_config is False
        if index_path.exists() and chunks_path.exists() and not update_config:
            print("Using existing index (update_config is False)")
            return

        print("Creating new index for backstory")
        try:
            embedded_chunks = self.embedding_api.embed_long_text(backstory)
            if not embedded_chunks:
                print("No embedded chunks generated")
                return

            vector_length = len(embedded_chunks[0]["embedding"])
            index = AnnoyIndex(vector_length, 'angular')

            for i, chunk in enumerate(embedded_chunks):
                index.add_item(i, chunk["embedding"])

            index.build(3)  # 10 trees
            index.save(str(index_path))

            # Save the backstory chunks separately
            with chunks_path.open("wb") as f:
                pickle.dump([chunk["chunk"] for chunk in embedded_chunks], f)

            print("Index created and saved successfully")
        except Exception as e:
            print(f"Error in create_and_save_index: {str(e)}")
            raise
    def save_image_to_bucket(self, image_url: str, agent_config: AgentConfig):
        # Generate a unique filename
        filename = f"{shortuuid.uuid()}.png"
        image_path = pathlib.Path(f"/cloud-images/{agent_config.workspace_id}/{filename}")
        print(f"Downloading image from {image_url}")
        print(f"Saving image to path: {image_path}")
        # Ensure the directory exists
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        # Write the image data to the file
        image_path.write_bytes(response.content)
        print(f"Image saved successfully to {image_path}")
        return str(image_path)
    
@app.local_entrypoint()
async def main():
    # Create a Generation instance
    generation = Generation(
        prompt="Do you like Pizza?",
        system_prompt ="You are a helpful assistant.",
        provider="deepinfra",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        workspace_id="test-1",
        context_id="test-1",
        agent_id="test-1",
    )

    # Create ModalAgent instance
    model = ModalAgent()

    # Call the run method with the serialized dictionary
    for token in model.run.remote_gen(generation):
        print(token, end="")