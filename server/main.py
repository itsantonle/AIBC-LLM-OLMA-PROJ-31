from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"message": "Demo with Ollama API"}
# problems with GPU may cause the server to not be ready immediately

async def wait_for_llama_ready(retries=5, delay=3):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/v1/models")
                if response.status_code == 200:
                    return True
        except httpx.RequestError:
            pass
        await asyncio.sleep(delay * (attempt + 1)) 
    return False

@app.post("/generate")
async def generate_response(data: PromptInput):
    if not await wait_for_llama_ready():
        raise HTTPException(status_code=503, detail="LLaMA server not ready after retries")

    messages = [
        {"role": "system", "content": "You are a cat-like chatbot who only speaks cat."},
        {"role": "user", "content": data.prompt},
        {"role": "assistant", "content": "Meow! Mrewo!! (How can I assist you today?)"}
    ]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": "llama3",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 20,
                    "stream": False,
                }
            )

        result = response.json()
        return {"response": result["choices"][0]["message"]["content"].strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
