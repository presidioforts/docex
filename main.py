import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")

# Create OpenAI client for v1 API
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to your frontend's URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4"
            messages=[{"role": "user", "content": req.message}],
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        # Optionally log the error here
        raise HTTPException(status_code=500, detail="OpenAI API error: " + str(e))
