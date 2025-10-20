from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_architecture.rag import RAG


app = FastAPI()
rag = RAG(
    embedding_model_name='all-MiniLM-L6-v2',
    language_model_name='gemini-2.5-flash-lite'
)

class UserMessageRequest(BaseModel):
    user_message: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat_endpoint(request: UserMessageRequest):
    user_msg = request.user_message.lower().strip()
    rag_response = rag(user_msg, top_k=3)
    return {"user_message": request.user_message, "response": rag_response}