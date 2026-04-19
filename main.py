from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import ask
import os

load_dotenv()

app = FastAPI(title="Legal AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    result = ask(req.question)
    return AnswerResponse(answer=result["answer"], sources=result["sources"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Railway PORT ni o'zi beradi
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)