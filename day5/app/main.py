import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .rag import Index
from .tracing import TRACE_FILE

load_dotenv()

state = {}


@asynccontextmanager
async def lifespan(app):
    index = Index()
    state["agent"] = Agent(index)
    state["chunks"] = len(index.chunks)
    yield


app = FastAPI(title="AquaPure Doc Q&A Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    tools_used: list[str]
    steps: int
    trace_id: str
    cost_usd: float
    latency_s: float


@app.get("/health")
def health():
    return {"status": "ok", "chunks": state.get("chunks", 0)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return state["agent"].ask(req.question, req.session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    if TRACE_FILE.exists():
        for line in TRACE_FILE.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            if t["trace_id"] == trace_id:
                return t
    raise HTTPException(status_code=404, detail="trace not found")
