from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid
from ..graph.research_graph import build_research_graph

app = FastAPI(title="Autonomous Multi-Agent Research System", version="1.0.0")
graph = build_research_graph()


class ResearchRequest(BaseModel):
    query: str
    depth: str = "deep"
    output_format: str = "markdown"


class ResearchResponse(BaseModel):
    session_id: str
    report: str
    confidence_score: float
    citations: list
    time_taken_seconds: float


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    import time
    start = time.time()
    session_id = str(uuid.uuid4())

    initial_state = {
        "query": request.query,
        "subtasks": [],
        "search_results": [],
        "retrieved_docs": [],
        "critic_feedback": None,
        "critic_passed": False,
        "retry_count": 0,
        "synthesized_findings": "",
        "final_report": "",
        "citations": [],
        "confidence_score": 0.0,
        "current_step": "plan",
        "session_id": session_id
    }

    config = {"configurable": {"thread_id": session_id}}
    result = await asyncio.to_thread(graph.invoke, initial_state, config)

    return ResearchResponse(
        session_id=session_id,
        report=result["final_report"],
        confidence_score=result["confidence_score"],
        citations=result["citations"],
        time_taken_seconds=round(time.time() - start, 2)
    )


@app.get("/health")
def health():
    return {"status": "healthy", "service": "autonomous-research-agent"}
