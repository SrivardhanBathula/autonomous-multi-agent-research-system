from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from ..agents.planner_agent import PlannerAgent
from ..agents.research_agent import ResearchAgent
from ..agents.critic_agent import CriticAgent
from ..agents.synthesis_agent import SynthesisAgent
from ..agents.writer_agent import WriterAgent
from .state import ResearchState
import redis


def build_research_graph(redis_url: str = "redis://localhost:6379") -> StateGraph:
    planner = PlannerAgent()
    researcher = ResearchAgent()
    critic = CriticAgent()
    synthesizer = SynthesisAgent()
    writer = WriterAgent()

    def should_retry(state: ResearchState) -> str:
        if state.get("retry_count", 0) >= 2:
            return "synthesize"
        if state.get("critic_passed", True):
            return "synthesize"
        return "research"

    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", planner.run)
    workflow.add_node("research", researcher.run)
    workflow.add_node("critique", critic.run)
    workflow.add_node("synthesize", synthesizer.run)
    workflow.add_node("write", writer.run)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "research")
    workflow.add_edge("research", "critique")
    workflow.add_conditional_edges("critique", should_retry, {
        "research": "research",
        "synthesize": "synthesize"
    })
    workflow.add_edge("synthesize", "write")
    workflow.add_edge("write", END)

    r = redis.from_url(redis_url)
    memory = RedisSaver(r)
    return workflow.compile(checkpointer=memory)
