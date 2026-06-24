import json
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..graph.state import ResearchState


PLANNER_SYSTEM_PROMPT = """You are a research planning agent. Given a user query,
decompose it into 3-5 specific research subtasks that can be executed in parallel.
Return a JSON list of subtask strings. Be specific and actionable."""


class PlannerAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def run(self, state: ResearchState) -> ResearchState:
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {state['query']}\nDecompose into research subtasks.")
        ]
        response = self.llm.invoke(messages)
        try:
            subtasks = json.loads(response.content)
        except json.JSONDecodeError:
            subtasks = [state["query"]]

        state["subtasks"] = subtasks
        state["current_step"] = "research"
        return state
