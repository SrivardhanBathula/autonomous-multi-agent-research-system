from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..graph.state import ResearchState


CRITIC_SYSTEM = """You are a research critic. Review the search results and determine:
1. Are there enough sources? (need at least 3 credible ones)
2. Is the information relevant to the original query?
3. Are there obvious contradictions that need resolution?
Return JSON: {"passed": true/false, "feedback": "reason", "confidence": 0.0-1.0}"""


class CriticAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.confidence_threshold = 0.75

    def run(self, state: ResearchState) -> ResearchState:
        import json
        docs_preview = "\n".join(state.get("retrieved_docs", [])[:5])
        messages = [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=f"Query: {state['query']}\n\nSources found:\n{docs_preview}")
        ]
        response = self.llm.invoke(messages)
        try:
            result = json.loads(response.content)
            state["critic_passed"] = result.get("passed", True)
            state["critic_feedback"] = result.get("feedback", "")
            state["confidence_score"] = result.get("confidence", 0.8)
        except json.JSONDecodeError:
            state["critic_passed"] = True
            state["confidence_score"] = 0.75
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["current_step"] = "synthesize" if state["critic_passed"] else "research"
        return state
