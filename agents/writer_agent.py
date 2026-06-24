from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..graph.state import ResearchState


WRITER_SYSTEM = """You are a technical writer. Given synthesized research findings,
produce a well-structured markdown report with:
- Executive Summary
- Key Findings (3-5 bullet points)
- Detailed Analysis
- Conclusions
- References

Be concise, accurate, and professional."""


class WriterAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def run(self, state: ResearchState) -> ResearchState:
        messages = [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content=f"Research Query: {state['query']}\n\nFindings:\n{state['synthesized_findings']}\n\nCitations: {state['citations']}")
        ]
        response = self.llm.invoke(messages)
        state["final_report"] = response.content
        state["current_step"] = "complete"
        return state
