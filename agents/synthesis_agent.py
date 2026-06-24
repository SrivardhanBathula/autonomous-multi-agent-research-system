from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..graph.state import ResearchState


SYNTHESIS_SYSTEM = """You are a research synthesis expert. Given multiple source documents,
synthesize the key findings into a coherent, comprehensive summary. Extract citations."""


class SynthesisAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def run(self, state: ResearchState) -> ResearchState:
        docs = "\n\n".join(state.get("retrieved_docs", [])[:10])
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=f"Query: {state['query']}\n\nDocuments:\n{docs}\n\nSynthesize key findings.")
        ]
        response = self.llm.invoke(messages)
        state["synthesized_findings"] = response.content
        citations = [r.get("url", "") for r in state.get("search_results", []) if r.get("url")]
        state["citations"] = list(set(citations))[:10]
        state["current_step"] = "write"
        return state
