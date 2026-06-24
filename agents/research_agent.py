import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ..graph.state import ResearchState


class ResearchAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.search_tool = TavilySearchResults(max_results=8)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def _web_search(self, query: str) -> List[Dict[str, Any]]:
        try:
            return self.search_tool.invoke(query)
        except Exception as e:
            return [{"content": f"Search failed: {e}", "url": ""}]

    def _retrieve_from_store(self, query: str, vectorstore: FAISS, k: int = 5) -> List[str]:
        try:
            docs = vectorstore.similarity_search(query, k=k)
            return [d.page_content for d in docs]
        except Exception:
            return []

    def run(self, state: ResearchState) -> ResearchState:
        all_results = []
        all_docs = []
        for subtask in state.get("subtasks", [state["query"]]):
            results = self._web_search(subtask)
            all_results.extend(results)
            all_docs.extend([r.get("content", "") for r in results if r.get("content")])

        state["search_results"] = all_results
        state["retrieved_docs"] = all_docs[:20]
        state["retry_count"] = state.get("retry_count", 0)
        state["current_step"] = "critique"
        return state
