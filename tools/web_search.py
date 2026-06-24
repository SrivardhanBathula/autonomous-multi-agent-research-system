from typing import List, Dict, Any, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
import logging

logger = logging.getLogger(__name__)


class WebSearchTool:
    def __init__(self, max_results: int = 10):
        self.search = TavilySearchResults(max_results=max_results)

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        try:
            results = self.search.invoke(query)
            return [{"content": r.get("content", ""), "url": r.get("url", ""), "title": r.get("title", "")} for r in results]
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []

    def fetch_page(self, url: str) -> Optional[str]:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return docs[0].page_content[:3000] if docs else None
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
