from typing import List, Dict
import re

class CitationFormatter:
    def __init__(self, style: str = "apa"):
        self.style = style

    def format_url(self, url: str, title: str = "", accessed: str = "") -> str:
        if self.style == "apa":
            return f"{title}. Retrieved from {url}"
        return f"[{title}]({url})"

    def deduplicate(self, citations: List[str]) -> List[str]:
        seen = set()
        result = []
        for c in citations:
            url = re.search(r"https?://[^\s]+", c)
            key = url.group() if url else c
            if key not in seen:
                seen.add(key)
                result.append(c)
        return result

    def format_all(self, citations: List[str]) -> str:
        unique = self.deduplicate(citations)
        return "\n".join(f"{i+1}. {c}" for i, c in enumerate(unique))
