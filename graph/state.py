from typing import TypedDict, List, Optional, Dict, Any


class ResearchState(TypedDict):
    query: str
    subtasks: List[str]
    search_results: List[Dict[str, Any]]
    retrieved_docs: List[str]
    critic_feedback: Optional[str]
    critic_passed: bool
    retry_count: int
    synthesized_findings: str
    final_report: str
    citations: List[str]
    confidence_score: float
    current_step: str
    session_id: str
