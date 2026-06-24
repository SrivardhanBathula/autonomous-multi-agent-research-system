# Autonomous Multi-Agent Research System

> Production-grade multi-agent pipeline for autonomous research, synthesis, and report generation using LangGraph and CrewAI.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Key Metrics
| Metric | Value |
|--------|-------|
| Task Completion Rate | 94% |
| Avg Research Time | <45 seconds |
| Agent Types | 5 specialized agents |
| Concurrent Sessions | 50+ via Redis |

## Architecture

```
User Query → Planner Agent → [Research Agent → Critic Agent → Synthesis Agent] → Writer Agent → Report
                                      ↕
                              Web Search (Tavily)
                              Vector Store (FAISS)
                              Redis State
```

## Agents

- **Planner Agent** — Decomposes queries into subtasks, assigns to specialist agents
- **Research Agent** — Web search via Tavily + retrieval from FAISS vector store
- **Critic Agent** — Validates sources, flags hallucinations, requests re-search if needed
- **Synthesis Agent** — Merges findings, resolves contradictions, scores confidence
- **Writer Agent** — Generates structured markdown report with citations

## Tech Stack

- **Orchestration:** LangGraph (stateful multi-agent graphs), CrewAI
- **LLMs:** OpenAI GPT-4o, GPT-4o-mini (cost routing)
- **Search:** Tavily API, FAISS vector store
- **State:** Redis for session persistence and agent memory
- **Serving:** FastAPI with SSE streaming
- **Infra:** Docker, Docker Compose

## Quick Start

```bash
git clone https://github.com/SrivardhanBathula/autonomous-multi-agent-research-system
cd autonomous-multi-agent-research-system
cp .env.example .env  # Add your API keys
docker-compose up --build
```

## API

```bash
POST /research
{
  "query": "Latest developments in LLM evaluation frameworks",
  "depth": "deep",  # shallow | deep | expert
  "output_format": "markdown"
}
```

## Project Structure

```
autonomous-multi-agent-research-system/
├── agents/
│   ├── planner_agent.py
│   ├── research_agent.py
│   ├── critic_agent.py
│   ├── synthesis_agent.py
│   └── writer_agent.py
├── graph/
│   ├── research_graph.py
│   └── state.py
├── tools/
│   ├── web_search.py
│   ├── vector_retriever.py
│   └── citation_formatter.py
├── api/
│   └── main.py
├── notebooks/
│   └── 01_multi_agent_demo.ipynb
├── config/
│   └── config.yaml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Author

**Srivardhan Bathula** — AI/ML Engineer
- Portfolio: [srivardhanbathula.github.io/srivardhanb.github.io](https://srivardhanbathula.github.io/srivardhanb.github.io)
- LinkedIn: [linkedin.com/in/srivardhanb](https://linkedin.com/in/srivardhanb)
