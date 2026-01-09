"""
Knowledge Ingestion - Fetch papers from various sources
"""

from .arxiv import ingest_arxiv_papers
from .llm4ec import ingest_llm4ec_papers

__all__ = ["ingest_arxiv_papers", "ingest_llm4ec_papers"]
