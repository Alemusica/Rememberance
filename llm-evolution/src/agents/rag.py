"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              RAG AGENT                                        ║
║                                                                              ║
║   Agente per Retrieval-Augmented Generation.                                 ║
║   NON eredita da BaseAgent (non usa LLM direttamente).                      ║
║   Fa semantic search su SurrealDB e inietta papers nel context.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
from typing import List, Optional

from ..knowledge.surrealdb import SurrealDBClient, Paper, get_db_client
from .base import BaseAgent

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Agente RAG per retrieval di papers e injection nel context.
    
    Non eredita da BaseAgent perché non usa LLM direttamente.
    Si occupa solo di database queries e formatting.
    """
    
    def __init__(self, db_client: SurrealDBClient = None):
        """
        Args:
            db_client: Client SurrealDB (default: singleton)
        """
        self.db = db_client or get_db_client()
        logger.info("RAGAgent initialized")
    
    async def get_relevant_papers(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Paper]:
        """
        Cerca papers rilevanti per la query.
        
        Usa semantic search se embeddings disponibili,
        altrimenti text search.
        
        Args:
            query: Query di ricerca
            limit: Numero massimo di papers da ritornare
        
        Returns:
            Lista di Paper rilevanti
        """
        # Try semantic search if embeddings available
        # Note: Per ora usa text search. Semantic search richiede
        # un embedding model per generare query_embedding.
        # TODO: Aggiungere supporto per embedding generation
        
        papers = await self.db.search_papers(query, limit=limit)
        
        if not papers:
            logger.warning(f"No papers found for query: {query}")
        
        logger.debug(f"Found {len(papers)} papers for query: {query}")
        return papers
    
    async def inject_context(
        self,
        agent: BaseAgent,
        topic: str,
    ) -> None:
        """
        Cerca papers su un topic e li inietta nel context dell'agente.
        
        Args:
            agent: Agente BaseAgent a cui iniettare il context
            topic: Topic per la ricerca papers
        """
        papers = await self.get_relevant_papers(topic, limit=5)
        
        if papers:
            formatted_text = self._format_papers_for_prompt(papers)
            agent.inject_context(formatted_text)
            logger.info(f"Injected {len(papers)} papers into {agent.AGENT_NAME} context")
        else:
            logger.warning(f"No papers found for topic: {topic}")
    
    async def get_domain_knowledge(
        self,
        domain: str,
    ) -> str:
        """
        Ritorna conoscenza di dominio formattata per prompt LLM.
        
        Args:
            domain: Dominio di ricerca (es. "genetic algorithms", "neural networks")
        
        Returns:
            Testo formattato con papers rilevanti
        """
        papers = await self.get_relevant_papers(domain, limit=10)
        
        if not papers:
            return f"## Domain Knowledge: {domain}\n\nNo relevant papers found."
        
        return self._format_papers_for_prompt(papers)
    
    def _format_papers_for_prompt(
        self,
        papers: List[Paper],
    ) -> str:
        """
        Formatta lista di papers per injection in prompt LLM.
        
        Args:
            papers: Lista di Paper da formattare
        
        Returns:
            Testo formattato con papers
        """
        if not papers:
            return ""
        
        lines = ["## Relevant Research\n"]
        
        for i, paper in enumerate(papers, 1):
            # Title and year
            year_str = f" ({paper.year})" if paper.year else ""
            lines.append(f"- **{paper.title}**{year_str}")
            
            # Authors
            if paper.authors:
                authors_str = ", ".join(paper.authors[:3])  # Max 3 authors
                if len(paper.authors) > 3:
                    authors_str += f" et al. ({len(paper.authors)} total)"
                lines.append(f"  Authors: {authors_str}")
            
            # Abstract (truncated)
            if paper.abstract:
                abstract = paper.abstract[:300]  # Max 300 chars
                if len(paper.abstract) > 300:
                    abstract += "..."
                lines.append(f"  Abstract: {abstract}")
            
            # Source and URL
            if paper.url:
                lines.append(f"  URL: {paper.url}")
            elif paper.source:
                lines.append(f"  Source: {paper.source}")
            
            # Tags (if available)
            if paper.tags:
                tags_str = ", ".join(paper.tags[:5])  # Max 5 tags
                lines.append(f"  Tags: {tags_str}")
            
            lines.append("")  # Empty line between papers
        
        return "\n".join(lines)
