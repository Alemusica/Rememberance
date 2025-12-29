#!/usr/bin/env python3
"""
Knowledge Base Interface - Access distilled research from SurrealDB.

This module provides semantic search over the research paper database,
enabling context-aware retrieval of relevant literature for:
- Plate optimization (SIMP, topology, ABH)
- Multi-exciter placement (Lu, Shen, Bai papers)
- Vibroacoustic therapy (spine coupling, ear response)
- Modal analysis and FEM

Usage:
    from utils.knowledge_base import KnowledgeBase
    
    kb = KnowledgeBase()
    papers = kb.search("acoustic black hole energy focusing")
    papers = kb.by_domain("abh")
    papers = kb.related_to("peninsula", "resonator")
"""

import json
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from functools import lru_cache


@dataclass
class Paper:
    """Research paper with metadata and relevance info."""
    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    domains: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    section: Optional[str] = None
    doi: Optional[str] = None
    relevance_score: float = 0.0
    
    def __repr__(self):
        year_str = f" ({self.year})" if self.year else ""
        authors_str = f" - {', '.join(self.authors[:2])}" if self.authors else ""
        return f"📄 {self.title}{year_str}{authors_str}"
    
    def summary(self) -> str:
        """Return formatted summary for display."""
        lines = [f"📄 {self.title}"]
        if self.authors:
            lines.append(f"   Authors: {', '.join(self.authors[:3])}")
        if self.year:
            lines.append(f"   Year: {self.year}")
        if self.domains:
            lines.append(f"   Domains: {', '.join(self.domains)}")
        if self.keywords:
            lines.append(f"   Keywords: {', '.join(self.keywords[:5])}")
        if self.abstract:
            abstract_short = self.abstract[:200] + "..." if len(self.abstract) > 200 else self.abstract
            lines.append(f"   Abstract: {abstract_short}")
        return "\n".join(lines)


class KnowledgeBase:
    """
    Interface to SurrealDB research knowledge base.
    
    Provides semantic search over 70+ vibroacoustic research papers
    with domain classification and keyword matching.
    """
    
    # Domain keywords for semantic matching
    DOMAIN_KEYWORDS = {
        "abh": ["acoustic black hole", "abh", "energy focusing", "isolated region", 
                "tapered thickness", "wave trapping", "krylov", "deng"],
        "multi_exciter": ["multi-exciter", "multiple exciters", "exciter placement",
                         "exciter optimization", "dml", "distributed mode", "lu", "shen", "bai"],
        "topology": ["topology optimization", "simp", "solid isotropic", "material penalty",
                    "structural optimization", "density method"],
        "modal": ["modal analysis", "eigenfrequency", "mode shape", "natural frequency",
                 "vibration mode", "fem", "finite element"],
        "vibroacoustic": ["vibroacoustic", "soundboard", "plate vibration", "acoustic radiation",
                         "sound pressure", "loudspeaker", "flat panel"],
        "binaural": ["binaural", "ear", "head", "spatial audio", "stereo", "panning"],
        "spine": ["spine", "spinal", "vertebra", "low frequency", "therapy", "massage"],
        "lutherie": ["guitar", "violin", "lutherie", "instrument", "soundboard", "bridge"],
        "psychoacoustic": ["psychoacoustic", "perception", "loudness", "masking", "hearing"],
        "optimization": ["genetic algorithm", "evolutionary", "optimization", "fitness",
                        "objective function", "pareto", "multi-objective"],
    }
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        namespace: str = "research",
        database: str = "knowledge",
        user: str = "root",
        password: str = "root"
    ):
        self.base_url = f"http://{host}:{port}"
        self.namespace = namespace
        self.database = database
        self.auth = (user, password)
        self._cache: Dict[str, List[Paper]] = {}
    
    def _query(self, sql: str) -> List[Dict]:
        """Execute SQL query against SurrealDB."""
        try:
            response = requests.post(
                f"{self.base_url}/sql",
                headers={
                    "Accept": "application/json",
                    "surreal-ns": self.namespace,
                    "surreal-db": self.database,
                },
                auth=self.auth,
                data=sql,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if result[0].get("status") == "OK":
                    return result[0].get("result", [])
            return []
        except Exception as e:
            print(f"⚠️ SurrealDB query failed: {e}")
            return []
    
    def _to_paper(self, record: Dict) -> Paper:
        """Convert DB record to Paper object."""
        return Paper(
            id=record.get("id", ""),
            title=record.get("title", "Unknown"),
            authors=record.get("authors", []),
            year=record.get("year"),
            abstract=record.get("abstract"),
            domains=record.get("domains", []),
            keywords=record.get("keywords", []),
            section=record.get("section"),
            doi=record.get("doi"),
        )
    
    @lru_cache(maxsize=100)
    def all_papers(self) -> List[Paper]:
        """Get all papers from the database."""
        results = self._query("SELECT * FROM paper ORDER BY year DESC")
        return [self._to_paper(r) for r in results]
    
    def count(self) -> int:
        """Get total number of papers."""
        results = self._query("SELECT count() FROM paper GROUP ALL")
        if results:
            return results[0].get("count", 0)
        return 0
    
    def by_domain(self, domain: str) -> List[Paper]:
        """Get papers by domain classification."""
        # Normalize domain name
        domain = domain.lower().replace("-", "_").replace(" ", "_")
        
        # Query papers containing this domain
        results = self._query(f"SELECT * FROM paper WHERE '{domain}' IN domains ORDER BY year DESC")
        papers = [self._to_paper(r) for r in results]
        
        # Also search in keywords if few results
        if len(papers) < 5:
            keywords = self.DOMAIN_KEYWORDS.get(domain, [domain])
            for kw in keywords:
                extra = self.search(kw, limit=10)
                for p in extra:
                    if p.id not in [x.id for x in papers]:
                        papers.append(p)
        
        return papers
    
    def search(self, query: str, limit: int = 20) -> List[Paper]:
        """
        Semantic search across papers.
        
        Searches in: title, abstract, keywords, authors
        Uses fuzzy matching and domain expansion.
        """
        query_lower = query.lower()
        
        # Get all papers (cached)
        all_papers = self.all_papers()
        
        # Score each paper
        scored = []
        for paper in all_papers:
            score = self._relevance_score(paper, query_lower)
            if score > 0:
                paper.relevance_score = score
                scored.append(paper)
        
        # Sort by relevance
        scored.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored[:limit]
    
    def _relevance_score(self, paper: Paper, query: str) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0
        query_terms = query.split()
        
        # Title match (highest weight)
        title_lower = paper.title.lower()
        for term in query_terms:
            if term in title_lower:
                score += 10.0
        if query in title_lower:
            score += 20.0
        
        # Abstract match
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            for term in query_terms:
                if term in abstract_lower:
                    score += 2.0
            if query in abstract_lower:
                score += 5.0
        
        # Keywords match
        keywords_str = " ".join(paper.keywords).lower()
        for term in query_terms:
            if term in keywords_str:
                score += 5.0
        
        # Domain match
        domains_str = " ".join(paper.domains).lower()
        for term in query_terms:
            if term in domains_str:
                score += 3.0
        
        # Author match
        authors_str = " ".join(paper.authors).lower()
        for term in query_terms:
            if term in authors_str:
                score += 4.0
        
        # Domain keyword expansion
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                if domain in paper.domains:
                    score += 5.0
        
        return score
    
    def related_to(self, *concepts: str) -> List[Paper]:
        """Find papers related to multiple concepts (AND logic)."""
        if not concepts:
            return []
        
        # Start with first concept
        papers = self.search(concepts[0], limit=50)
        
        # Filter by remaining concepts
        for concept in concepts[1:]:
            papers = [
                p for p in papers 
                if self._relevance_score(p, concept.lower()) > 0
            ]
        
        return papers[:20]
    
    def for_context(self, context: str) -> List[Paper]:
        """
        Get papers relevant to a development context.
        
        Examples:
            kb.for_context("peninsula optimization")
            kb.for_context("ear lobe flatness")
            kb.for_context("multi exciter placement")
        """
        # Map contexts to search queries
        context_mappings = {
            "peninsula": ["acoustic black hole", "isolated region", "energy focusing"],
            "ear": ["binaural", "flat panel speaker", "directivity"],
            "spine": ["low frequency vibration", "vibroacoustic therapy", "massage"],
            "exciter": ["exciter placement", "multi-exciter", "dml optimization"],
            "topology": ["topology optimization", "simp", "structural"],
            "modal": ["modal analysis", "eigenfrequency", "fem"],
            "flatness": ["frequency response", "acoustic optimization", "flat panel"],
        }
        
        # Expand context to search terms
        search_terms = []
        for key, terms in context_mappings.items():
            if key in context.lower():
                search_terms.extend(terms)
        
        if not search_terms:
            search_terms = [context]
        
        # Combine results
        all_results = []
        seen_ids = set()
        
        for term in search_terms[:3]:  # Limit to avoid too many queries
            results = self.search(term, limit=10)
            for p in results:
                if p.id not in seen_ids:
                    all_results.append(p)
                    seen_ids.add(p.id)
        
        # Re-score by original context
        for p in all_results:
            p.relevance_score = self._relevance_score(p, context.lower())
        
        all_results.sort(key=lambda p: p.relevance_score, reverse=True)
        return all_results[:15]
    
    def get_insights(self, topic: str) -> Dict[str, any]:
        """
        Get aggregated insights for a topic.
        
        Returns dict with:
            - key_papers: Most relevant papers
            - key_authors: Frequent authors in this area
            - key_concepts: Related keywords
            - summary: Brief topic summary
        """
        papers = self.for_context(topic)
        
        # Aggregate authors
        author_counts = {}
        for p in papers:
            for a in p.authors:
                author_counts[a] = author_counts.get(a, 0) + 1
        key_authors = sorted(author_counts.items(), key=lambda x: -x[1])[:5]
        
        # Aggregate keywords
        keyword_counts = {}
        for p in papers:
            for kw in p.keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        key_concepts = sorted(keyword_counts.items(), key=lambda x: -x[1])[:10]
        
        # Domain distribution
        domain_counts = {}
        for p in papers:
            for d in p.domains:
                domain_counts[d] = domain_counts.get(d, 0) + 1
        
        return {
            "topic": topic,
            "paper_count": len(papers),
            "key_papers": papers[:5],
            "key_authors": [a[0] for a in key_authors],
            "key_concepts": [k[0] for k in key_concepts],
            "domains": domain_counts,
        }
    
    def print_insights(self, topic: str):
        """Print formatted insights for a topic."""
        insights = self.get_insights(topic)
        
        print(f"\n{'═' * 60}")
        print(f"  KNOWLEDGE BASE: {topic.upper()}")
        print(f"{'═' * 60}")
        print(f"\n📚 Found {insights['paper_count']} relevant papers\n")
        
        print("📄 Key Papers:")
        for i, p in enumerate(insights['key_papers'], 1):
            year = f" ({p.year})" if p.year else ""
            print(f"   {i}. {p.title}{year}")
        
        if insights['key_authors']:
            print(f"\n👤 Key Authors: {', '.join(insights['key_authors'])}")
        
        if insights['key_concepts']:
            print(f"\n🏷️  Related Concepts: {', '.join(insights['key_concepts'][:7])}")
        
        if insights['domains']:
            print(f"\n🔬 Domains: {', '.join(insights['domains'].keys())}")
        
        print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_kb_instance: Optional[KnowledgeBase] = None

def get_kb() -> KnowledgeBase:
    """Get singleton KnowledgeBase instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance


def search_papers(query: str, limit: int = 10) -> List[Paper]:
    """Quick search for papers."""
    return get_kb().search(query, limit)


def papers_for(context: str) -> List[Paper]:
    """Get papers relevant to development context."""
    return get_kb().for_context(context)


def insights(topic: str):
    """Print insights for a topic."""
    get_kb().print_insights(topic)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    kb = KnowledgeBase()
    
    print(f"\n🗄️  SurrealDB Knowledge Base")
    print(f"   {kb.count()} papers indexed\n")
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"🔍 Searching: '{query}'\n")
        
        papers = kb.for_context(query)
        for i, p in enumerate(papers[:10], 1):
            print(f"{i}. {p}")
            if p.domains:
                print(f"   Domains: {', '.join(p.domains)}")
            print()
    else:
        # Demo searches
        print("Demo searches:\n")
        
        for topic in ["peninsula", "multi exciter", "ear flatness", "spine coupling"]:
            kb.print_insights(topic)
