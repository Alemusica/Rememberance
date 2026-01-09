"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ARXIV INGESTION                                      ║
║                                                                              ║
║   Fetch papers from ArXiv API.                                               ║
║   Basato su Mirror7/knowledge_ingest_mirror7.py ma adattato per LLM-Evo.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import ssl
import time
import logging
from typing import List, Dict, Set
from dataclasses import dataclass

from ..surrealdb import SurrealDBClient, Paper, get_db_client

logger = logging.getLogger(__name__)

# SSL context for compatibility
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

ARXIV_API = "http://export.arxiv.org/api/query"

# Query sets for LLM-Evolution domain
QUERY_SETS = {
    "llm_evolution": {
        "description": "LLM-guided evolutionary computation",
        "queries": [
            "large language model evolutionary algorithm",
            "LLM optimization evolutionary",
            "neural network guided evolution",
            "foundation model evolutionary computation",
            "prompt optimization evolutionary algorithm",
            "language model crossover mutation",
        ],
        "tags": ["llm", "evolutionary", "optimization"],
    },
    
    "neuroevolution": {
        "description": "Neural architecture and weight evolution",
        "queries": [
            "neuroevolution neural architecture search",
            "evolutionary neural network",
            "genetic algorithm neural network",
            "NEAT neuroevolution",
        ],
        "tags": ["neuroevolution", "nas", "neural"],
    },
    
    "multi_objective": {
        "description": "Multi-objective optimization",
        "queries": [
            "NSGA multi objective optimization",
            "pareto evolutionary algorithm",
            "multi-objective genetic algorithm",
        ],
        "tags": ["multi-objective", "pareto", "nsga"],
    },
    
    "curriculum_learning": {
        "description": "Curriculum learning for evolution",
        "queries": [
            "curriculum learning evolutionary",
            "adaptive difficulty evolution",
            "staged complexity optimization",
        ],
        "tags": ["curriculum", "adaptive", "staged"],
    },
}


def fetch_arxiv(query: str, max_results: int = 30) -> List[Dict]:
    """Fetch papers from ArXiv API."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    
    # Rate limit: 1 request per 3 seconds
    time.sleep(3)
    
    try:
        with urllib.request.urlopen(url, timeout=30, context=ssl_context) as resp:
            xml_data = resp.read().decode()
        
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = {
                'id': entry.find('atom:id', ns).text.split('/')[-1] if entry.find('atom:id', ns) is not None else '',
                'title': entry.find('atom:title', ns).text.strip().replace('\n', ' ') if entry.find('atom:title', ns) is not None else '',
                'summary': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else '',
                'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                'authors': [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None],
                'categories': [c.get('term') for c in entry.findall('atom:category', ns)],
                'link': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
            }
            papers.append(paper)
        
        return papers
        
    except Exception as e:
        logger.error(f"ArXiv API error: {e}")
        return []


async def ingest_arxiv_papers(
    query_set: str = "llm_evolution",
    max_per_query: int = 20,
    client: SurrealDBClient = None,
) -> int:
    """
    Ingest ArXiv papers for a query set.
    
    Args:
        query_set: Key from QUERY_SETS
        max_per_query: Max papers per query
        client: SurrealDB client
    
    Returns:
        Number of papers ingested
    """
    client = client or get_db_client()
    
    if query_set not in QUERY_SETS:
        logger.error(f"Unknown query set: {query_set}")
        return 0
    
    config = QUERY_SETS[query_set]
    tags = config["tags"]
    
    logger.info(f"Ingesting ArXiv papers for: {config['description']}")
    
    total_ingested = 0
    
    for query in config["queries"]:
        logger.info(f"  📄 Query: {query}")
        
        papers = fetch_arxiv(query, max_results=max_per_query + 5)
        
        for p in papers[:max_per_query]:
            arxiv_id = p['id'].replace('.', '_').replace('/', '_')
            
            # Extract year from published date
            year = None
            if p.get('published'):
                try:
                    year = int(p['published'][:4])
                except (ValueError, IndexError):
                    pass
            
            paper = Paper(
                id=f"arxiv_{arxiv_id}",
                title=p['title'],
                authors=p['authors'][:5],  # Limit authors
                year=year,
                abstract=p['summary'][:5000],
                source="arxiv",
                tags=list(set(tags + p.get('categories', [])[:3])),
                url=p['link'],
            )
            
            success = await client.upsert_paper(paper)
            if success:
                total_ingested += 1
                logger.info(f"    ✓ {p['title'][:50]}...")
    
    logger.info(f"Total ingested: {total_ingested}")
    return total_ingested


async def ingest_all_query_sets(
    max_per_query: int = 15,
    client: SurrealDBClient = None,
) -> int:
    """Ingest papers from all query sets."""
    total = 0
    for query_set in QUERY_SETS:
        count = await ingest_arxiv_papers(query_set, max_per_query, client)
        total += count
    return total


# CLI
if __name__ == "__main__":
    import sys
    
    async def main():
        client = get_db_client()
        
        if not await client.health_check():
            print("❌ Cannot connect to SurrealDB!")
            print("Start it with:")
            print("  surreal start --log warn --user root --pass root file:~/.config/surrealdb/evolution.db")
            return
        
        print("✅ Connected to SurrealDB")
        
        if len(sys.argv) > 1:
            query_set = sys.argv[1]
            await ingest_arxiv_papers(query_set)
        else:
            await ingest_all_query_sets()
        
        count = await client.count_papers()
        print(f"\n📚 Total papers in database: {count}")
    
    asyncio.run(main())
