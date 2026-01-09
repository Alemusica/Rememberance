"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SURREALDB CLIENT                                     ║
║                                                                              ║
║   Client per SurrealDB knowledge base.                                       ║
║   Gestisce papers, embeddings, e query semantiche.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import base64
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import httpx
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Research paper record."""
    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: str = "arxiv"  # arxiv, pubmed, etc.
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    url: Optional[str] = None
    
    def __repr__(self):
        return f"📄 {self.title} ({self.year})"


class SurrealDBClient:
    """
    Async client per SurrealDB.
    
    Usa HTTP API per compatibilità massima.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8000",
        namespace: str = "evolution",
        database: str = "knowledge",
        username: str = "root",
        password: str = "root",
    ):
        self.url = url.rstrip("/")
        self.namespace = namespace
        self.database = database
        
        # Auth
        auth_str = f"{username}:{password}"
        self.auth_header = base64.b64encode(auth_str.encode()).decode()
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Basic {self.auth_header}",
            "surreal-ns": self.namespace,
            "surreal-db": self.database,
        }
    
    async def query(self, sql: str) -> List[Dict]:
        """Execute SurrealQL query."""
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.url}/sql",
                headers=self._headers(),
                content=sql,
            )
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                result = data[0]
                if result.get("status") == "OK":
                    return result.get("result", [])
                else:
                    logger.error(f"Query error: {result}")
            return []
            
        except httpx.HTTPError as e:
            logger.error(f"SurrealDB error: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if SurrealDB is reachable."""
        try:
            result = await self.query("INFO FOR DB;")
            return len(result) > 0
        except Exception:
            return False
    
    # =========================================================================
    # PAPER OPERATIONS
    # =========================================================================
    
    async def upsert_paper(self, paper: Paper) -> bool:
        """Insert or update paper."""
        # Escape strings
        title = paper.title.replace('"', '\\"').replace('\n', ' ')
        abstract = (paper.abstract or "").replace('"', '\\"').replace('\n', ' ')[:5000]
        
        # Build content
        content = {
            "title": title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": abstract,
            "source": paper.source,
            "tags": paper.tags,
            "url": paper.url,
        }
        
        if paper.embedding:
            content["embedding"] = paper.embedding
        
        sql = f"UPSERT paper:{paper.id} CONTENT {json.dumps(content)};"
        
        result = await self.query(sql)
        return len(result) > 0
    
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID."""
        result = await self.query(f"SELECT * FROM paper:{paper_id};")
        if result:
            return self._to_paper(result[0])
        return None
    
    async def search_papers(
        self,
        query: str,
        limit: int = 20,
        source: str = None,
    ) -> List[Paper]:
        """
        Search papers by title/abstract text.
        """
        query_lower = query.lower()
        
        sql = f"""
        SELECT * FROM paper
        WHERE string::lowercase(title) CONTAINS '{query_lower}'
           OR string::lowercase(abstract) CONTAINS '{query_lower}'
        """
        
        if source:
            sql += f" AND source = '{source}'"
        
        sql += f" LIMIT {limit};"
        
        results = await self.query(sql)
        return [self._to_paper(r) for r in results]
    
    async def papers_by_tags(self, tags: List[str], limit: int = 20) -> List[Paper]:
        """Get papers with specific tags."""
        tag_conditions = " OR ".join([f"'{t}' IN tags" for t in tags])
        sql = f"SELECT * FROM paper WHERE {tag_conditions} LIMIT {limit};"
        
        results = await self.query(sql)
        return [self._to_paper(r) for r in results]
    
    async def all_papers(self, limit: int = 1000) -> List[Paper]:
        """Get all papers."""
        results = await self.query(f"SELECT * FROM paper LIMIT {limit};")
        return [self._to_paper(r) for r in results]
    
    async def count_papers(self) -> int:
        """Count total papers."""
        result = await self.query("SELECT count() FROM paper GROUP ALL;")
        if result:
            return result[0].get("count", 0)
        return 0
    
    async def papers_without_embedding(self, limit: int = 100) -> List[Paper]:
        """Get papers that need embeddings."""
        sql = f"""
        SELECT * FROM paper 
        WHERE embedding = NONE OR array::len(embedding) = 0
        LIMIT {limit};
        """
        results = await self.query(sql)
        return [self._to_paper(r) for r in results]
    
    async def update_embedding(self, paper_id: str, embedding: List[float]) -> bool:
        """Update paper embedding."""
        # Format as SurrealQL array
        emb_str = "[" + ",".join(f"{v}f" for v in embedding) + "]"
        sql = f"UPDATE paper:{paper_id} SET embedding = {emb_str};"
        
        result = await self.query(sql)
        return len(result) > 0
    
    # =========================================================================
    # SEMANTIC SEARCH
    # =========================================================================
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5,
    ) -> List[Paper]:
        """
        Find papers by embedding similarity.
        
        Note: SurrealDB v2 has native vector search.
        For v1, we fetch all and compute in Python.
        """
        import numpy as np
        
        # Get all papers with embeddings
        results = await self.query("""
            SELECT * FROM paper WHERE embedding != NONE;
        """)
        
        if not results:
            return []
        
        # Compute similarities
        query_vec = np.array(query_embedding)
        
        scored = []
        for r in results:
            emb = r.get("embedding")
            if emb:
                doc_vec = np.array(emb)
                # Cosine similarity
                sim = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
                )
                if sim >= threshold:
                    paper = self._to_paper(r)
                    scored.append((paper, sim))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in scored[:limit]]
    
    # =========================================================================
    # LTM (Long-Term Memory) Operations
    # =========================================================================
    
    async def store_ltm_insight(
        self,
        domain: str,
        insight: str,
        confidence: float,
        metadata: Dict = None,
    ) -> bool:
        """Store long-term memory insight from evolution."""
        import uuid
        
        content = {
            "domain": domain,
            "insight": insight.replace('"', '\\"'),
            "confidence": confidence,
            "metadata": metadata or {},
            "created_at": "time::now()",
        }
        
        sql = f"CREATE ltm_insight:{uuid.uuid4().hex[:12]} CONTENT {json.dumps(content)};"
        result = await self.query(sql)
        return len(result) > 0
    
    async def get_ltm_insights(self, domain: str, limit: int = 20) -> List[Dict]:
        """Get LTM insights for domain."""
        sql = f"""
        SELECT * FROM ltm_insight 
        WHERE domain = '{domain}'
        ORDER BY confidence DESC
        LIMIT {limit};
        """
        return await self.query(sql)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _to_paper(self, record: Dict) -> Paper:
        """Convert DB record to Paper."""
        # Extract ID from SurrealDB format
        paper_id = record.get("id", "")
        if ":" in str(paper_id):
            paper_id = str(paper_id).split(":")[-1]
        
        return Paper(
            id=paper_id,
            title=record.get("title", ""),
            authors=record.get("authors", []),
            year=record.get("year"),
            abstract=record.get("abstract"),
            source=record.get("source", "unknown"),
            tags=record.get("tags", []),
            embedding=record.get("embedding"),
            url=record.get("url"),
        )
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# SINGLETON
# =============================================================================

_client: Optional[SurrealDBClient] = None


def get_db_client() -> SurrealDBClient:
    """Get singleton client."""
    global _client
    if _client is None:
        _client = SurrealDBClient()
    return _client


async def ensure_db_schema():
    """Ensure database schema exists."""
    client = get_db_client()
    
    # Create namespace/database if needed
    await client.query(f"USE NS {client.namespace} DB {client.database};")
    
    # Schema is schemaless by default in SurrealDB
    logger.info("Database schema ready")
