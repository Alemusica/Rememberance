#!/usr/bin/env python3
"""
SurrealDB MCP Server - Knowledge Base Integration

This MCP server exposes SurrealDB research papers as tools for Copilot/Claude.
Allows any agent to query the vibroacoustic research knowledge base.

Usage:
    python -m utils.surrealdb_mcp_server
    
Or configure in .vscode/mcp.json
"""

import json
import asyncio
import httpx
import base64
from typing import Any
import sys
import logging

# Configure logging to stderr (NEVER stdout for MCP!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("surrealdb-mcp")

# SurrealDB Configuration
SURREAL_URL = "http://localhost:8000/sql"
SURREAL_NS = "research"
SURREAL_DB = "knowledge"
SURREAL_USER = "root"
SURREAL_PASS = "root"

def get_auth_headers() -> dict:
    """Get authentication headers for SurrealDB"""
    auth = base64.b64encode(f"{SURREAL_USER}:{SURREAL_PASS}".encode()).decode()
    return {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "surreal-ns": SURREAL_NS,
        "surreal-db": SURREAL_DB
    }

async def query_surrealdb(query: str) -> list[dict]:
    """Execute a SurrealQL query"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SURREAL_URL,
                headers=get_auth_headers(),
                content=query,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0 and "result" in data[0]:
                return data[0]["result"]
            return []
        except Exception as e:
            logger.error(f"SurrealDB query error: {e}")
            return []

# ============================================================================
# MCP Tool Implementations
# ============================================================================

async def search_papers(query: str, limit: int = 10) -> str:
    """
    Search research papers by keyword in title, abstract, or authors.
    
    Args:
        query: Search term (e.g., "genetic algorithm", "ABH", "vibroacoustic")
        limit: Maximum results to return
    """
    sql = f"""
    SELECT cite_key, title, authors, year, section, project_relevance, abstract
    FROM paper
    WHERE 
        string::lowercase(title) CONTAINS string::lowercase('{query}')
        OR string::lowercase(abstract) CONTAINS string::lowercase('{query}')
        OR authors CONTAINS '{query}'
    LIMIT {limit}
    """
    results = await query_surrealdb(sql)
    
    if not results:
        return f"No papers found matching '{query}'"
    
    output = [f"Found {len(results)} papers:\n"]
    for paper in results:
        output.append(f"""
**{paper.get('title', 'Unknown')}** ({paper.get('year', 'N/A')})
- Authors: {', '.join(paper.get('authors', [])[:3])}...
- Section: {paper.get('section', 'N/A')}
- Cite Key: `{paper.get('cite_key', 'N/A')}`
- Relevance: {paper.get('project_relevance', 'N/A')[:200]}...
""")
    return "\n".join(output)


async def get_papers_by_section(section: str) -> str:
    """
    Get all papers from a specific research section.
    
    Args:
        section: One of: LUTHERIE, HUMAN_BODY, VIBROACOUSTIC, PLATE_VIBRATION,
                 EXCITER, GOLDEN_RATIO, CALIBRATION, WOOD, TRANSIENT,
                 MULTI_EXCITER, ACOUSTIC_BLACK, MODE_COUPLING
    """
    sql = f"""
    SELECT cite_key, title, authors, year, project_relevance
    FROM paper
    WHERE section = '{section.upper()}'
    ORDER BY year DESC
    """
    results = await query_surrealdb(sql)
    
    if not results:
        return f"No papers found in section '{section}'"
    
    output = [f"## {section.upper()} Section ({len(results)} papers)\n"]
    for paper in results:
        authors_str = ', '.join(paper.get('authors', [])[:2])
        output.append(f"- [{paper.get('cite_key')}] {paper.get('title', 'Unknown')[:60]}... ({paper.get('year')}) - {authors_str}")
    
    return "\n".join(output)


async def get_paper_details(cite_key: str) -> str:
    """
    Get full details of a specific paper by its citation key.
    
    Args:
        cite_key: The BibTeX citation key (e.g., "bai2004genetic", "krylov2014abh")
    """
    sql = f"""
    SELECT * FROM paper WHERE cite_key = '{cite_key}'
    """
    results = await query_surrealdb(sql)
    
    if not results:
        return f"Paper with cite_key '{cite_key}' not found"
    
    paper = results[0]
    return f"""
# {paper.get('title', 'Unknown')}

**Citation Key:** `{paper.get('cite_key')}`
**Year:** {paper.get('year')}
**Type:** {paper.get('type')}
**Journal:** {paper.get('journal', 'N/A')}

## Authors
{', '.join(paper.get('authors', []))}

## Section & Domains
- **Section:** {paper.get('section')}
- **Domains:** {', '.join(paper.get('domains', []))}

## Project Relevance
{paper.get('project_relevance', 'N/A')}

## Abstract
{paper.get('abstract', 'No abstract available')}

## Source
{paper.get('source', 'N/A')}
"""


async def get_key_papers() -> str:
    """
    Get the most important foundational papers for Golden Studio project.
    These are marked as KEY, CORE, or FOUNDATIONAL in the bibliography.
    """
    key_papers = [
        "bai2004genetic",      # NSGA-II basis
        "griffin1990handbook", # Body resonance
        "harris2010fundamentals", # DML theory
        "krylov2014abh",       # Acoustic Black Holes
        "skille1989vibroacoustic", # VAT founder
        "lu2012multiexciter",  # Multi-exciter optimization
        "deng2019abh",         # ABH focusing
        "sum2000coupling"      # Modal coupling
    ]
    
    output = ["# Key Foundational Papers for Golden Studio\n"]
    
    for cite_key in key_papers:
        sql = f"SELECT cite_key, title, year, project_relevance FROM paper WHERE cite_key = '{cite_key}'"
        results = await query_surrealdb(sql)
        if results:
            paper = results[0]
            output.append(f"""
### {paper.get('title', 'Unknown')} ({paper.get('year')})
- **Cite:** `{cite_key}`
- **Why Important:** {paper.get('project_relevance', 'N/A')[:300]}
""")
    
    return "\n".join(output)


async def get_knowledge_stats() -> str:
    """Get statistics about the knowledge base."""
    total_sql = "SELECT count() FROM paper GROUP ALL"
    sections_sql = "SELECT section, count() as cnt FROM paper GROUP BY section"
    algo_sql = "SELECT count() FROM algorithm GROUP ALL"
    
    total = await query_surrealdb(total_sql)
    sections = await query_surrealdb(sections_sql)
    algos = await query_surrealdb(algo_sql)
    
    output = ["# Knowledge Base Statistics\n"]
    output.append(f"**Total Papers:** {total[0]['count'] if total else 0}")
    output.append(f"**Total Algorithms:** {algos[0]['count'] if algos else 0}\n")
    output.append("## Papers by Section:")
    
    for s in sorted(sections, key=lambda x: x.get('cnt', 0), reverse=True):
        output.append(f"- {s.get('section', 'Unknown')}: {s.get('cnt', 0)}")
    
    return "\n".join(output)


# ============================================================================
# ALGORITHM TOOLS - Validated algorithms with paper traceability
# ============================================================================

async def get_algorithms(domain: str = None, min_success_rate: float = 0.0) -> str:
    """
    Get validated algorithms from the knowledge base.
    
    Args:
        domain: Filter by domain (optimization, physics, acoustics, therapy, dsp)
        min_success_rate: Minimum success rate (0-1)
    """
    conditions = [f"success_rate >= {min_success_rate}"]
    if domain:
        conditions.append(f"domain = '{domain}'")
    
    where_clause = " AND ".join(conditions)
    sql = f"""
    SELECT id, name, domain, description, success_rate, paper_sources, importance, implementation
    FROM algorithm
    WHERE {where_clause}
    ORDER BY success_rate DESC
    """
    results = await query_surrealdb(sql)
    
    if not results:
        return f"No algorithms found matching criteria"
    
    output = [f"# Validated Algorithms ({len(results)} found)\n"]
    for algo in results:
        papers = ', '.join(algo.get('paper_sources', []))
        output.append(f"""
## {algo.get('name')} [{algo.get('importance', '')}]
- **Domain:** {algo.get('domain')}
- **Success Rate:** {algo.get('success_rate', 0):.0%}
- **Implementation:** `{algo.get('implementation', 'N/A')}`
- **Paper Sources:** {papers}
- {algo.get('description', '')[:200]}
""")
    return "\n".join(output)


async def get_algorithm_details(algorithm_id: str) -> str:
    """
    Get full details of an algorithm including all paper sources.
    
    Args:
        algorithm_id: Algorithm ID (e.g., "nsga2_plate_optimizer", "fem_modal_analysis")
    """
    # Normalize ID
    if not algorithm_id.startswith("algorithm:"):
        algorithm_id = f"algorithm:{algorithm_id}"
    
    sql = f"SELECT * FROM {algorithm_id}"
    results = await query_surrealdb(sql)
    
    if not results:
        return f"Algorithm '{algorithm_id}' not found"
    
    algo = results[0]
    
    # Get details of source papers
    paper_details = []
    for cite_key in algo.get('paper_sources', []):
        paper_sql = f"SELECT cite_key, title, year FROM paper WHERE cite_key = '{cite_key}'"
        papers = await query_surrealdb(paper_sql)
        if papers:
            p = papers[0]
            paper_details.append(f"  - [{p.get('cite_key')}] {p.get('title', 'Unknown')[:50]}... ({p.get('year')})")
    
    params = algo.get('parameters', {})
    params_str = "\n".join([f"  - {k}: {v}" for k, v in params.items()]) if params else "  N/A"
    
    metrics = algo.get('success_metrics', {})
    metrics_str = "\n".join([f"  - {k}: {v}" for k, v in metrics.items()]) if metrics else "  N/A"
    
    return f"""
# {algo.get('name')}

**ID:** `{algo.get('id')}`
**Domain:** {algo.get('domain')}
**Importance:** {algo.get('importance', 'N/A')}
**Success Rate:** {algo.get('success_rate', 0):.0%}
**Project Validated:** {algo.get('project_validated', 'N/A')}

## Description
{algo.get('description', 'N/A')}

## Implementation
`{algo.get('implementation', 'N/A')}`

## Parameters
{params_str}

## Success Metrics
{metrics_str}

## Source Papers (Knowledge Provenance)
{chr(10).join(paper_details) if paper_details else '  No papers linked'}

## Notes
{algo.get('notes', 'N/A')}
"""


async def find_algorithms_by_paper(cite_key: str) -> str:
    """
    Find all algorithms that were inspired by a specific paper.
    
    Args:
        cite_key: The paper's citation key
    """
    sql = f"""
    SELECT id, name, domain, success_rate, importance
    FROM algorithm
    WHERE paper_sources CONTAINS '{cite_key}'
    """
    results = await query_surrealdb(sql)
    
    if not results:
        return f"No algorithms found that reference paper '{cite_key}'"
    
    output = [f"# Algorithms inspired by `{cite_key}` ({len(results)} found)\n"]
    for algo in results:
        output.append(f"- **{algo.get('name')}** ({algo.get('domain')}) - {algo.get('success_rate', 0):.0%} success [{algo.get('importance', '')}]")
    
    return "\n".join(output)


# ============================================================================
# MCP Server Protocol Implementation (STDIO)
# ============================================================================

class MCPServer:
    """Simple MCP Server using STDIO transport"""
    
    def __init__(self):
        self.tools = {
            "search_papers": {
                "description": "Search research papers by keyword in title, abstract, or authors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10}
                    },
                    "required": ["query"]
                },
                "handler": search_papers
            },
            "get_papers_by_section": {
                "description": "Get all papers from a specific research section",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string", "description": "Section name (e.g., LUTHERIE, MULTI_EXCITER)"}
                    },
                    "required": ["section"]
                },
                "handler": get_papers_by_section
            },
            "get_paper_details": {
                "description": "Get full details of a specific paper by citation key",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cite_key": {"type": "string", "description": "BibTeX citation key"}
                    },
                    "required": ["cite_key"]
                },
                "handler": get_paper_details
            },
            "get_key_papers": {
                "description": "Get the most important foundational papers for Golden Studio",
                "parameters": {"type": "object", "properties": {}},
                "handler": get_key_papers
            },
            "get_knowledge_stats": {
                "description": "Get statistics about the knowledge base (papers + algorithms)",
                "parameters": {"type": "object", "properties": {}},
                "handler": get_knowledge_stats
            },
            "get_algorithms": {
                "description": "Get validated algorithms with paper traceability",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "description": "Filter by domain: optimization, physics, acoustics, therapy, dsp"},
                        "min_success_rate": {"type": "number", "description": "Minimum success rate 0-1", "default": 0.0}
                    }
                },
                "handler": get_algorithms
            },
            "get_algorithm_details": {
                "description": "Get full details of an algorithm including source papers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "algorithm_id": {"type": "string", "description": "Algorithm ID (e.g., nsga2_plate_optimizer)"}
                    },
                    "required": ["algorithm_id"]
                },
                "handler": get_algorithm_details
            },
            "find_algorithms_by_paper": {
                "description": "Find algorithms inspired by a specific paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cite_key": {"type": "string", "description": "Paper citation key"}
                    },
                    "required": ["cite_key"]
                },
                "handler": find_algorithms_by_paper
            }
        }
    
    async def handle_request(self, request: dict) -> dict:
        """Handle incoming MCP request"""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "surrealdb-knowledge",
                        "version": "1.0.0"
                    }
                }
            }
        
        elif method == "tools/list":
            tools_list = []
            for name, tool in self.tools.items():
                tools_list.append({
                    "name": name,
                    "description": tool["description"],
                    "inputSchema": tool["parameters"]
                })
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools_list}
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name not in self.tools:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }
            
            handler = self.tools[tool_name]["handler"]
            try:
                result = await handler(**arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result}]
                    }
                }
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }
        
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }
    
    async def run(self):
        """Run the MCP server on STDIO"""
        logger.info("SurrealDB MCP Server starting...")
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                # Write response to stdout (MCP protocol)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
            except Exception as e:
                logger.error(f"Server error: {e}")


def main():
    """Entry point"""
    server = MCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
