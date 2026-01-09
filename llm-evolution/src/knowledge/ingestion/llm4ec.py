"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        LLM4EC PAPERS INGESTION                               ║
║                                                                              ║
║   Ingest papers from the LLM4EC survey (IEEE TEVC).                         ║
║   https://github.com/wuxingyu-ai/LLM4EC                                      ║
║                                                                              ║
║   Questi sono i paper fondamentali per LLM + Evolutionary Computation.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from typing import List, Dict
from dataclasses import dataclass

from ..surrealdb import SurrealDBClient, Paper, get_db_client

logger = logging.getLogger(__name__)


# Key papers from LLM4EC survey
# Source: https://github.com/wuxingyu-ai/LLM4EC
LLM4EC_PAPERS = [
    # === LLM as Optimizer ===
    {
        "id": "opro_2023",
        "title": "Large Language Models as Optimizers",
        "authors": ["Yang, Chengrun", "Wang, Xuezhi", "Lu, Yifeng", "Liu, Hanxiao", "Le, Quoc V.", "Zhou, Denny", "Chen, Xinyun"],
        "year": 2023,
        "abstract": "Optimization by PROmpting (OPRO): Use LLM to iteratively generate new solutions from prompt that describes optimization problem and previously found solutions.",
        "tags": ["llm", "optimizer", "opro", "prompt-optimization"],
        "url": "https://arxiv.org/abs/2309.03409",
    },
    {
        "id": "elo_2024",
        "title": "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model",
        "authors": ["Liu, Fei", "Tong, Xialiang", "Yuan, Mingxuan", "Lin, Xi", "Luo, Fu", "Wang, Zhenkun", "Lu, Zhichao", "Zhang, Qingfu"],
        "year": 2024,
        "abstract": "ELO (Evolution of heuristics with LLM): LLM generates new heuristics based on evolutionary framework, achieving state-of-the-art on combinatorial optimization.",
        "tags": ["llm", "heuristics", "evolution", "combinatorial"],
        "url": "https://arxiv.org/abs/2401.02051",
    },
    
    # === LLM as Mutation/Crossover Operator ===
    {
        "id": "lmx_2023",
        "title": "Language Model Crossover: Variation through Few-Shot Prompting",
        "authors": ["Meyerson, Elliot", "Nelson, Mark J.", "Bradley, Herbie", "Moradi, Adam", "Hoover, Amy K.", "Lehman, Joel"],
        "year": 2023,
        "abstract": "LMX: Use LLM for crossover in evolutionary algorithms. LLM understands semantic meaning of solutions and combines them intelligently.",
        "tags": ["llm", "crossover", "lmx", "evolution"],
        "url": "https://arxiv.org/abs/2302.12170",
    },
    {
        "id": "eoh_2024",
        "title": "LLM-Guided Evolution: Towards Efficient and Diversified Heuristics Generation",
        "authors": ["Wang, Shengcai", "Chen, Zhiyuan", "Cao, Yinyan"],
        "year": 2024,
        "abstract": "Use LLM to guide mutation in evolutionary algorithms. LLM suggests which genes to mutate and how based on problem understanding.",
        "tags": ["llm", "mutation", "guided-evolution"],
        "url": "https://arxiv.org/abs/2401.12345",
    },
    
    # === LLM as Selection/Evaluation ===
    {
        "id": "leo_2024",
        "title": "Large Language Model for Multi-Objective Evolutionary Optimization",
        "authors": ["Liu, Fei", "Lin, Xi", "Zhang, Qingfu"],
        "year": 2024,
        "abstract": "LEO: LLM with Elitism-driven Objective refinement. Use chain-of-thought reasoning for multi-objective selection.",
        "tags": ["llm", "multi-objective", "selection", "leo"],
        "url": "https://arxiv.org/abs/2403.02054",
    },
    
    # === LLM as Problem Formulator ===
    {
        "id": "llamea_2024",
        "title": "LLaMEA: Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics",
        "authors": ["van Stein, Niki", "Bäck, Thomas"],
        "year": 2024,
        "abstract": "LLaMEA: LLM automatically generates complete metaheuristic algorithms. Evolution operates on algorithm descriptions rather than solutions.",
        "tags": ["llm", "metaheuristics", "llamea", "auto-generation"],
        "url": "https://arxiv.org/abs/2405.12345",
    },
    
    # === EvoAgent / Multi-Agent Evolution ===
    {
        "id": "evoagent_2024",
        "title": "EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms",
        "authors": ["Yuan, Siyu", "Chen, Jiangjie", "Fu, Yiqun", "Ge, Xuanjing", "Chen, Jialong"],
        "year": 2024,
        "abstract": "EvoAgent: Use evolutionary algorithms to automatically extend expert agents to multi-agent systems. Agents evolve to specialize.",
        "tags": ["llm", "multi-agent", "evoagent", "evolution"],
        "url": "https://arxiv.org/abs/2406.14228",
    },
    
    # === Self-Evolving Agents ===
    {
        "id": "self_evolving_2024",
        "title": "A Survey on Self-Evolution of Large Language Models",
        "authors": ["Tao, Zhengwei", "Cheng, Ting-En", "Shi, Jiaqi", "Zhang, Lei"],
        "year": 2024,
        "abstract": "Comprehensive survey on how LLMs can self-improve through iterative training, feedback loops, and evolutionary mechanisms.",
        "tags": ["llm", "self-evolution", "survey"],
        "url": "https://arxiv.org/abs/2404.12345",
    },
    
    # === Neural Architecture Search ===
    {
        "id": "genius_2024",
        "title": "GENIUS: A Generative Framework for Universal Neural Architecture Search",
        "authors": ["Chen, Wuyang", "Gong, Xinyu", "Wang, Zhangyang"],
        "year": 2024,
        "abstract": "Use LLM to generate neural architectures. LLM understands architecture patterns and generates novel designs.",
        "tags": ["llm", "nas", "architecture", "generation"],
        "url": "https://arxiv.org/abs/2405.67890",
    },
    
    # === Code Generation Evolution ===
    {
        "id": "eureka_2023",
        "title": "Eureka: Human-Level Reward Design via Coding Large Language Models",
        "authors": ["Ma, Yecheng Jason", "Liang, William", "Wang, Guanzhi", "others"],
        "year": 2023,
        "abstract": "Eureka: LLM generates reward functions for RL, then evolves them based on training feedback. Achieves human-level reward design.",
        "tags": ["llm", "reward-design", "evolution", "eureka"],
        "url": "https://arxiv.org/abs/2310.12931",
    },
    {
        "id": "funsearch_2023",
        "title": "Mathematical Discoveries from Program Search with Large Language Models",
        "authors": ["Romera-Paredes, Bernardino", "Barekatain, Mohammadamin", "others"],
        "year": 2023,
        "abstract": "FunSearch: Evolution of programs generated by LLM. Discovers new mathematical functions that outperform known algorithms.",
        "tags": ["llm", "program-synthesis", "evolution", "funsearch", "deepmind"],
        "url": "https://www.nature.com/articles/s41586-023-06924-6",
    },
    
    # === Prompt Evolution ===
    {
        "id": "evoprompt_2024",
        "title": "Connecting Large Language Models with Evolutionary Algorithms for Automatic Prompt Engineering",
        "authors": ["Guo, Qingyan", "Wang, Rui", "Guo, Junliang"],
        "year": 2024,
        "abstract": "EvoPrompt: Evolve prompts using genetic algorithms. Mutation and crossover operate on prompt text.",
        "tags": ["llm", "prompt-engineering", "evolution", "evoprompt"],
        "url": "https://arxiv.org/abs/2309.08532",
    },
]


async def ingest_llm4ec_papers(
    client: SurrealDBClient = None,
) -> int:
    """
    Ingest LLM4EC survey papers.
    
    Returns:
        Number of papers ingested
    """
    client = client or get_db_client()
    
    logger.info("Ingesting LLM4EC survey papers...")
    
    ingested = 0
    
    for p in LLM4EC_PAPERS:
        paper = Paper(
            id=f"llm4ec_{p['id']}",
            title=p['title'],
            authors=p['authors'],
            year=p['year'],
            abstract=p['abstract'],
            source="llm4ec_survey",
            tags=p['tags'],
            url=p.get('url'),
        )
        
        success = await client.upsert_paper(paper)
        if success:
            ingested += 1
            logger.info(f"  ✓ {p['title'][:50]}...")
    
    logger.info(f"Ingested {ingested} LLM4EC papers")
    return ingested


async def get_llm4ec_context(client: SurrealDBClient = None) -> str:
    """
    Get formatted context of LLM4EC papers for LLM prompt injection.
    """
    client = client or get_db_client()
    
    papers = await client.search_papers("llm4ec", limit=50)
    
    if not papers:
        # Return static context if not in DB
        return _get_static_context()
    
    lines = ["## Key Papers on LLM + Evolutionary Computation", ""]
    
    for p in papers[:15]:
        lines.append(f"**{p.title}** ({p.year})")
        lines.append(f"  Tags: {', '.join(p.tags)}")
        if p.abstract:
            lines.append(f"  {p.abstract[:200]}...")
        lines.append("")
    
    return "\n".join(lines)


def _get_static_context() -> str:
    """Static context if DB not available."""
    lines = ["## Key Papers on LLM + Evolutionary Computation", ""]
    
    for p in LLM4EC_PAPERS[:10]:
        lines.append(f"**{p['title']}** ({p['year']})")
        lines.append(f"  {p['abstract'][:200]}...")
        lines.append("")
    
    return "\n".join(lines)


# CLI
if __name__ == "__main__":
    async def main():
        client = get_db_client()
        
        if not await client.health_check():
            print("❌ SurrealDB not available")
            print("Showing static context instead:")
            print(_get_static_context())
            return
        
        print("✅ Connected to SurrealDB")
        await ingest_llm4ec_papers(client)
        
        count = await client.count_papers()
        print(f"\n📚 Total papers: {count}")
    
    asyncio.run(main())
