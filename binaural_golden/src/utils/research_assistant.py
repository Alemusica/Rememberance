#!/usr/bin/env python3
"""
Research Assistant - AI-powered research integration.

Integrates SurrealDB knowledge base with development workflow.
Provides contextual research suggestions based on current code context.

Usage in development:
    from utils.research_assistant import ask, cite, suggest_papers
    
    # Get papers for current problem
    papers = ask("How to optimize exciter placement for flat frequency response?")
    
    # Get citation for code comment
    cite("lu2012")  # Returns formatted citation
    
    # Contextual suggestions
    suggest_papers("peninsula_detection")  # Based on function name
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from utils.knowledge_base import KnowledgeBase, Paper, get_kb


@dataclass
class ResearchContext:
    """Current development context for research suggestions."""
    topic: str
    related_concepts: List[str]
    suggested_papers: List[Paper]
    key_insights: List[str]


class ResearchAssistant:
    """
    AI research assistant for vibroacoustic plate optimization.
    
    Provides contextual research based on:
    - Current development context (function names, comments)
    - Historical session queries
    - Domain-specific knowledge mapping
    """
    
    # Map code concepts to research topics
    CONCEPT_MAPPING = {
        # Plate geometry
        "peninsula": ["acoustic black hole", "isolated region", "energy focusing", "ABH"],
        "cutout": ["topology optimization", "structural optimization", "mass reduction"],
        "thickness": ["tapered plate", "variable thickness", "gradient"],
        "taper": ["acoustic black hole", "wave trapping", "thickness profile"],
        
        # Exciters
        "exciter": ["exciter placement", "multi-exciter", "DML", "force location"],
        "multi_exciter": ["distributed mode", "exciter optimization", "genetic algorithm"],
        "placement": ["optimal placement", "force location", "modal coupling"],
        
        # Response zones
        "ear": ["binaural", "flat panel speaker", "directivity", "head position"],
        "spine": ["low frequency vibration", "therapy", "vibroacoustic stimulation"],
        "head": ["head response", "ear position", "binaural listening"],
        
        # Optimization
        "fitness": ["objective function", "optimization", "multi-objective"],
        "simp": ["topology optimization", "SIMP method", "density penalty"],
        "topology": ["structural optimization", "material distribution"],
        "evolution": ["genetic algorithm", "evolutionary optimization"],
        "modal": ["modal analysis", "eigenfrequency", "mode shape"],
        
        # Quality metrics
        "flatness": ["frequency response", "flat response", "equalization"],
        "uniformity": ["spatial uniformity", "balanced response"],
        "coupling": ["modal coupling", "acoustic coupling", "vibration transfer"],
    }
    
    # Key insights from research (distilled knowledge)
    DISTILLED_KNOWLEDGE = {
        "peninsula": [
            "Acoustic Black Holes (ABH) show that tapered regions FOCUS energy rather than trap it (Krylov 2014)",
            "Peninsula-like structures can enhance low-frequency response if properly designed (Deng 2019)",
            "Ring-shaped ABH achieves broadband vibration isolation - peninsulas can do similar (Deng 2019)",
            "Energy harvesting increases 3-5x with proper ABH taper design (Zhao 2014)",
            "Isolated regions redirect rather than block vibration energy (Feurtado 2017)",
        ],
        "peninsula_benefit": [
            "Acoustic Black Holes (ABH) show that tapered regions FOCUS energy rather than trap it (Krylov 2014)",
            "Peninsula-like structures can enhance low-frequency response if properly designed (Deng 2019)",
            "Ring-shaped ABH achieves broadband vibration isolation (Deng 2019)",
            "Energy harvesting increases with proper ABH design (Zhao 2014)",
        ],
        "multi_exciter": [
            "Optimal exciter number: 2-4 for most DML applications (Lu 2012)",
            "Genetic algorithm effective for exciter placement (Bai & Liu 2004)",
            "Multi-exciter reduces localization, improves uniformity (Jeon 2020)",
            "Golden ratio positions (0.382, 0.618) often near-optimal (Lu 2009)",
        ],
        "ear_flatness": [
            "Target: ±3dB for high-quality binaural (Pueo 2009)",
            "DML inherently more diffuse than cone speakers (Harris 1997)",
            "Head position affects perceived flatness (Anderson 2017)",
            "Equalization can compensate ±10dB variations (Shen 2006)",
        ],
        "spine_coupling": [
            "0-300Hz most effective for vibroacoustic therapy (clinical papers)",
            "40-80Hz optimal for deep tissue vibration (Campbell review)",
            "Somatosensory integration peaks at low frequencies (neuroscience)",
            "Plate size affects lower frequency limit (physics)",
        ],
        "topology_optimization": [
            "SIMP with p=3 works well for acoustic topology (Bokhari 2023)",
            "Start with 50% volume fraction, iterate (standard practice)",
            "Penalize disconnected regions or use filtering (manufacturability)",
            "Multi-objective: flatness vs mass vs manufacturability",
        ],
    }
    
    def __init__(self):
        self.kb = get_kb()
        self.session_history: List[str] = []
    
    def ask(self, question: str) -> ResearchContext:
        """
        Ask a research question in natural language.
        
        Returns context with relevant papers and distilled insights.
        """
        self.session_history.append(question)
        
        # Extract concepts from question
        concepts = self._extract_concepts(question)
        
        # Get relevant papers
        papers = self.kb.for_context(question)
        
        # Get distilled insights
        insights = []
        for concept in concepts:
            if concept in self.DISTILLED_KNOWLEDGE:
                insights.extend(self.DISTILLED_KNOWLEDGE[concept])
        
        # Deduplicate insights
        insights = list(dict.fromkeys(insights))
        
        return ResearchContext(
            topic=question,
            related_concepts=concepts,
            suggested_papers=papers[:10],
            key_insights=insights[:5]
        )
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract research concepts from text."""
        text_lower = text.lower()
        concepts = []
        
        for concept, keywords in self.CONCEPT_MAPPING.items():
            if concept in text_lower:
                concepts.append(concept)
            for kw in keywords:
                if kw.lower() in text_lower:
                    if concept not in concepts:
                        concepts.append(concept)
                    break
        
        return concepts
    
    def suggest_for_code(self, code_context: str) -> ResearchContext:
        """
        Get research suggestions based on code context.
        
        Pass function name, docstring, or code snippet.
        """
        return self.ask(code_context)
    
    def cite(self, key: str) -> Optional[str]:
        """
        Get formatted citation for a paper key.
        
        Example: cite("lu2012") -> "Lu et al. (2012) - Multi-exciter optimization..."
        """
        # Search for paper
        papers = self.kb.search(key, limit=5)
        
        for p in papers:
            # Check if key matches
            if key.lower() in p.id.lower() or key.lower() in p.title.lower():
                authors = p.authors[0] if p.authors else "Unknown"
                if len(p.authors) > 1:
                    authors += " et al."
                year = f" ({p.year})" if p.year else ""
                return f"{authors}{year} - {p.title}"
        
        return None
    
    def get_key_references(self, topic: str) -> Dict[str, str]:
        """
        Get key references for a topic, formatted for code comments.
        
        Returns dict of {short_key: formatted_citation}
        """
        papers = self.kb.for_context(topic)[:5]
        refs = {}
        
        for p in papers:
            if p.authors and p.year:
                author_key = p.authors[0].split(",")[0].split()[-1].lower()
                short_key = f"{author_key}{p.year}"
                refs[short_key] = f"{p.authors[0]}{' et al.' if len(p.authors) > 1 else ''} ({p.year})"
        
        return refs
    
    def print_context(self, ctx: ResearchContext):
        """Print formatted research context."""
        print(f"\n{'═' * 60}")
        print(f"  RESEARCH CONTEXT: {ctx.topic[:50]}")
        print(f"{'═' * 60}")
        
        if ctx.related_concepts:
            print(f"\n🔗 Related Concepts: {', '.join(ctx.related_concepts)}")
        
        if ctx.key_insights:
            print(f"\n💡 Key Insights:")
            for i, insight in enumerate(ctx.key_insights, 1):
                print(f"   {i}. {insight}")
        
        if ctx.suggested_papers:
            print(f"\n📄 Suggested Papers:")
            for i, p in enumerate(ctx.suggested_papers[:5], 1):
                year = f" ({p.year})" if p.year else ""
                print(f"   {i}. {p.title}{year}")
        
        print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_assistant: Optional[ResearchAssistant] = None

def get_assistant() -> ResearchAssistant:
    """Get singleton ResearchAssistant instance."""
    global _assistant
    if _assistant is None:
        _assistant = ResearchAssistant()
    return _assistant


def ask(question: str) -> ResearchContext:
    """Ask a research question."""
    return get_assistant().ask(question)


def cite(key: str) -> Optional[str]:
    """Get citation for a paper key."""
    return get_assistant().cite(key)


def suggest_papers(context: str) -> List[Paper]:
    """Get paper suggestions for code context."""
    return get_assistant().suggest_for_code(context).suggested_papers


def research(topic: str):
    """Print full research context for a topic."""
    ctx = ask(topic)
    get_assistant().print_context(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    assistant = ResearchAssistant()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        ctx = assistant.ask(query)
        assistant.print_context(ctx)
    else:
        # Interactive demo
        print("\n🔬 Research Assistant for Vibroacoustic Plate Optimization")
        print("=" * 60)
        
        demo_queries = [
            "How to optimize peninsula for energy focusing?",
            "Best exciter placement for flat ear response",
            "SIMP topology optimization for soundboard",
        ]
        
        for query in demo_queries:
            print(f"\n❓ {query}")
            ctx = assistant.ask(query)
            
            if ctx.key_insights:
                print(f"   💡 {ctx.key_insights[0]}")
            if ctx.suggested_papers:
                print(f"   📄 Top paper: {ctx.suggested_papers[0].title}")
