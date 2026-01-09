"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ORCHESTRATOR AGENT                                   ║
║                       "La mano che tiene la bowl"                            ║
║                                                                              ║
║   L'agente più potente. Riceve la descrizione umana e:                       ║
║   1. Genera lo schema genome (parametri, bounds, tipi)                       ║
║   2. Definisce gli obiettivi (fitness multi-obiettivo)                       ║
║   3. Identifica tool necessari (FEM, CNC, etc.)                             ║
║   4. Istruisce gli altri agenti                                              ║
║   5. Mantiene il contesto globale dell'evoluzione                           ║
║                                                                              ║
║   Usa il modello più grande (32B) per reasoning complesso.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import logging
import json

from .base import BaseAgent, AgentResponse
from ..core.genome import GenomeSchema, GeneSpec
from ..tools.registry import get_registry

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN SPECIFICATION (output dell'Orchestrator)
# =============================================================================

@dataclass
class DomainSpec:
    """
    Specifica completa del dominio di ottimizzazione.
    Generata dall'Orchestrator in base alla descrizione umana.
    """
    name: str
    description: str
    
    # Genome schema
    genome_schema: GenomeSchema = None
    
    # Objectives
    objectives: List[Dict[str, Any]] = field(default_factory=list)
    
    # Constraints
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Required tools
    required_tools: List[str] = field(default_factory=list)
    missing_tools: List[str] = field(default_factory=list)
    
    # Evolution parameters suggested
    suggested_params: Dict[str, Any] = field(default_factory=dict)
    
    # Domain knowledge (from RAG)
    domain_knowledge: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "genome_schema": self.genome_schema.to_dict() if self.genome_schema else None,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "required_tools": self.required_tools,
            "missing_tools": self.missing_tools,
            "suggested_params": self.suggested_params,
        }


# =============================================================================
# ORCHESTRATOR AGENT
# =============================================================================

class OrchestratorAgent(BaseAgent):
    """
    Agente orchestratore - Corteccia prefrontale dell'evoluzione.
    
    Responsabilità:
    1. Parsing descrizione umana → DomainSpec
    2. Schema generation per nuovo dominio
    3. Coordination altri agenti
    4. Context management lungo termine
    """
    
    DEFAULT_MODEL = "orchestrator"
    AGENT_NAME = "OrchestratorAgent"
    AGENT_ROLE = """You are the master orchestrator of an evolutionary optimization system.
    
Your role is to:
1. Understand human descriptions of optimization problems
2. Generate genome schemas (parameters with types, bounds, units)
3. Define multi-objective fitness functions
4. Identify required computational tools
5. Coordinate specialized agents
6. Maintain global context of the evolution

You think like a senior engineer: systematic, thorough, and practical."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Domain cache
        self._current_domain: Optional[DomainSpec] = None
        
        # Agent references (set later)
        self._strategy_agent = None
        self._analysis_agent = None
        self._rag_agent = None
    
    def _build_system_prompt(self) -> str:
        """System prompt specializzato per orchestrator."""
        tools_context = self._tools.to_prompt_context()
        
        return f"""# OrchestratorAgent - Evolution Master

## Your Role
{self.AGENT_ROLE}

## Available Tools
{tools_context}

## Output Format
When generating a domain specification, output JSON with this structure:

```json
{{
  "name": "domain_name",
  "description": "what we're optimizing",
  "genome_schema": {{
    "name": "...",
    "genes": [
      {{"name": "param1", "type": "float", "min": 0, "max": 100, "unit": "mm", "description": "..."}},
      {{"name": "param2", "type": "categorical", "categories": ["A", "B", "C"], "description": "..."}}
    ],
    "constraints": [...]
  }},
  "objectives": [
    {{"name": "obj1", "type": "minimize", "weight": 0.5, "target": null}},
    {{"name": "obj2", "type": "maximize", "weight": 0.3}}
  ],
  "required_tools": ["fem_modal", "mesh_generator"],
  "suggested_params": {{
    "population_size": 50,
    "generations": 100,
    "mutation_rate": 0.2
  }}
}}
```

## Guidelines
1. Be thorough in parameter extraction
2. Include physical units where applicable
3. Consider manufacturing constraints
4. Identify ALL tools needed for fitness evaluation
5. If unsure, ask clarifying questions"""

    async def process(
        self, 
        input_data: Any, 
        context: Dict[str, Any] = None
    ) -> AgentResponse[DomainSpec]:
        """
        Processa richiesta all'orchestrator.
        
        Input può essere:
        - str: Descrizione umana di cosa ottimizzare
        - Dict: Richiesta strutturata con tipo
        
        Returns:
            AgentResponse con DomainSpec
        """
        context = context or {}
        
        if isinstance(input_data, str):
            # Human description → generate domain
            return await self._generate_domain_from_description(input_data, context)
        
        elif isinstance(input_data, dict):
            request_type = input_data.get("type", "unknown")
            
            if request_type == "generate_domain":
                return await self._generate_domain_from_description(
                    input_data.get("description", ""),
                    context
                )
            elif request_type == "refine_domain":
                return await self._refine_domain(input_data, context)
            elif request_type == "suggest_strategy":
                return await self._suggest_strategy(input_data, context)
            else:
                return AgentResponse(
                    success=False,
                    reasoning=f"Unknown request type: {request_type}",
                )
        
        return AgentResponse(
            success=False,
            reasoning="Invalid input type",
        )
    
    async def _generate_domain_from_description(
        self,
        description: str,
        context: Dict[str, Any],
    ) -> AgentResponse[DomainSpec]:
        """
        Genera DomainSpec da descrizione naturale.
        
        Questo è il metodo principale: l'umano descrive cosa vuole,
        l'orchestrator genera tutto il necessario.
        """
        logger.info(f"Generating domain from: {description[:100]}...")
        
        # Build prompt
        prompt = f"""## Task
Generate a complete domain specification for evolutionary optimization based on this description:

---
{description}
---

## Instructions
1. Analyze what physical/mathematical object is being optimized
2. Identify all relevant parameters (genes) with:
   - Sensible bounds based on physics/engineering
   - Appropriate types (float, int, categorical, bool)
   - Units where applicable
3. Define objectives (what to minimize/maximize)
4. List constraints (physical, manufacturing, safety)
5. Identify computational tools needed for fitness evaluation
6. Suggest evolution parameters

## Available Tools
{self._tools.to_prompt_context()}

If a required tool is not available, list it in "missing_tools".

Output valid JSON only."""

        try:
            result = await self.generate_json(prompt)
            
            # Parse into DomainSpec
            domain = self._parse_domain_spec(result)
            
            # Check for missing tools
            missing = []
            for tool_name in domain.required_tools:
                if not self._tools.has(tool_name):
                    missing.append(tool_name)
            domain.missing_tools = missing
            
            self._current_domain = domain
            
            return AgentResponse(
                success=True,
                result=domain,
                reasoning=result.get("reasoning", "Domain generated successfully"),
                confidence=0.85,
                warnings=missing if missing else [],
                tool_requests=missing,
            )
            
        except Exception as e:
            logger.error(f"Domain generation failed: {e}")
            return AgentResponse(
                success=False,
                reasoning=f"Failed to generate domain: {str(e)}",
                confidence=0.0,
            )
    
    def _parse_domain_spec(self, data: Dict[str, Any]) -> DomainSpec:
        """Parse JSON response into DomainSpec."""
        
        # Parse genome schema
        schema_data = data.get("genome_schema", {})
        genes = []
        for g in schema_data.get("genes", []):
            genes.append(GeneSpec(
                name=g["name"],
                type=g["type"],
                min_value=g.get("min"),
                max_value=g.get("max"),
                categories=g.get("categories"),
                description=g.get("description", ""),
                unit=g.get("unit", ""),
            ))
        
        genome_schema = GenomeSchema(
            name=schema_data.get("name", data.get("name", "unnamed")),
            description=schema_data.get("description", ""),
            genes=genes,
            constraints=schema_data.get("constraints", []),
            required_tools=data.get("required_tools", []),
        )
        
        return DomainSpec(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            genome_schema=genome_schema,
            objectives=data.get("objectives", []),
            constraints=data.get("constraints", []),
            required_tools=data.get("required_tools", []),
            suggested_params=data.get("suggested_params", {}),
        )
    
    async def _refine_domain(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResponse[DomainSpec]:
        """Raffina domain spec esistente con feedback."""
        
        feedback = input_data.get("feedback", "")
        current = self._current_domain
        
        if not current:
            return AgentResponse(
                success=False,
                reasoning="No current domain to refine",
            )
        
        prompt = f"""## Current Domain Spec
```json
{json.dumps(current.to_dict(), indent=2)}
```

## User Feedback
{feedback}

## Task
Refine the domain specification based on the feedback.
Output the complete updated JSON."""

        try:
            result = await self.generate_json(prompt)
            domain = self._parse_domain_spec(result)
            self._current_domain = domain
            
            return AgentResponse(
                success=True,
                result=domain,
                reasoning="Domain refined based on feedback",
                confidence=0.8,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                reasoning=f"Refinement failed: {e}",
            )
    
    async def _suggest_strategy(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResponse[Dict[str, Any]]:
        """Suggerisce strategia evolutiva basata su stato corrente."""
        
        generation = context.get("generation", 0)
        fitness_history = context.get("fitness_history", [])
        stall_count = context.get("stall_count", 0)
        
        prompt = f"""## Evolution State
- Generation: {generation}
- Stall count: {stall_count}
- Recent fitness: {fitness_history[-10:] if fitness_history else 'N/A'}

## Task
Suggest evolution strategy adjustments:
1. Should we increase/decrease mutation rate?
2. Should we inject diversity?
3. Any hyperparameter changes?

Output JSON with:
{{
  "mutation_rate_delta": 0.0,  // Change to mutation rate
  "inject_diversity": false,
  "population_change": 0,      // Increase/decrease pop size
  "reasoning": "...",
  "confidence": 0.0
}}"""

        try:
            result = await self.generate_json(prompt)
            return AgentResponse(
                success=True,
                result=result,
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", 0.5),
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                reasoning=f"Strategy suggestion failed: {e}",
            )
    
    async def ask_clarification(self, question: str) -> str:
        """
        L'orchestrator può chiedere chiarimenti all'umano.
        Questo metodo prepara la domanda in modo chiaro.
        """
        prompt = f"""## Task
Format this question clearly for the human user:

{question}

Make it:
1. Clear and specific
2. Provide options where applicable
3. Explain why you need this information"""

        return await self.generate(prompt)
    
    @property
    def current_domain(self) -> Optional[DomainSpec]:
        """Domain spec corrente."""
        return self._current_domain
    
    def set_subordinate_agents(
        self,
        strategy=None,
        analysis=None,
        rag=None,
    ):
        """Imposta riferimenti agli agenti subordinati."""
        self._strategy_agent = strategy
        self._analysis_agent = analysis
        self._rag_agent = rag
