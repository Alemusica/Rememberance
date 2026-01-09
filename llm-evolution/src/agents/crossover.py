"""
CrossoverAgent - Language Model Crossover for Evolutionary Optimization

Implements Language Model Crossover (LMX) as described in:
Meyerson et al. (2023) "Language Model Crossover: Variation Through Few-Shot Prompting"

The agent takes two parent genomes with their fitness breakdowns and generates
an offspring that combines the best traits from both parents.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class CrossoverAgent(BaseAgent):
    """
    Agent that performs crossover between two parent genomes using LLM reasoning.
    
    Given two parents with their fitness breakdowns, the agent analyzes which
    aspects each parent excels at and combines them into a new valid genome.
    """
    
    CROSSOVER_PROMPT_TEMPLATE = """You are an expert evolutionary algorithm designer. Your task is to perform crossover between two parent genomes to create an offspring that inherits the best traits from both.

PARENT A:
Genome: {parent_a_genome}
Fitness Breakdown:
{parent_a_fitness}

Strengths: {parent_a_strengths}
Weaknesses: {parent_a_weaknesses}

PARENT B:
Genome: {parent_b_genome}
Fitness Breakdown:
{parent_b_fitness}

Strengths: {parent_b_strengths}
Weaknesses: {parent_b_weaknesses}

TASK:
Create a new offspring genome that:
1. Inherits the best aspects from Parent A (especially: {parent_a_strengths})
2. Inherits the best aspects from Parent B (especially: {parent_b_strengths})
3. Avoids the weaknesses of both parents
4. Is a valid genome in the domain (satisfies all constraints)

DOMAIN CONSTRAINTS:
{domain_constraints}

OUTPUT FORMAT:
Respond with a JSON object containing:
{{
    "offspring_genome": {{...}},  // The new genome (same structure as parents)
    "inherited_from_a": ["trait1", "trait2"],  // What was inherited from Parent A
    "inherited_from_b": ["trait1", "trait2"],  // What was inherited from Parent B
    "rationale": "Explanation of why this combination should work well",
    "validation": {{
        "is_valid": true,
        "constraints_satisfied": ["constraint1", "constraint2"]
    }}
}}

Remember: The offspring must be a valid genome that satisfies all domain constraints."""

    def __init__(
        self,
        llm_client: Any,
        config: Optional[AgentConfig] = None,
        domain_validator: Optional[callable] = None,
    ):
        """
        Initialize CrossoverAgent.
        
        Args:
            llm_client: LLMClient instance
            config: Agent configuration
            domain_validator: Optional function to validate genome validity
                            (genome) -> bool
        """
        super().__init__(llm_client, config)
        self.domain_validator = domain_validator
        logger.info("CrossoverAgent initialized")
    
    def _extract_strengths_weaknesses(self, fitness_breakdown: Dict[str, float]) -> tuple[List[str], List[str]]:
        """
        Extract strengths and weaknesses from fitness breakdown.
        
        Args:
            fitness_breakdown: Dictionary mapping objective names to values
        
        Returns:
            (strengths, weaknesses) tuple
        """
        # Sort objectives by value (higher is better)
        sorted_obj = sorted(fitness_breakdown.items(), key=lambda x: x[1], reverse=True)
        
        # Top 40% are strengths, bottom 40% are weaknesses
        n = len(sorted_obj)
        top_n = max(1, int(n * 0.4))
        bottom_n = max(1, int(n * 0.4))
        
        strengths = [obj[0] for obj in sorted_obj[:top_n]]
        weaknesses = [obj[0] for obj in sorted_obj[-bottom_n:]]
        
        return strengths, weaknesses
    
    def generate_prompt(
        self,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate crossover prompt.
        
        Expected context keys:
        - parent_a: Parent A genome (dict)
        - parent_b: Parent B genome (dict)
        - parent_a_fitness: Parent A fitness breakdown (dict)
        - parent_b_fitness: Parent B fitness breakdown (dict)
        - domain_constraints: Optional string describing domain constraints
        """
        parent_a = context["parent_a"]
        parent_b = context["parent_b"]
        parent_a_fitness = context["parent_a_fitness"]
        parent_b_fitness = context["parent_b_fitness"]
        domain_constraints = context.get("domain_constraints", "No specific constraints provided.")
        
        # Extract strengths/weaknesses
        a_strengths, a_weaknesses = self._extract_strengths_weaknesses(parent_a_fitness)
        b_strengths, b_weaknesses = self._extract_strengths_weaknesses(parent_b_fitness)
        
        # Format genomes as JSON
        parent_a_str = json.dumps(parent_a, indent=2)
        parent_b_str = json.dumps(parent_b, indent=2)
        fitness_a_str = json.dumps(parent_a_fitness, indent=2)
        fitness_b_str = json.dumps(parent_b_fitness, indent=2)
        
        prompt = self.CROSSOVER_PROMPT_TEMPLATE.format(
            parent_a_genome=parent_a_str,
            parent_a_fitness=fitness_a_str,
            parent_a_strengths=", ".join(a_strengths),
            parent_a_weaknesses=", ".join(a_weaknesses),
            parent_b_genome=parent_b_str,
            parent_b_fitness=fitness_b_str,
            parent_b_strengths=", ".join(b_strengths),
            parent_b_weaknesses=", ".join(b_weaknesses),
            domain_constraints=domain_constraints,
        )
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured crossover result.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Dictionary with:
            - offspring_genome: The new genome
            - inherited_from_a: List of traits from Parent A
            - inherited_from_b: List of traits from Parent B
            - rationale: Explanation
            - validation: Validation info
        """
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                # Remove markdown code blocks if present
                if "```json" in response:
                    start = response.find("```json") + 7
                    end = response.find("```", start)
                    response = response[start:end].strip()
                elif "```" in response:
                    start = response.find("```") + 3
                    end = response.find("```", start)
                    response = response[start:end].strip()
                
                parsed = json.loads(response)
            else:
                parsed = response
            
            # Validate structure
            required_keys = ["offspring_genome", "inherited_from_a", "inherited_from_b", "rationale"]
            for key in required_keys:
                if key not in parsed:
                    raise ValueError(f"Missing required key in response: {key}")
            
            # Validate genome if validator provided
            if self.domain_validator:
                genome = parsed["offspring_genome"]
                is_valid = self.domain_validator(genome)
                if not is_valid:
                    logger.warning("LLM generated invalid genome, validation failed")
                    parsed["validation"] = parsed.get("validation", {})
                    parsed["validation"]["is_valid"] = False
                    parsed["validation"]["validation_error"] = "Domain validator rejected genome"
                else:
                    parsed["validation"] = parsed.get("validation", {})
                    parsed["validation"]["is_valid"] = True
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response[:500]}")
            raise ValueError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise
    
    async def crossover(
        self,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
        parent_a_fitness: Dict[str, float],
        parent_b_fitness: Dict[str, float],
        domain_constraints: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform crossover between two parents.
        
        Convenience method that calls invoke with proper context.
        
        Args:
            parent_a: Parent A genome
            parent_b: Parent B genome
            parent_a_fitness: Parent A fitness breakdown
            parent_b_fitness: Parent B fitness breakdown
            domain_constraints: Optional domain constraint description
        
        Returns:
            Crossover result dictionary
        """
        context = {
            "parent_a": parent_a,
            "parent_b": parent_b,
            "parent_a_fitness": parent_a_fitness,
            "parent_b_fitness": parent_b_fitness,
            "domain_constraints": domain_constraints or "Standard genome constraints apply.",
        }
        
        return await self.invoke(context)
