"""
ExplainerAgent - Human-readable anomaly explanation and action suggestion.

Integrates with PokayokeObserver to provide intelligent explanations
of evolutionary anomalies and suggest appropriate actions.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .base import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class ExplainerAgent(BaseAgent):
    """
    Agent that explains anomalies in evolutionary optimization and suggests actions.
    
    Takes anomaly context (type, generation, fitness history, genome) and generates:
    - Human-readable explanation
    - Severity assessment (1-10)
    - Recommended action (continue/pause/rollback)
    """
    
    EXPLAIN_PROMPT_TEMPLATE = """You are an expert evolutionary algorithm analyst. An anomaly has been detected during optimization. Your task is to explain what's happening and suggest an appropriate action.

ANOMALY TYPE: {anomaly_type}

GENERATION: {generation}

FITNESS HISTORY (last 10 generations):
{fitness_history}

CURRENT GENOME:
{genome}

ANOMALY DETAILS:
{anomaly_details}

TASK:
1. Explain in human-readable terms what anomaly was detected and why it's concerning
2. Assess severity on a scale of 1-10 (1=minor, 10=critical)
3. Recommend an action: "continue", "pause", or "rollback"

SEVERITY GUIDELINES:
- 1-3: Minor issue, can continue normally
- 4-6: Moderate concern, should pause and review
- 7-10: Critical issue, should rollback or abort

ACTION GUIDELINES:
- "continue": Anomaly is expected/normal, optimization can proceed
- "pause": Stop optimization to allow user review and decision
- "rollback": Revert to previous generation and try different approach

OUTPUT FORMAT:
Respond with a JSON object:
{{
    "explanation": "Clear, human-readable explanation of what's happening and why",
    "severity": 5,  // Integer 1-10
    "action": "pause",  // "continue", "pause", or "rollback"
    "reasoning": "Why this action is recommended",
    "suggested_adjustments": ["adjustment1", "adjustment2"]  // Optional suggestions
}}"""

    # Map anomaly types to descriptions
    ANOMALY_DESCRIPTIONS = {
        "STAGNATION": "No fitness improvement for multiple generations",
        "DIVERSITY_COLLAPSE": "Population becoming too similar (low diversity)",
        "FITNESS_REGRESSION": "Fitness getting worse instead of better",
        "CONSTRAINT_VIOLATION": "Physics or domain constraints violated",
        "GENE_ACTIVATION_READY": "Time to activate new genes (curriculum learning)",
        "PHYSICS_SUGGESTION": "Physics rules suggest intervention needed",
        "MEMORY_CHECKPOINT": "Good checkpoint for saving progress",
    }
    
    def __init__(
        self,
        llm_client: Any,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize ExplainerAgent.
        
        Args:
            llm_client: LLMClient instance
            config: Agent configuration
        """
        super().__init__(llm_client, config)
        logger.info("ExplainerAgent initialized")
    
    def _format_fitness_history(self, fitness_history: List[Union[float, Dict[str, float]]]) -> str:
        """
        Format fitness history for prompt.
        
        Args:
            fitness_history: List of fitness values or breakdowns
        
        Returns:
            Formatted string
        """
        if not fitness_history:
            return "No history available"
        
        lines = []
        start_gen = max(0, len(fitness_history) - 10)  # Last 10
        
        for i, fitness in enumerate(fitness_history[start_gen:], start=start_gen):
            if isinstance(fitness, dict):
                # Multi-objective: show breakdown
                fitness_str = json.dumps(fitness, indent=2)
                lines.append(f"Generation {i}:\n{fitness_str}")
            else:
                # Single objective
                lines.append(f"Generation {i}: {fitness:.4f}")
        
        return "\n".join(lines)
    
    def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate explanation prompt.
        
        Expected context keys:
        - anomaly_type: Type of anomaly (str or AnomalyType enum)
        - generation: Current generation number
        - fitness_history: List of fitness values/breakdowns
        - genome: Current genome (dict)
        - anomaly_details: Optional additional details (dict)
        """
        anomaly_type = context["anomaly_type"]
        generation = context["generation"]
        fitness_history = context.get("fitness_history", [])
        genome = context.get("genome", {})
        anomaly_details = context.get("anomaly_details", {})
        
        # Convert anomaly type to string
        if hasattr(anomaly_type, 'name'):
            anomaly_type_str = anomaly_type.name
        elif hasattr(anomaly_type, 'value'):
            anomaly_type_str = anomaly_type.value
        else:
            anomaly_type_str = str(anomaly_type)
        
        # Get anomaly description
        anomaly_desc = self.ANOMALY_DESCRIPTIONS.get(
            anomaly_type_str.upper(),
            f"Anomaly type: {anomaly_type_str}"
        )
        
        # Format fitness history
        fitness_history_str = self._format_fitness_history(fitness_history)
        
        # Format genome
        genome_str = json.dumps(genome, indent=2) if isinstance(genome, dict) else str(genome)
        
        # Format anomaly details
        details_str = json.dumps(anomaly_details, indent=2) if anomaly_details else "No additional details"
        
        prompt = self.EXPLAIN_PROMPT_TEMPLATE.format(
            anomaly_type=f"{anomaly_type_str} - {anomaly_desc}",
            generation=generation,
            fitness_history=fitness_history_str,
            genome=genome_str,
            anomaly_details=details_str,
        )
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into explanation result.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Dictionary with:
            - explanation: Human-readable explanation
            - severity: Integer 1-10
            - action: "continue", "pause", or "rollback"
            - reasoning: Why this action
            - suggested_adjustments: Optional list of suggestions
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
            
            # Validate required fields
            required_keys = ["explanation", "severity", "action"]
            for key in required_keys:
                if key not in parsed:
                    raise ValueError(f"Missing required key in response: {key}")
            
            # Validate severity range
            severity = parsed["severity"]
            if not isinstance(severity, int) or severity < 1 or severity > 10:
                logger.warning(f"Invalid severity {severity}, clamping to 1-10")
                parsed["severity"] = max(1, min(10, severity))
            
            # Validate action
            action = parsed["action"].lower()
            valid_actions = ["continue", "pause", "rollback"]
            if action not in valid_actions:
                logger.warning(f"Invalid action {action}, defaulting to 'pause'")
                parsed["action"] = "pause"
            else:
                parsed["action"] = action
            
            # Ensure optional fields exist
            parsed.setdefault("reasoning", "No reasoning provided")
            parsed.setdefault("suggested_adjustments", [])
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response[:500]}")
            raise ValueError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise
    
    async def explain(
        self,
        anomaly_type: Union[str, Any],
        generation: int,
        fitness_history: List[Union[float, Dict[str, float]]],
        genome: Dict[str, Any],
        anomaly_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Explain an anomaly and suggest action.
        
        Convenience method that calls invoke with proper context.
        
        Args:
            anomaly_type: Type of anomaly (str or AnomalyType enum)
            generation: Current generation number
            fitness_history: List of fitness values/breakdowns
            genome: Current genome
            anomaly_details: Optional additional anomaly details
        
        Returns:
            Explanation result dictionary
        """
        context = {
            "anomaly_type": anomaly_type,
            "generation": generation,
            "fitness_history": fitness_history,
            "genome": genome,
            "anomaly_details": anomaly_details or {},
        }
        
        return await self.invoke(context)
    
    async def explain_from_pokayoke(
        self,
        anomaly_context: Any,  # AnomalyContext from pokayoke_observer
        fitness_history: List[Union[float, Dict[str, float]]],
        genome: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Explain anomaly from PokayokeObserver's AnomalyContext.
        
        This method integrates directly with PokayokeObserver.
        
        Args:
            anomaly_context: AnomalyContext object from pokayoke_observer
            fitness_history: List of fitness values/breakdowns
            genome: Current genome
        
        Returns:
            Explanation result dictionary
        """
        # Extract info from AnomalyContext
        anomaly_type = anomaly_context.anomaly_type
        generation = anomaly_context.generation
        
        anomaly_details = {
            "severity": anomaly_context.severity,
            "current_fitness": anomaly_context.current_fitness,
            "best_fitness_ever": anomaly_context.best_fitness_ever,
            "fitness_velocity": anomaly_context.fitness_velocity,
            "population_diversity": anomaly_context.population_diversity,
            "stagnation_generations": anomaly_context.stagnation_generations,
            "suggested_actions": [str(a) for a in anomaly_context.suggested_actions],
            "explanation": anomaly_context.explanation,
        }
        
        return await self.explain(
            anomaly_type=anomaly_type,
            generation=generation,
            fitness_history=fitness_history,
            genome=genome,
            anomaly_details=anomaly_details,
        )
