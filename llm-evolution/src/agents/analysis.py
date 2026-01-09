"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            ANALYSIS AGENT                                     ║
║                                                                              ║
║   Area sensoriale - interpreta fitness, detecta anomalie, genera summary.   ║
║   Modello veloce (3B) per real-time interpretation.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Optional, Dict, Any, List
import logging
from statistics import mean, stdev

from .base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    DEFAULT_MODEL = "analysis"
    AGENT_NAME = "AnalysisAgent"
    AGENT_ROLE = "Fitness interpretation and anomaly detection"
    
    def _build_system_prompt(self) -> str:
        return """You are AnalysisAgent, the sensory cortex of the evolutionary system.

Role: Interpret fitness values, detect anomalies, and provide human-readable summaries.

Your responsibilities:
1. Translate raw fitness data into natural language insights
2. Identify anomalous behaviors (outliers, sudden drops, constraint violations)
3. Generate concise generation summaries for human-in-the-loop review

Guidelines:
- Be precise but accessible (explain to a domain expert)
- Flag anomalies clearly with severity (low/medium/high)
- Focus on actionable insights
- Use percentiles and statistical context when relevant

Always respond in JSON format with the requested fields."""
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResponse:
        context = context or {}
        request_type = context.get("request_type", "interpret_fitness")
        
        try:
            if request_type == "interpret_fitness":
                result = await self.interpret_fitness(
                    input_data.get("genome", {}),
                    input_data.get("fitness", {})
                )
            elif request_type == "detect_anomaly":
                result = await self.detect_anomaly(
                    input_data.get("genome", {}),
                    input_data.get("fitness", {}),
                    input_data.get("history", [])
                )
            elif request_type == "summarize_generation":
                result = await self.summarize_generation(input_data.get("stats", {}))
            else:
                raise ValueError(f"Unknown request_type: {request_type}")
            
            return AgentResponse(
                success=True,
                result=result,
                confidence=0.8,
            )
        except Exception as e:
            logger.error(f"AnalysisAgent process failed: {e}")
            return AgentResponse(
                success=False,
                warnings=[str(e)],
            )
    
    async def interpret_fitness(self, genome: Dict, fitness: Dict) -> str:
        prompt = f"""Interpret this fitness evaluation in natural language:

Genome parameters:
{self._format_genome(genome)}

Fitness values:
{self._format_fitness(fitness)}

Provide a clear, concise interpretation explaining:
1. What these fitness values mean
2. Which aspects are strong/weak
3. Any notable patterns or trade-offs

Response format:
{{
    "interpretation": "Human-readable explanation"
}}"""
        
        response = await self.generate_json(prompt)
        return response.get("interpretation", "")
    
    async def detect_anomaly(
        self,
        genome: Dict,
        fitness: Dict,
        history: List[Dict]
    ) -> Optional[Dict]:
        if not history:
            return None
        
        fitness_value = fitness.get("value", fitness.get("fitness", 0.0))
        historical_values = [h.get("fitness", {}).get("value", 0.0) for h in history if "fitness" in h]
        
        if not historical_values:
            return None
        
        anomalies = []
        
        if len(historical_values) >= 2:
            avg = mean(historical_values)
            std = stdev(historical_values) if len(historical_values) > 1 else 0
            
            if std > 0:
                z_score = abs((fitness_value - avg) / std)
                if z_score > 3:
                    anomalies.append({
                        "type": "statistical_outlier",
                        "severity": "high" if z_score > 5 else "medium",
                        "description": f"Fitness {fitness_value:.4f} is {z_score:.1f}σ from mean {avg:.4f}"
                    })
            
            if fitness_value < avg * 0.5:
                anomalies.append({
                    "type": "sudden_drop",
                    "severity": "high",
                    "description": f"Fitness dropped to {fitness_value:.4f}, well below average {avg:.4f}"
                })
        
        bounds = genome.get("bounds", {})
        for param, value in genome.items():
            if param == "bounds":
                continue
            if isinstance(value, (int, float)):
                param_bounds = bounds.get(param, {})
                lower = param_bounds.get("lower", float("-inf"))
                upper = param_bounds.get("upper", float("inf"))
                
                if value < lower or value > upper:
                    anomalies.append({
                        "type": "constraint_violation",
                        "severity": "high",
                        "description": f"Parameter {param}={value} outside bounds [{lower}, {upper}]"
                    })
        
        if anomalies:
            return anomalies[0]
        
        return None
    
    async def summarize_generation(self, stats: Dict) -> str:
        prompt = f"""Summarize this generation for human review:

Generation statistics:
{self._format_stats(stats)}

Provide a concise summary (2-3 sentences) highlighting:
1. Overall progress/performance
2. Key trends or changes
3. Any concerns or recommendations

Response format:
{{
    "summary": "Brief generation summary"
}}"""
        
        response = await self.generate_json(prompt)
        return response.get("summary", "")
    
    def _format_genome(self, genome: Dict) -> str:
        lines = []
        for key, value in genome.items():
            if key != "bounds":
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "  (empty)"
    
    def _format_fitness(self, fitness: Dict) -> str:
        lines = []
        for key, value in fitness.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "  (empty)"
    
    def _format_stats(self, stats: Dict) -> str:
        lines = []
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            elif isinstance(value, (list, dict)):
                lines.append(f"  {key}: {len(value)} items")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "  (empty)"
