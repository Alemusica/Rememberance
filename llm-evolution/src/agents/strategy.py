"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          STRATEGY AGENT                                      ║
║                           "Area motoria"                                     ║
║                                                                              ║
║   Agente responsabile di decisioni su hyperparameters evolutivi (GA/ES).     ║
║   Analizza la convergence curve e suggerisce modifiche a:                    ║
║   - mutation rate                                                           ║
║   - crossover rate                                                          ║
║   - iniezione di diversità                                                   ║
║                                                                              ║
║   Modello: strategy (14B)                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import math

from .base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class StrategyAgent(BaseAgent):
    """
    StrategyAgent - area motoria del sistema nervoso dell'evoluzione.

    Responsabilità:
    - Analizzare la curva di convergenza (fitness_history)
    - Suggerire aggiustamenti agli hyperparameters per migliorare esplorazione/sfruttamento
    - Decidere quando invocare operatori basati su LLM (mutation/crossover) vs operatori standard
    """

    DEFAULT_MODEL = "strategy"
    AGENT_NAME = "StrategyAgent"
    AGENT_ROLE = """You are the motor cortex ("area motoria") of an evolutionary optimization nervous system.

Your job is to adjust evolutionary hyperparameters in real time to keep progress stable:
- Detect improvement, stagnation, and regressions from fitness history
- Tune mutation and crossover rates
- Decide when to inject diversity
- Decide when to invoke LLM-based operators versus standard operators

You are pragmatic and conservative: prefer small deltas, clamp to safe ranges, and only take big actions when clear stagnation is detected."""

    def _build_system_prompt(self) -> str:
        tools_context = self._tools.to_prompt_context()

        return f"""# StrategyAgent - Motor Cortex (Hyperparameter Control)

## Your Role
{self.AGENT_ROLE}

## Available Tools
{tools_context}

## Output Rules
- When asked for structured output, respond with valid JSON only.
- Use numeric deltas (not absolute rates) unless explicitly asked.
- Prefer small changes and explain reasoning briefly in the "reasoning" field when included in the schema.
"""

    async def analyze_convergence(self, fitness_history: List[float]) -> Dict[str, Any]:
        """
        Analizza fitness_history e ritorna trend/velocity/stall/recommendation.

        Returns keys:
        - trend: str
        - velocity: float
        - stall_detected: bool
        - recommendation: str
        """
        if not fitness_history or len(fitness_history) < 2:
            return {
                "trend": "insufficient_data",
                "velocity": 0.0,
                "stall_detected": False,
                "recommendation": "collect_more_data",
            }

        window = min(12, len(fitness_history) - 1)
        recent = fitness_history[-(window + 1) :]
        deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]

        velocity = float(sum(deltas) / max(1, len(deltas)))
        start = float(recent[0])
        end = float(recent[-1])
        span = float(end - start)

        scale = max(1e-9, float(sum(abs(x) for x in recent) / max(1, len(recent))))
        eps = max(1e-9, 0.002 * scale)

        if span > eps:
            trend = "improving"
        elif span < -eps:
            trend = "regressing"
        else:
            trend = "stagnating"

        stall_threshold = max(1e-9, 0.001 * scale)
        stall_detected = abs(span) < stall_threshold and (len(recent) >= 6)

        if stall_detected:
            recommendation = "increase_exploration"
        elif trend == "regressing":
            recommendation = "reduce_disruption"
        elif trend == "improving":
            recommendation = "keep_or_slightly_exploit"
        else:
            recommendation = "minor_adjustments"

        return {
            "trend": trend,
            "velocity": velocity,
            "stall_detected": bool(stall_detected),
            "recommendation": recommendation,
        }

    async def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggerisce delta per hyperparameters.

        Input context (expected keys):
        - generation: int
        - stall_count: int
        - diversity: float (0-1 recommended)
        - mutation_rate: float (0-1)

        Returns keys:
        - mutation_rate_delta: float
        - crossover_rate_delta: float
        - inject_diversity: bool
        """
        generation = int(context.get("generation", 0) or 0)
        stall_count = int(context.get("stall_count", 0) or 0)
        diversity = context.get("diversity", None)
        mutation_rate = float(context.get("mutation_rate", 0.1) or 0.1)

        heuristic = self._heuristic_hyperparameter_deltas(
            generation=generation,
            stall_count=stall_count,
            diversity=diversity,
            mutation_rate=mutation_rate,
        )

        schema = {
            "type": "object",
            "properties": {
                "mutation_rate_delta": {"type": "number"},
                "crossover_rate_delta": {"type": "number"},
                "inject_diversity": {"type": "boolean"},
            },
            "required": ["mutation_rate_delta", "crossover_rate_delta", "inject_diversity"],
            "additionalProperties": True,
        }

        prompt = f"""## Evolution State
Generation: {generation}
Stall count: {stall_count}
Diversity: {diversity}
Mutation rate (current): {mutation_rate}

## Baseline (heuristic) suggestion
{heuristic}

## Task
Return JSON with keys:
- mutation_rate_delta (float)
- crossover_rate_delta (float)
- inject_diversity (bool)

Constraints:
- Keep deltas small unless stall_count is high.
- mutation_rate should remain in [0.01, 0.90] after applying delta.
- crossover_rate_delta should generally be within [-0.20, 0.20].
"""

        try:
            llm = await self.generate_json(prompt, schema=schema)
            result = self._validate_hyperparameter_deltas(
                llm,
                current_mutation_rate=mutation_rate,
            )
            return result
        except Exception as e:
            logger.warning(f"{self.AGENT_NAME} falling back to heuristic deltas: {e}")
            return heuristic

    async def should_invoke_llm_operator(self, context: Dict[str, Any]) -> bool:
        """
        Decide quando usare LLM per mutation/crossover vs operatori standard.
        """
        generation = int(context.get("generation", 0) or 0)
        stall_count = int(context.get("stall_count", 0) or 0)
        diversity = context.get("diversity", None)

        if stall_count >= 2:
            return True

        if isinstance(diversity, (int, float)) and float(diversity) < 0.15:
            return True

        if generation > 0 and generation % 10 == 0:
            return True

        return False

    async def process(
        self,
        input_data: Any,
        context: Dict[str, Any] = None,
    ) -> AgentResponse:
        """
        Router per i task dello StrategyAgent.

        input_data deve essere un dict con campo "type":
        - {"type": "analyze"} -> analyze_convergence
        - {"type": "suggest"} -> suggest_hyperparameters
        - {"type": "should_llm"} -> should_invoke_llm_operator
        """
        context = context or {}

        if not isinstance(input_data, dict):
            return AgentResponse(
                success=False,
                reasoning="StrategyAgent expects dict input_data with a 'type' field",
                confidence=0.0,
            )

        request_type = input_data.get("type")

        try:
            if request_type == "analyze":
                history = input_data.get("fitness_history", context.get("fitness_history", []))
                result = await self.analyze_convergence(history)
                return AgentResponse(success=True, result=result, confidence=0.75)

            if request_type == "suggest":
                hp_context = input_data.get("context", context)
                result = await self.suggest_hyperparameters(hp_context)
                return AgentResponse(success=True, result=result, confidence=0.75)

            if request_type == "should_llm":
                op_context = input_data.get("context", context)
                result = await self.should_invoke_llm_operator(op_context)
                return AgentResponse(success=True, result=result, confidence=0.7)

            return AgentResponse(
                success=False,
                reasoning=f"Unknown request type: {request_type}",
                confidence=0.0,
            )
        except Exception as e:
            logger.error(f"{self.AGENT_NAME} process failed: {e}")
            return AgentResponse(
                success=False,
                reasoning=f"StrategyAgent failed: {e}",
                confidence=0.0,
            )

    def _heuristic_hyperparameter_deltas(
        self,
        generation: int,
        stall_count: int,
        diversity: Optional[float],
        mutation_rate: float,
    ) -> Dict[str, Any]:
        """
        Heuristic fallback for deltas without LLM.
        """
        mutation_delta = 0.0
        crossover_delta = 0.0
        inject = False

        if generation < 10:
            crossover_delta += 0.03

        if stall_count >= 1:
            mutation_delta += 0.02

        if stall_count >= 3:
            mutation_delta += 0.04 + 0.01 * min(10, stall_count - 3)
            crossover_delta -= 0.02
            inject = True

        if isinstance(diversity, (int, float)):
            d = float(diversity)
            if d < 0.20:
                mutation_delta += 0.05
                crossover_delta -= 0.03
                inject = True
            elif d > 0.65 and stall_count > 0:
                crossover_delta += 0.03

        mutation_delta = self._clamp_delta_to_bounds(
            current=mutation_rate,
            delta=mutation_delta,
            min_value=0.01,
            max_value=0.90,
        )

        crossover_delta = float(max(-0.20, min(0.20, crossover_delta)))

        return {
            "mutation_rate_delta": float(mutation_delta),
            "crossover_rate_delta": float(crossover_delta),
            "inject_diversity": bool(inject),
        }

    def _validate_hyperparameter_deltas(
        self,
        data: Dict[str, Any],
        current_mutation_rate: float,
    ) -> Dict[str, Any]:
        """
        Normalizza/clampa output LLM per deltas.
        """
        mutation_delta = float(data.get("mutation_rate_delta", 0.0))
        crossover_delta = float(data.get("crossover_rate_delta", 0.0))
        inject = bool(data.get("inject_diversity", False))

        mutation_delta = self._clamp_delta_to_bounds(
            current=current_mutation_rate,
            delta=mutation_delta,
            min_value=0.01,
            max_value=0.90,
        )

        if math.isnan(crossover_delta) or math.isinf(crossover_delta):
            crossover_delta = 0.0
        crossover_delta = float(max(-0.20, min(0.20, crossover_delta)))

        return {
            "mutation_rate_delta": float(mutation_delta),
            "crossover_rate_delta": float(crossover_delta),
            "inject_diversity": bool(inject),
        }

    def _clamp_delta_to_bounds(
        self,
        current: float,
        delta: float,
        min_value: float,
        max_value: float,
    ) -> float:
        """
        Clamps delta so that current+delta stays within [min_value, max_value].
        """
        if math.isnan(delta) or math.isinf(delta):
            return 0.0

        target = current + delta
        if target < min_value:
            return float(min_value - current)
        if target > max_value:
            return float(max_value - current)
        return float(delta)

