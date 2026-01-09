"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              BASE AGENT                                       ║
║                                                                              ║
║   Classe base per tutti gli agenti LLM.                                       ║
║   Ogni agente specializzato (Orchestrator, Strategy, Analysis...) eredita.   ║
║                                                                              ║
║   Architettura sistema nervoso:                                              ║
║   • Corteccia prefrontale: Orchestrator (32B, reasoning)                     ║
║   • Area motoria: StrategyAgent (14B, parametri GA)                          ║
║   • Area sensoriale: AnalysisAgent (7B, interpretazione)                     ║
║   • Memoria: RAGAgent (embeddings, paper retrieval)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Optional, Dict, Any, List, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json

from ..llm.client import OllamaClient, ModelConfig, MODELS, get_client
from ..tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# AGENT RESPONSE
# =============================================================================

@dataclass
class AgentResponse(Generic[T]):
    """
    Risposta strutturata da un agente.
    
    Contiene sia il risultato che metadata utili per debugging/logging.
    """
    success: bool
    result: Optional[T] = None
    
    # Reasoning (chain of thought)
    reasoning: str = ""
    
    # Confidence (0-1)
    confidence: float = 0.0
    
    # Actions suggested
    suggested_actions: List[str] = field(default_factory=list)
    
    # Warnings/notes
    warnings: List[str] = field(default_factory=list)
    
    # Tool requests (if agent needs tools not available)
    tool_requests: List[str] = field(default_factory=list)
    
    # Raw LLM response (for debugging)
    raw_response: str = ""
    
    # Token usage
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggested_actions": self.suggested_actions,
            "warnings": self.warnings,
        }


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent(ABC):
    """
    Classe base per agenti LLM.
    
    Ogni agente ha:
    - Un modello LLM associato
    - Accesso alla tool registry
    - System prompt personalizzato
    - Metodi per generazione strutturata
    """
    
    # Class-level defaults (override in subclasses)
    DEFAULT_MODEL: str = "fast"
    AGENT_NAME: str = "BaseAgent"
    AGENT_ROLE: str = "General purpose agent"
    
    def __init__(
        self,
        model: str = None,
        llm_client: OllamaClient = None,
        tool_registry: ToolRegistry = None,
        system_prompt: str = None,
    ):
        """
        Args:
            model: Nome modello o chiave in MODELS
            llm_client: Client LLM (default: singleton)
            tool_registry: Registry tool (default: singleton)
            system_prompt: Override system prompt
        """
        self.model_name = model or self.DEFAULT_MODEL
        self.model_config = MODELS.get(self.model_name, ModelConfig(name=self.model_name))
        
        self._llm = llm_client or get_client()
        self._tools = tool_registry or get_registry()
        
        self._system_prompt = system_prompt or self._build_system_prompt()
        
        # Conversation history for multi-turn
        self._history: List[Dict[str, str]] = []
        
        # Stats
        self.stats = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_tokens": 0,
        }
        
        logger.info(f"Initialized {self.AGENT_NAME} with model {self.model_name}")
    
    def _build_system_prompt(self) -> str:
        """
        Costruisce system prompt per questo agente.
        Override in subclasses per prompt specializzati.
        """
        return f"""You are {self.AGENT_NAME}, a specialized AI agent.

Role: {self.AGENT_ROLE}

You have access to the following tools:
{self._tools.to_prompt_context()}

Guidelines:
1. Think step by step before responding
2. Be precise and concise
3. If you need a tool that doesn't exist, explicitly request it
4. Always provide confidence level (0-1) for your decisions
5. Explain your reasoning

Respond in JSON format when asked for structured output."""
    
    @abstractmethod
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Processa input e ritorna risposta strutturata.
        
        Args:
            input_data: Input specifico per l'agente
            context: Contesto aggiuntivo (stato evoluzione, etc.)
        
        Returns:
            AgentResponse con risultato e metadata
        """
        pass
    
    async def generate(
        self,
        prompt: str,
        use_cache: bool = True,
    ) -> str:
        """
        Genera risposta testo da LLM.
        """
        self.stats["calls"] += 1
        
        response = await self._llm.generate(
            prompt=prompt,
            model=self.model_config,
            system=self._system_prompt,
            use_cache=use_cache,
        )
        
        return response
    
    async def generate_json(
        self,
        prompt: str,
        schema: Dict = None,
    ) -> Dict[str, Any]:
        """
        Genera risposta JSON strutturata.
        """
        self.stats["calls"] += 1
        
        try:
            result = await self._llm.generate_json(
                prompt=prompt,
                model=self.model_config,
                system=self._system_prompt,
                schema=schema,
            )
            self.stats["successes"] += 1
            return result
        except Exception as e:
            self.stats["failures"] += 1
            logger.error(f"{self.AGENT_NAME} JSON generation failed: {e}")
            raise
    
    async def chat(
        self,
        message: str,
        reset_history: bool = False,
    ) -> str:
        """
        Chat multi-turn con history.
        """
        if reset_history:
            self._history.clear()
        
        self._history.append({"role": "user", "content": message})
        
        # Add system message if first
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)
        
        response = await self._llm.chat(messages, self.model_config)
        
        self._history.append({"role": "assistant", "content": response})
        self.stats["calls"] += 1
        
        return response
    
    def reset_history(self):
        """Reset conversation history."""
        self._history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche agente."""
        return {
            "agent": self.AGENT_NAME,
            "model": self.model_name,
            **self.stats,
        }
    
    def inject_context(self, context: str):
        """
        Inietta contesto aggiuntivo nel system prompt.
        Utile per RAG (paper injection).
        """
        self._system_prompt = self._build_system_prompt() + f"\n\n## Additional Context\n{context}"
    
    async def request_tool(self, tool_name: str, description: str) -> Dict[str, Any]:
        """
        Richiede creazione di un tool mancante.
        Ritorna specifica per ToolRequestAgent.
        """
        request = {
            "tool_name": tool_name,
            "description": description,
            "requested_by": self.AGENT_NAME,
            "suggested_input_schema": {},
            "suggested_output_schema": {},
        }
        
        logger.warning(f"{self.AGENT_NAME} requesting tool: {tool_name}")
        return request


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_agent_response(
    raw_response: str,
    expected_fields: List[str] = None,
) -> AgentResponse:
    """
    Parsa risposta LLM in AgentResponse strutturato.
    
    Gestisce sia JSON che testo libero.
    """
    import re
    
    # Try JSON first
    try:
        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            data = json.loads(json_match.group())
            
            return AgentResponse(
                success=data.get("success", True),
                result=data.get("result"),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
                suggested_actions=data.get("suggested_actions", []),
                warnings=data.get("warnings", []),
                tool_requests=data.get("tool_requests", []),
                raw_response=raw_response,
            )
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: extract from text
    confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', raw_response, re.I)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
    
    return AgentResponse(
        success=True,
        result=raw_response,
        reasoning=raw_response,
        confidence=confidence,
        raw_response=raw_response,
    )
