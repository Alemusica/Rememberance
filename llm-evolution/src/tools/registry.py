"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TOOL REGISTRY                                       ║
║                                                                              ║
║   Registry dei "muscoli" disponibili per l'evoluzione.                       ║
║   Tool = qualsiasi funzione che trasforma input → output:                    ║
║   • FEM simulator (genome → frequenze modali)                                ║
║   • CNC virtual (genome → G-code)                                            ║
║   • Neural surrogate (genome → fitness prediction)                           ║
║   • Qualsiasi altra cosa                                                     ║
║                                                                              ║
║   Se un tool richiesto non esiste, viene loggato e può essere generato       ║
║   dal ToolRequestAgent (che chiede a Cursor/umano di crearlo).               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Protocol, Dict, Any, Callable, List, Optional, Union
from dataclasses import dataclass, field
import inspect
import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL SPECIFICATION
# =============================================================================

@dataclass
class ToolSpec:
    """
    Specifica completa di un tool.
    
    Contiene tutto ciò che serve per:
    1. Eseguire il tool
    2. Descriverlo all'LLM
    3. Validare input/output
    """
    name: str
    description: str
    function: Callable
    
    # Schema per LLM
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    domain: str = "general"       # fem, manufacturing, surrogate, etc.
    cost: str = "medium"          # low, medium, high (tempo/risorse)
    is_async: bool = False        # Se la funzione è async
    requires_gpu: bool = False    # Se richiede GPU
    
    # Examples per few-shot
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_prompt_description(self) -> str:
        """Genera descrizione per prompt LLM."""
        # Input params
        if self.input_schema:
            params = []
            for name, spec in self.input_schema.items():
                if isinstance(spec, dict):
                    type_str = spec.get("type", "any")
                    desc = spec.get("description", "")
                    params.append(f"{name}: {type_str}")
                else:
                    params.append(f"{name}: {spec}")
            params_str = ", ".join(params)
        else:
            params_str = "..."
        
        # Output
        if self.output_schema:
            returns = ", ".join(self.output_schema.keys())
        else:
            returns = "result"
        
        # Cost indicator
        cost_emoji = {"low": "⚡", "medium": "⏱️", "high": "🐢"}.get(self.cost, "")
        
        return f"- **{self.name}**({params_str}) → {returns} {cost_emoji}\n  {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza specifica (senza function)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "domain": self.domain,
            "cost": self.cost,
            "is_async": self.is_async,
            "requires_gpu": self.requires_gpu,
        }


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """
    Registry centrale di tutti i tool disponibili.
    
    Responsabilità:
    1. Registrazione tool (via decorator o metodo)
    2. Lookup per nome/dominio
    3. Esecuzione (sync e async)
    4. Tracking tool mancanti
    5. Generazione context per LLM
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._missing_requests: List[Dict[str, Any]] = []
        self._execution_stats: Dict[str, Dict] = {}
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        domain: str = "general",
        cost: str = "medium",
        examples: List[Dict] = None,
    ) -> ToolSpec:
        """
        Registra un tool.
        
        Args:
            name: Nome univoco
            description: Descrizione per LLM
            function: Funzione da eseguire
            input_schema: Schema input {param: type_or_spec}
            output_schema: Schema output {field: type}
            domain: Categoria (fem, manufacturing, etc.)
            cost: Costo esecuzione (low, medium, high)
            examples: Esempi per few-shot
        
        Returns:
            ToolSpec creato
        """
        is_async = asyncio.iscoroutinefunction(function)
        
        spec = ToolSpec(
            name=name,
            description=description,
            function=function,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            domain=domain,
            cost=cost,
            is_async=is_async,
            examples=examples or [],
        )
        
        self._tools[name] = spec
        self._execution_stats[name] = {
            "calls": 0,
            "errors": 0,
            "total_time_ms": 0,
        }
        
        logger.info(f"Registered tool: {name} (domain={domain}, async={is_async})")
        return spec
    
    def get(self, name: str) -> Optional[ToolSpec]:
        """Ottieni tool per nome."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Verifica se tool esiste."""
        return name in self._tools
    
    def execute(self, name: str, **kwargs) -> Any:
        """
        Esegui tool sincronamente.
        
        Raises:
            ValueError: Se tool non esiste
            Exception: Se esecuzione fallisce
        """
        tool = self._tools.get(name)
        
        if not tool:
            self._log_missing(name, kwargs)
            raise ValueError(f"Tool '{name}' not found. Request logged for creation.")
        
        import time
        start = time.time()
        
        try:
            if tool.is_async:
                # Wrap async in sync
                result = asyncio.get_event_loop().run_until_complete(
                    tool.function(**kwargs)
                )
            else:
                result = tool.function(**kwargs)
            
            elapsed = (time.time() - start) * 1000
            self._execution_stats[name]["calls"] += 1
            self._execution_stats[name]["total_time_ms"] += elapsed
            
            return result
            
        except Exception as e:
            self._execution_stats[name]["errors"] += 1
            logger.error(f"Tool '{name}' failed: {e}")
            raise
    
    async def execute_async(self, name: str, **kwargs) -> Any:
        """
        Esegui tool asincronamente.
        """
        tool = self._tools.get(name)
        
        if not tool:
            self._log_missing(name, kwargs)
            raise ValueError(f"Tool '{name}' not found. Request logged for creation.")
        
        import time
        start = time.time()
        
        try:
            if tool.is_async:
                result = await tool.function(**kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: tool.function(**kwargs)
                )
            
            elapsed = (time.time() - start) * 1000
            self._execution_stats[name]["calls"] += 1
            self._execution_stats[name]["total_time_ms"] += elapsed
            
            return result
            
        except Exception as e:
            self._execution_stats[name]["errors"] += 1
            logger.error(f"Tool '{name}' failed: {e}")
            raise
    
    def list_by_domain(self, domain: str) -> List[ToolSpec]:
        """Lista tool per dominio."""
        return [t for t in self._tools.values() if t.domain == domain]
    
    def list_all(self) -> List[ToolSpec]:
        """Lista tutti i tool."""
        return list(self._tools.values())
    
    def list_domains(self) -> List[str]:
        """Lista tutti i domini."""
        return list(set(t.domain for t in self._tools.values()))
    
    def get_missing_requests(self) -> List[Dict[str, Any]]:
        """Tool richiesti ma non disponibili (per ToolRequestAgent)."""
        return list(self._missing_requests)
    
    def clear_missing_requests(self):
        """Pulisci lista missing."""
        self._missing_requests.clear()
    
    def _log_missing(self, name: str, kwargs: Dict):
        """Log richiesta tool mancante."""
        request = {
            "tool_name": name,
            "requested_params": list(kwargs.keys()),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        
        # Avoid duplicates
        if not any(r["tool_name"] == name for r in self._missing_requests):
            self._missing_requests.append(request)
            logger.warning(f"Missing tool requested: {name}")
    
    def to_prompt_context(self, domains: List[str] = None) -> str:
        """
        Genera descrizione completa per prompt LLM.
        
        Args:
            domains: Filtra per domini (None = tutti)
        """
        lines = ["# Available Tools", ""]
        
        # Group by domain
        by_domain: Dict[str, List[ToolSpec]] = {}
        for tool in self._tools.values():
            if domains and tool.domain not in domains:
                continue
            by_domain.setdefault(tool.domain, []).append(tool)
        
        for domain in sorted(by_domain.keys()):
            tools = by_domain[domain]
            lines.append(f"## {domain.replace('_', ' ').title()}")
            lines.append("")
            for tool in sorted(tools, key=lambda t: t.name):
                lines.append(tool.to_prompt_description())
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche esecuzione."""
        return {
            "total_tools": len(self._tools),
            "tools_by_domain": {
                domain: len(tools) 
                for domain, tools in self._group_by_domain().items()
            },
            "missing_requests": len(self._missing_requests),
            "execution_stats": self._execution_stats,
        }
    
    def _group_by_domain(self) -> Dict[str, List[ToolSpec]]:
        result = {}
        for tool in self._tools.values():
            result.setdefault(tool.domain, []).append(tool)
        return result


# =============================================================================
# SINGLETON & DECORATOR
# =============================================================================

_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get singleton registry."""
    return _registry


def tool(
    name: str = None,
    description: str = "",
    domain: str = "general",
    cost: str = "medium",
    input_schema: Dict = None,
    output_schema: Dict = None,
):
    """
    Decorator per registrare funzioni come tool.
    
    Usage:
        @tool(name="my_tool", description="Does something", domain="fem")
        def my_function(param1: float, param2: str) -> Dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Execute {tool_name}"
        
        # Auto-extract schema from type hints if not provided
        hints = getattr(func, "__annotations__", {})
        auto_input = {}
        auto_output = {}
        
        for param, hint in hints.items():
            if param == "return":
                auto_output["result"] = str(hint)
            else:
                auto_input[param] = str(hint)
        
        _registry.register(
            name=tool_name,
            description=tool_desc,
            function=func,
            input_schema=input_schema or auto_input,
            output_schema=output_schema or auto_output,
            domain=domain,
            cost=cost,
        )
        
        return func
    
    return decorator


# =============================================================================
# BUILT-IN TOOLS
# =============================================================================

@tool(
    name="echo",
    description="Echo input for testing",
    domain="debug",
    cost="low",
)
def echo_tool(message: str) -> str:
    """Simply return the input message."""
    return f"Echo: {message}"


@tool(
    name="random_vector",
    description="Generate random vector of given dimension",
    domain="debug",
    cost="low",
)
def random_vector_tool(dimension: int, seed: int = None) -> List[float]:
    """Generate random vector in [0, 1]."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.random(dimension).tolist()
