# 🧬 Agnostic LLM-Evolution Architecture

## La Metafora: Sistema Nervoso

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HUMAN IN THE LOOP                                  │
│                    "Voglio una bowl A432Hz, bronzo, 20cm"                    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CORTECCIA PREFRONTALE                                 │
│                      OrchestratorAgent (LLM 32B)                             │
│                                                                              │
│   • Riceve descrizione umana                                                 │
│   • Pianifica strategia globale                                              │
│   • Istruisce agenti specializzati                                           │
│   • Decide quando chiedere tool/codice                                       │
│   • Tiene contesto lungo termine                                             │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   AREA MOTORIA    │    │   AREA SENSORIALE │    │   AREA MEMORIA    │
│  StrategyAgent    │    │    AnalysisAgent  │    │     RAGAgent      │
│     (LLM 14B)     │    │      (LLM 7B)     │    │    (Embeddings)   │
│                   │    │                   │    │                   │
│ • Mutation rates  │    │ • Fitness interp  │    │ • Paper retrieval │
│ • Crossover type  │    │ • Anomaly detect  │    │ • LTM distill     │
│ • When to explore │    │ • Mode analysis   │    │ • Context inject  │
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MIDOLLO SPINALE                                   │
│                    EvolutionCoordinator (Python)                             │
│                                                                              │
│   • Traduce decisioni LLM → parametri GA                                     │
│   • Gestisce popolazione                                                     │
│   • Applica operatori genetici                                               │
│   • Chiama fitness function                                                  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MUSCOLI                                         │
│                     Neural Networks + Simulators                             │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│   │    RDNN     │   │   FEM Sim   │   │ CNC Virtual │   │  Surrogate  │     │
│   │  Trajectory │   │  scikit-fem │   │    Tool     │   │    Model    │     │
│   │  Predictor  │   │   JAX/GPU   │   │   G-code    │   │   PyTorch   │     │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                              │
│   Rigidi, veloci, specializzati - i "muscoli" che eseguono                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Principio Fondamentale: AGNOSTICISMO

L'architettura **NON SA** cosa sta ottimizzando. Sa solo:

1. **Genome**: Un dizionario di parametri con bounds e tipi
2. **Fitness**: Una funzione che ritorna metriche numeriche
3. **Constraints**: Regole che il genome deve rispettare
4. **Tools**: Risorse disponibili (FEM, CNC, API, etc.)

### Esempio: Come l'Orchestrator "Impara" il Dominio

```python
# L'umano dice:
human_request = """
Voglio ottimizzare una singing bowl tibetana.
- Diametro: 18-25 cm
- Materiale: bronzo (8% stagno)
- Frequenza fondamentale: 432 Hz (A4)
- Seconda armonica: quinta giusta (648 Hz)
- Produzione: 3D printing in cera persa
- Priorità: purezza tono > sustain > peso
"""

# L'Orchestrator genera:
domain_spec = {
    "name": "tibetan_bowl",
    "genome_schema": {
        "diameter_mm": {"type": "float", "min": 180, "max": 250},
        "wall_thickness_mm": {"type": "float", "min": 2, "max": 6},
        "wall_angle_deg": {"type": "float", "min": 0, "max": 20},
        "bottom_thickness_mm": {"type": "float", "min": 3, "max": 10},
        "bottom_curvature": {"type": "float", "min": 0, "max": 0.15},
        "rim_profile": {"type": "categorical", "values": ["flat", "rounded", "thickened"]},
    },
    "objectives": [
        {"name": "freq_error_fundamental", "target": 432.0, "weight": 0.4, "minimize": True},
        {"name": "freq_error_fifth", "target": 648.0, "weight": 0.3, "minimize": True},
        {"name": "sustain_seconds", "weight": 0.2, "minimize": False},
        {"name": "mass_kg", "weight": 0.1, "minimize": True},
    ],
    "constraints": [
        {"type": "structural", "rule": "wall_thickness >= 2mm everywhere"},
        {"type": "manufacturing", "rule": "no undercuts for 3D print"},
        {"type": "physics", "rule": "eigenfrequencies must be real"},
    ],
    "required_tools": ["fem_modal_analysis", "mesh_generator", "gcode_export"],
    "missing_tools": ["acoustic_radiation"],  # Chiederà di creare!
}
```

---

## 📁 Struttura Branch Definitiva

```
llm-evolution/
├── src/
│   ├── core/                        # AGNOSTIC EVOLUTION CORE
│   │   ├── __init__.py
│   │   ├── genome.py                # Protocol + BaseGenome
│   │   ├── population.py            # Population management
│   │   ├── operators.py             # Mutation, Crossover, Selection
│   │   ├── fitness.py               # FitnessResult protocol
│   │   ├── evolution.py             # Main GA loop
│   │   └── coordinator.py           # Midollo: traduce LLM → GA
│   │
│   ├── agents/                      # SISTEMA NERVOSO (LLM)
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent con LLM client
│   │   ├── orchestrator.py          # Corteccia prefrontale (32B)
│   │   ├── strategy.py              # Area motoria (14B)
│   │   ├── analysis.py              # Area sensoriale (7B)
│   │   ├── rag.py                   # Area memoria (embeddings)
│   │   └── tool_request.py          # Richiede tool/codice a Cursor
│   │
│   ├── memory/                      # MEMORIA LUNGO TERMINE
│   │   ├── __init__.py
│   │   ├── rdnn.py                  # Recurrent trajectory predictor
│   │   ├── ltm.py                   # Long-term distillation
│   │   └── stm.py                   # Short-term working memory
│   │
│   ├── tools/                       # MUSCOLI (Neural + Simulators)
│   │   ├── __init__.py
│   │   ├── registry.py              # Tool discovery & registration
│   │   ├── fem/                     # FEM simulators
│   │   │   ├── __init__.py
│   │   │   ├── scikit_fem.py
│   │   │   └── modal_analysis.py
│   │   ├── manufacturing/           # Production tools
│   │   │   ├── __init__.py
│   │   │   ├── cnc_virtual.py       # G-code generation
│   │   │   └── slicer.py            # 3D print slicing
│   │   └── surrogate/               # Neural surrogates
│   │       ├── __init__.py
│   │       └── fitness_predictor.py
│   │
│   ├── llm/                         # LLM INFRASTRUCTURE
│   │   ├── __init__.py
│   │   ├── client.py                # Ollama client (locale)
│   │   ├── prompts/                 # Jinja2 templates
│   │   │   ├── orchestrator.j2
│   │   │   ├── strategy.j2
│   │   │   ├── analysis.j2
│   │   │   └── tool_request.j2
│   │   ├── parser.py                # JSON/structured output
│   │   └── cache.py                 # Response caching
│   │
│   ├── knowledge/                   # KNOWLEDGE BASE
│   │   ├── __init__.py
│   │   ├── surrealdb.py             # SurrealDB client
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── arxiv.py             # ArXiv fetcher
│   │   │   ├── llm4ec.py            # LLM4EC papers
│   │   │   └── domain_specific.py   # Per-domain papers
│   │   ├── embeddings.py            # Vector generation
│   │   └── mcp_server.py            # MCP server per questo progetto
│   │
│   └── domains/                     # DOMAIN EXAMPLES (non core!)
│       ├── __init__.py
│       ├── tibetan_bowl.py          # Bowl A432
│       ├── vibroacoustic_plate.py   # DML therapy
│       └── benchmark.py             # Rastrigin, TSP
│
├── tests/
│   ├── test_core/
│   ├── test_agents/
│   ├── test_tools/
│   └── test_integration/
│
├── config/
│   ├── models.yaml                  # LLM model configs
│   ├── tools.yaml                   # Available tools
│   └── defaults.yaml                # Default parameters
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🔧 FASE 1: Fondamenta (Settimana 1)

### Task 1.1: Setup Branch e Struttura
**Tempo**: 2h
**Output**: Branch `llm-evolution` con folder structure

```bash
git checkout -b llm-evolution
mkdir -p src/{core,agents,memory,tools,llm,knowledge,domains}
mkdir -p src/tools/{fem,manufacturing,surrogate}
mkdir -p src/llm/prompts
mkdir -p src/knowledge/ingestion
mkdir -p tests/{test_core,test_agents,test_tools,test_integration}
mkdir -p config
touch src/__init__.py src/core/__init__.py # etc...
```

### Task 1.2: Genome Protocol Agnostico
**Tempo**: 3h
**Output**: `src/core/genome.py`

```python
# src/core/genome.py
"""
Agnostic Genome Protocol - Il genome NON sa cosa rappresenta.
"""

from typing import Protocol, Dict, Any, List, TypeVar, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

@runtime_checkable
class Genome(Protocol):
    """Protocol che ogni genome deve implementare."""
    
    def to_vector(self) -> np.ndarray:
        """Converte a vettore numerico per operatori GA."""
        ...
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, schema: 'GenomeSchema') -> 'Genome':
        """Ricostruisce da vettore."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza per LLM/storage."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Deserializza."""
        ...
    
    def validate(self) -> List[str]:
        """Ritorna lista errori (vuota se valido)."""
        ...


@dataclass
class GeneSpec:
    """Specifica di un singolo gene."""
    name: str
    type: str  # "float", "int", "categorical", "bool"
    
    # Per numerici
    min_value: float = None
    max_value: float = None
    
    # Per categorici
    categories: List[str] = None
    
    # Metadata per LLM
    description: str = ""
    unit: str = ""
    
    def random_value(self, rng: np.random.Generator = None):
        """Genera valore random valido."""
        rng = rng or np.random.default_rng()
        
        if self.type == "float":
            return rng.uniform(self.min_value, self.max_value)
        elif self.type == "int":
            return rng.integers(self.min_value, self.max_value + 1)
        elif self.type == "categorical":
            return rng.choice(self.categories)
        elif self.type == "bool":
            return rng.choice([True, False])


@dataclass
class GenomeSchema:
    """Schema completo del genome - generato da LLM orchestrator."""
    name: str
    genes: List[GeneSpec] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Per LLM context
    description: str = ""
    domain_knowledge: str = ""  # Injected from RAG
    
    @property
    def dimension(self) -> int:
        """Dimensione vettore numerico."""
        return len(self.genes)
    
    def to_prompt_context(self) -> str:
        """Genera context per prompt LLM."""
        lines = [f"# Genome: {self.name}", f"{self.description}", "", "## Genes:"]
        for g in self.genes:
            if g.type in ("float", "int"):
                lines.append(f"- {g.name}: {g.type} [{g.min_value}, {g.max_value}] {g.unit} - {g.description}")
            else:
                lines.append(f"- {g.name}: {g.type} {g.categories} - {g.description}")
        return "\n".join(lines)


class DictGenome:
    """Implementazione generica basata su dizionario."""
    
    def __init__(self, data: Dict[str, Any], schema: GenomeSchema):
        self._data = data
        self._schema = schema
    
    def to_vector(self) -> np.ndarray:
        """Normalizza tutto in [0,1] per GA."""
        vector = []
        for gene in self._schema.genes:
            val = self._data[gene.name]
            if gene.type == "float":
                norm = (val - gene.min_value) / (gene.max_value - gene.min_value)
            elif gene.type == "int":
                norm = (val - gene.min_value) / (gene.max_value - gene.min_value)
            elif gene.type == "categorical":
                norm = gene.categories.index(val) / (len(gene.categories) - 1)
            elif gene.type == "bool":
                norm = 1.0 if val else 0.0
            vector.append(norm)
        return np.array(vector)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, schema: GenomeSchema) -> 'DictGenome':
        """Denormalizza da [0,1]."""
        data = {}
        for i, gene in enumerate(schema.genes):
            norm = vector[i]
            if gene.type == "float":
                val = gene.min_value + norm * (gene.max_value - gene.min_value)
            elif gene.type == "int":
                val = int(round(gene.min_value + norm * (gene.max_value - gene.min_value)))
            elif gene.type == "categorical":
                idx = int(round(norm * (len(gene.categories) - 1)))
                val = gene.categories[idx]
            elif gene.type == "bool":
                val = norm > 0.5
            data[gene.name] = val
        return cls(data, schema)
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], schema: GenomeSchema) -> 'DictGenome':
        return cls(data, schema)
    
    def validate(self) -> List[str]:
        errors = []
        for gene in self._schema.genes:
            if gene.name not in self._data:
                errors.append(f"Missing gene: {gene.name}")
                continue
            val = self._data[gene.name]
            if gene.type in ("float", "int"):
                if val < gene.min_value or val > gene.max_value:
                    errors.append(f"{gene.name}={val} out of bounds [{gene.min_value}, {gene.max_value}]")
        return errors
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
    
    def __repr__(self):
        return f"DictGenome({self._data})"


def random_genome(schema: GenomeSchema, rng: np.random.Generator = None) -> DictGenome:
    """Genera genome random secondo schema."""
    data = {}
    for gene in schema.genes:
        data[gene.name] = gene.random_value(rng)
    return DictGenome(data, schema)
```

### Task 1.3: Ollama Client
**Tempo**: 2h
**Output**: `src/llm/client.py`

```python
# src/llm/client.py
"""
Ollama LLM Client - Solo locale, niente cloud.
"""

import asyncio
import httpx
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configurazione modello LLM."""
    name: str                    # es. "deepseek-r1:14b"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    context_length: int = 32768  # Per modelli grandi


class OllamaClient:
    """Client per Ollama API (locale)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        cache_dir: Optional[Path] = None,
    ):
        self.base_url = base_url
        self.cache_dir = cache_dir or Path.home() / ".cache" / "llm-evolution"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout
    
    async def generate(
        self,
        prompt: str,
        model: ModelConfig,
        system: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Genera risposta da LLM."""
        
        # Check cache
        if use_cache:
            cache_key = self._cache_key(prompt, model.name, system)
            cached = self._load_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                return cached
        
        # Build request
        payload = {
            "model": model.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
                "top_p": model.top_p,
            }
        }
        
        if system:
            payload["system"] = system
        
        # Call Ollama
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "")
            
            # Cache result
            if use_cache and text:
                self._save_cache(cache_key, text)
            
            return text
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def generate_json(
        self,
        prompt: str,
        model: ModelConfig,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Genera e parsa JSON."""
        
        # Append JSON instruction
        json_prompt = prompt + "\n\nRespond ONLY with valid JSON, no explanation."
        
        response = await self.generate(json_prompt, model, system)
        
        # Extract JSON
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Cannot parse JSON from: {response[:200]}...")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: ModelConfig,
    ) -> str:
        """Chat multi-turn."""
        
        payload = {
            "model": model.name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
            }
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
    
    async def list_models(self) -> List[str]:
        """Lista modelli disponibili."""
        response = await self._client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    
    async def ensure_model(self, model_name: str) -> bool:
        """Verifica che il modello sia disponibile."""
        models = await self.list_models()
        return model_name in models
    
    def _cache_key(self, prompt: str, model: str, system: Optional[str]) -> str:
        """Genera chiave cache."""
        content = f"{model}:{system or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self, key: str) -> Optional[str]:
        """Carica da cache."""
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text()
        return None
    
    def _save_cache(self, key: str, content: str):
        """Salva in cache."""
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(content)
    
    async def close(self):
        """Chiudi client."""
        await self._client.aclose()


# Configurazioni modelli predefinite
MODELS = {
    "orchestrator": ModelConfig(
        name="deepseek-r1:14b",  # O qwen2.5:32b se hai RAM
        temperature=0.3,        # Più deterministico per planning
        max_tokens=8192,
        context_length=65536,
    ),
    "strategy": ModelConfig(
        name="qwen2.5-coder:14b",
        temperature=0.5,
        max_tokens=4096,
    ),
    "analysis": ModelConfig(
        name="llama3.2:3b",      # Veloce per analisi frequente
        temperature=0.2,
        max_tokens=2048,
    ),
    "fast": ModelConfig(
        name="qwen2.5:3b",       # Ultra veloce per decisioni semplici
        temperature=0.1,
        max_tokens=512,
    ),
}
```

### Task 1.4: Tool Registry
**Tempo**: 2h
**Output**: `src/tools/registry.py`

```python
# src/tools/registry.py
"""
Tool Registry - Muscoli disponibili per l'evoluzione.
L'Orchestrator può chiedere nuovi tool se non esistono.
"""

from typing import Protocol, Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
import inspect
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specifica di un tool."""
    name: str
    description: str
    function: Callable
    
    # Input/output schema (per LLM)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    domain: str = "general"  # "fem", "manufacturing", "surrogate", etc.
    cost: str = "medium"     # "low", "medium", "high" (tempo/risorse)
    
    def to_prompt_description(self) -> str:
        """Descrizione per prompt LLM."""
        params = ", ".join(self.input_schema.keys()) if self.input_schema else "none"
        returns = ", ".join(self.output_schema.keys()) if self.output_schema else "result"
        return f"- **{self.name}**({params}) → {returns}: {self.description}"


class ToolRegistry:
    """Registry di tutti i tool disponibili."""
    
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._missing_requests: List[str] = []  # Tool richiesti ma non esistenti
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        domain: str = "general",
        cost: str = "medium",
    ):
        """Registra un tool."""
        spec = ToolSpec(
            name=name,
            description=description,
            function=function,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            domain=domain,
            cost=cost,
        )
        self._tools[name] = spec
        logger.info(f"Registered tool: {name}")
    
    def get(self, name: str) -> Optional[ToolSpec]:
        """Ottieni tool per nome."""
        return self._tools.get(name)
    
    def execute(self, name: str, **kwargs) -> Any:
        """Esegui tool."""
        tool = self._tools.get(name)
        if not tool:
            self._missing_requests.append(name)
            raise ValueError(f"Tool '{name}' not found. Request logged.")
        return tool.function(**kwargs)
    
    async def execute_async(self, name: str, **kwargs) -> Any:
        """Esegui tool async."""
        tool = self._tools.get(name)
        if not tool:
            self._missing_requests.append(name)
            raise ValueError(f"Tool '{name}' not found. Request logged.")
        
        if inspect.iscoroutinefunction(tool.function):
            return await tool.function(**kwargs)
        return tool.function(**kwargs)
    
    def list_by_domain(self, domain: str) -> List[ToolSpec]:
        """Lista tool per dominio."""
        return [t for t in self._tools.values() if t.domain == domain]
    
    def list_all(self) -> List[ToolSpec]:
        """Lista tutti i tool."""
        return list(self._tools.values())
    
    def get_missing_requests(self) -> List[str]:
        """Tool richiesti ma non disponibili."""
        return list(set(self._missing_requests))
    
    def to_prompt_context(self) -> str:
        """Genera descrizione tool per LLM."""
        lines = ["# Available Tools", ""]
        
        by_domain = {}
        for tool in self._tools.values():
            by_domain.setdefault(tool.domain, []).append(tool)
        
        for domain, tools in sorted(by_domain.items()):
            lines.append(f"## {domain.title()}")
            for tool in tools:
                lines.append(tool.to_prompt_description())
            lines.append("")
        
        return "\n".join(lines)


# Singleton globale
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Ottieni registry globale."""
    return _registry


def tool(
    name: str = None,
    description: str = "",
    domain: str = "general",
    cost: str = "medium",
):
    """Decorator per registrare funzioni come tool."""
    def decorator(func):
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        # Auto-extract schema from type hints
        hints = func.__annotations__
        input_schema = {k: str(v) for k, v in hints.items() if k != "return"}
        output_schema = {"return": str(hints.get("return", "Any"))}
        
        _registry.register(
            name=tool_name,
            description=tool_desc,
            function=func,
            input_schema=input_schema,
            output_schema=output_schema,
            domain=domain,
            cost=cost,
        )
        return func
    return decorator
```

---

## 🔧 FASE 2: Agenti (Settimana 2)

### Task 2.1: BaseAgent + OrchestratorAgent
**Tempo**: 4h
**Output**: `src/agents/base.py`, `src/agents/orchestrator.py`

### Task 2.2: StrategyAgent + AnalysisAgent
**Tempo**: 3h
**Output**: `src/agents/strategy.py`, `src/agents/analysis.py`

### Task 2.3: RAGAgent + Knowledge Integration
**Tempo**: 3h
**Output**: `src/agents/rag.py`

### Task 2.4: ToolRequestAgent
**Tempo**: 2h
**Output**: `src/agents/tool_request.py`
**Funzione**: Quando manca un tool, genera richiesta per Cursor/umano

---

## 🔧 FASE 3: Knowledge (Settimana 3)

### Task 3.1: SurrealDB Client + MCP Server
**Tempo**: 3h
**Output**: `src/knowledge/surrealdb.py`, `src/knowledge/mcp_server.py`
**Copia da**: `binaural_golden/src/utils/surrealdb_mcp_server.py`

### Task 3.2: Ingestion LLM4EC Papers
**Tempo**: 4h
**Output**: `src/knowledge/ingestion/llm4ec.py`
**Basato su**: `Mirror7/knowledge_ingest_mirror7.py`

### Task 3.3: Embedding Pipeline
**Tempo**: 2h
**Output**: `src/knowledge/embeddings.py`
**Usa**: `nomic-embed-text` (locale)

---

## 🔧 FASE 4: Integrazione (Settimana 4)

### Task 4.1: Evolution Coordinator
**Tempo**: 4h
**Output**: `src/core/coordinator.py`
**Funzione**: Midollo spinale che traduce LLM → GA

### Task 4.2: Main Evolution Loop
**Tempo**: 3h
**Output**: `src/core/evolution.py`

### Task 4.3: Domain Example (Bowl)
**Tempo**: 2h
**Output**: `src/domains/tibetan_bowl.py`

### Task 4.4: Test Suite
**Tempo**: 3h
**Output**: `tests/`

---

## 📋 CHECKLIST SENIOR ENGINEER

### Pre-Development
- [ ] Branch creato e struttura cartelle
- [ ] `.pre-commit` hooks configurati
- [ ] `pyproject.toml` con dependencies
- [ ] CI/CD basic (test on push)

### Code Quality
- [ ] Type hints ovunque
- [ ] Docstrings complete
- [ ] Logging strutturato
- [ ] Error handling robusto

### Testing
- [ ] Unit tests per ogni modulo
- [ ] Integration tests agents
- [ ] Benchmark vs vanilla GA

### Documentation
- [ ] README con quick start
- [ ] Architecture diagram
- [ ] API reference

---

## 🎯 SUCCESS CRITERIA

1. **Agnosticismo**: Posso ottimizzare bowl, piatto, o Rastrigin senza cambiare core
2. **LLM Integration**: Orchestrator genera schema genome da descrizione naturale
3. **Tool Discovery**: Sistema richiede tool mancanti
4. **Knowledge RAG**: Papers iniettati nel context degli agenti
5. **Performance**: Convergenza migliore del 20% vs vanilla GA su benchmark

---

*Piano creato: 2025-01-09*
*Senior Engineer: Alessio Ivoy Cazzaniga*
