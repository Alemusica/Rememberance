# 🎯 Task Atomici Delegati

> Prompts pronti per Composer/Sonnet 4.5/GPT-4.5
> Riportare output a Opus 4.5 per review

---

## TASK 1: LMX Crossover Semantico

**Modello**: `Claude Sonnet 4.5` (Composer)
**File output**: `llm-evolution/src/agents/crossover.py`
**Stima**: 20 minuti

### PROMPT (copia-incolla)

```
Sei nel repo llm-evolution. Devi SOSTITUIRE il file src/agents/crossover.py con una implementazione completa di LMX (LLM-based Crossover) dal pattern LLM4EC.

LEGGI PRIMA:
- src/agents/base.py (per capire BaseAgent)
- src/agents/strategy.py (per vedere pattern esistente)
- src/llm/client.py (per OllamaClient)

REQUISITI:
1. Classe CrossoverAgent che estende BaseAgent
2. Metodo `crossover(parent_a: dict, parent_b: dict, context: dict) -> dict`
3. LLM comprende SEMANTICAMENTE i due genomi
4. Genera figlio che combina tratti benefici
5. Prompt template in costante CROSSOVER_PROMPT
6. Supporta context con fitness history
7. JSON output parsato con fallback
8. DEFAULT_MODEL = "qwen2.5-coder:7b"

STRUTTURA FILE:
```python
"""LMX Crossover Agent - Semantic crossover via LLM."""
from dataclasses import dataclass
from typing import Any
from .base import BaseAgent, AgentResponse

CROSSOVER_PROMPT = \"\"\"...\"\"\"

@dataclass
class CrossoverResult:
    child: dict
    reasoning: str
    inherited_from_a: list[str]
    inherited_from_b: list[str]

class CrossoverAgent(BaseAgent):
    DEFAULT_MODEL = "qwen2.5-coder:7b"
    AGENT_NAME = "CrossoverAgent"
    
    async def crossover(self, parent_a: dict, parent_b: dict, context: dict = None) -> AgentResponse:
        ...
```

IMPORTANTE:
- Il prompt deve spiegare al LLM cosa sono i genomi
- Il prompt deve chiedere di giustificare le scelte
- Gestire errori JSON con retry
- Logging con self.logger

NON aggiungere dipendenze nuove.
Scrivi TUTTO il file, non placeholder.
```

---

## TASK 2: MutationAgent

**Modello**: `GPT-4.5` o `Claude Sonnet 4.5`
**File output**: `llm-evolution/src/agents/mutation.py` (NUOVO)
**Stima**: 20 minuti

### PROMPT (copia-incolla)

```
Sei nel repo llm-evolution. Devi CREARE il file src/agents/mutation.py con una implementazione di MutationAgent.

LEGGI PRIMA:
- src/agents/base.py (BaseAgent)
- src/agents/strategy.py (pattern)
- src/core/genome.py (GenomeSchema, mutate_gaussian)

REQUISITI:
1. Classe MutationAgent che estende BaseAgent
2. Metodo `suggest_mutations(genome: dict, fitness: float, history: list) -> MutationResult`
3. LLM analizza il genoma e suggerisce:
   - Quali geni mutare
   - In quale direzione (increase/decrease)
   - Di quanto (delta o percentage)
4. Basato su fitness history per capire trend
5. DEFAULT_MODEL = "llama3.2:3b" (veloce)

STRUTTURA FILE:
```python
"""MutationAgent - LLM-guided mutation suggestions."""
from dataclasses import dataclass, field
from typing import Literal
from .base import BaseAgent, AgentResponse

MUTATION_PROMPT = \"\"\"...\"\"\"

@dataclass
class MutationSuggestion:
    gene: str
    direction: Literal["increase", "decrease", "explore"]
    magnitude: float  # 0.0-1.0 normalized
    reason: str

@dataclass
class MutationResult:
    suggestions: list[MutationSuggestion]
    overall_strategy: str
    confidence: float

class MutationAgent(BaseAgent):
    DEFAULT_MODEL = "llama3.2:3b"
    AGENT_NAME = "MutationAgent"
    
    async def suggest_mutations(
        self, 
        genome: dict, 
        fitness: float,
        fitness_history: list[float] = None,
        generation: int = 0
    ) -> AgentResponse:
        ...
    
    def apply_suggestions(self, genome: dict, suggestions: list[MutationSuggestion]) -> dict:
        \"\"\"Applica le mutazioni suggerite al genoma.\"\"\"
        ...
```

IL PROMPT DEVE:
- Mostrare genoma corrente con bounds
- Mostrare fitness attuale e history (ultimi 5-10)
- Chiedere analisi trend (migliorando? stagnante? peggiorando?)
- Chiedere suggerimenti specifici con reasoning

OUTPUT JSON ATTESO:
```json
{
  "suggestions": [
    {"gene": "diameter_mm", "direction": "increase", "magnitude": 0.3, "reason": "..."},
    {"gene": "thickness_mm", "direction": "decrease", "magnitude": 0.1, "reason": "..."}
  ],
  "overall_strategy": "exploitation" | "exploration",
  "confidence": 0.75
}
```

NON aggiungere dipendenze nuove.
Scrivi TUTTO il file, non placeholder.
AGGIORNA anche src/agents/__init__.py per esportare MutationAgent.
```

---

## TASK 3: Surrogate Fitness Tool

**Modello**: `Claude Sonnet 4.5` (Composer)
**File output**: `llm-evolution/src/tools/evolution/surrogate.py` (NUOVO)
**Stima**: 25 minuti

### PROMPT (copia-incolla)

```
Sei nel repo llm-evolution. Devi CREARE il file src/tools/evolution/surrogate.py che implementa un Surrogate Fitness Model basato su LLM.

LEGGI PRIMA:
- src/tools/registry.py (@tool decorator)
- src/tools/evolution/pymoo_wrapper.py (pattern esistente)
- src/llm/client.py (OllamaClient)

CONCEPT:
Un surrogate model predice la fitness SENZA eseguire simulazioni costose (FEM).
L'LLM usa i sample precedenti (genome -> fitness) per predire nuovi genomi.

REQUISITI:
1. Classe SurrogateFitness con training data
2. Metodo `predict(genome: dict) -> SurrogatePrediction`
3. Metodo `add_sample(genome: dict, actual_fitness: float)`
4. Stima confidence basata su distanza dai training samples
5. Usa embeddings per similarity (opzionale, può essere semplice)
6. Decorato con @tool per registry

STRUTTURA FILE:
```python
"""Surrogate Fitness - LLM-based fitness prediction."""
from dataclasses import dataclass, field
from typing import Optional
import json
from ..registry import tool
from ...llm.client import OllamaClient

SURROGATE_PROMPT = \"\"\"...\"\"\"

@dataclass
class SurrogatePrediction:
    predicted_fitness: float
    confidence: float  # 0.0-1.0
    similar_samples: list[dict]  # Most similar training samples
    reasoning: str

@dataclass 
class TrainingSample:
    genome: dict
    fitness: float
    generation: int

class SurrogateFitness:
    def __init__(self, model: str = "qwen2.5:3b", max_samples: int = 100):
        self.client = OllamaClient()
        self.model = model
        self.samples: list[TrainingSample] = []
        self.max_samples = max_samples
    
    def add_sample(self, genome: dict, fitness: float, generation: int = 0):
        ...
    
    async def predict(self, genome: dict) -> SurrogatePrediction:
        ...
    
    def _find_similar(self, genome: dict, k: int = 5) -> list[TrainingSample]:
        \"\"\"Trova k sample più simili (distanza euclidea).\"\"\"
        ...
    
    def _genome_distance(self, g1: dict, g2: dict) -> float:
        \"\"\"Distanza euclidea normalizzata tra genomi.\"\"\"
        ...

# Tool registration
@tool(
    name="surrogate_fitness",
    description="Predict fitness without expensive simulation"
)
async def predict_fitness(genome: dict, surrogate: SurrogateFitness) -> dict:
    result = await surrogate.predict(genome)
    return {
        "predicted_fitness": result.predicted_fitness,
        "confidence": result.confidence,
        "reasoning": result.reasoning
    }
```

IL PROMPT SURROGATE DEVE:
- Mostrare training samples più simili con fitness
- Mostrare genoma da predire
- Chiedere stima fitness con confidence
- Chiedere reasoning

IMPORTANTE:
- Confidence bassa se genoma molto diverso da training
- Gestire caso con pochi samples (< 5)
- Non chiamare mai FEM reale, solo predizione

AGGIORNA anche src/tools/evolution/__init__.py per esportare.
```

---

## TASK 4: Update Coordinator per nuovi Agents

**Modello**: `Composer` (economico)
**File output**: `llm-evolution/src/core/coordinator.py` (modifica)
**Stima**: 15 minuti

### PROMPT (copia-incolla)

```
Sei nel repo llm-evolution. Devi MODIFICARE src/core/coordinator.py per integrare i nuovi agenti.

LEGGI PRIMA:
- src/core/coordinator.py (file esistente)
- src/agents/crossover.py (nuovo)
- src/agents/mutation.py (nuovo)
- src/tools/evolution/surrogate.py (nuovo)

MODIFICHE RICHIESTE:

1. Importa nuovi agenti:
```python
from ..agents import CrossoverAgent, MutationAgent
from ..tools.evolution.surrogate import SurrogateFitness
```

2. In __init__, aggiungi:
```python
self.crossover_agent: Optional[CrossoverAgent] = None
self.mutation_agent: Optional[MutationAgent] = None
self.surrogate: Optional[SurrogateFitness] = None
```

3. Aggiungi metodo `enable_llm_operators`:
```python
def enable_llm_operators(self, use_crossover: bool = True, use_mutation: bool = True, use_surrogate: bool = False):
    if use_crossover:
        self.crossover_agent = CrossoverAgent()
    if use_mutation:
        self.mutation_agent = MutationAgent()
    if use_surrogate:
        self.surrogate = SurrogateFitness()
```

4. Nel metodo `_crossover`, se self.crossover_agent esiste:
   - Usa LMX invece di crossover standard
   - Fallback a standard se LLM fallisce

5. Nel metodo `_mutate`, se self.mutation_agent esiste:
   - Chiedi suggerimenti prima di mutare
   - Applica mutazioni guidate

6. Nel metodo `_evaluate_fitness`, se self.surrogate esiste:
   - Prima prova surrogate
   - Se confidence > 0.8, usa predizione
   - Altrimenti chiama fitness_fn reale e aggiungi sample

NON riscrivere tutto il file. Mostra solo le MODIFICHE con contesto.
Usa search_replace o diff format.
```

---

## TASK 5: Integration Test

**Modello**: `Composer` (economico)
**File output**: `llm-evolution/tests/test_llm_operators.py` (NUOVO)
**Stima**: 15 minuti

### PROMPT (copia-incolla)

```
Sei nel repo llm-evolution. Crea test per i nuovi operatori LLM.

FILE: tests/test_llm_operators.py

REQUISITI:
1. Test CrossoverAgent con mock LLM response
2. Test MutationAgent con mock
3. Test SurrogateFitness con samples
4. Usa pytest e pytest-asyncio

STRUTTURA:
```python
"""Tests for LLM-guided operators."""
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.crossover import CrossoverAgent, CrossoverResult
from src.agents.mutation import MutationAgent, MutationResult
from src.tools.evolution.surrogate import SurrogateFitness

@pytest.fixture
def sample_genome():
    return {
        "diameter_mm": 200.0,
        "thickness_mm": 4.0,
        "angle_deg": 15.0
    }

@pytest.fixture
def mock_llm_response():
    ...

class TestCrossoverAgent:
    @pytest.mark.asyncio
    async def test_crossover_returns_child(self, sample_genome, mock_llm_response):
        ...
    
    @pytest.mark.asyncio
    async def test_crossover_handles_error(self):
        ...

class TestMutationAgent:
    @pytest.mark.asyncio
    async def test_mutation_suggestions(self, sample_genome):
        ...

class TestSurrogateFitness:
    def test_add_sample(self, sample_genome):
        ...
    
    @pytest.mark.asyncio
    async def test_predict_with_samples(self, sample_genome):
        ...
    
    def test_genome_distance(self):
        ...
```

IMPORTANTE:
- Mock OllamaClient per non chiamare LLM reale
- Test edge cases (empty history, no samples, etc)
- Verifica struttura output
```

---

## 📋 Checklist Esecuzione

| # | Task | Modello | File | Fatto |
|---|------|---------|------|-------|
| 1 | LMX Crossover | Sonnet 4.5 | `crossover.py` | ⬜ |
| 2 | MutationAgent | GPT-4.5 | `mutation.py` | ⬜ |
| 3 | Surrogate Fitness | Sonnet 4.5 | `surrogate.py` | ⬜ |
| 4 | Update Coordinator | Composer | `coordinator.py` | ⬜ |
| 5 | Integration Tests | Composer | `test_llm_operators.py` | ⬜ |

---

## 🔄 Workflow

1. Apri Composer/altro agent
2. Copia-incolla prompt del task
3. Verifica output creato
4. Torna qui con Opus 4.5 per review
5. Commit quando validato

---

*Prompts per task delegati - LLM-Evolution*
