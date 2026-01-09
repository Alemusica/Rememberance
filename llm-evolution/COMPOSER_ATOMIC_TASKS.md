# 🎯 COMPOSER ATOMIC TASKS - LLM-Evolution

## Stato Attuale

### ✅ GIÀ FATTO (Branch `llm-evolution`)
- [x] Genome Protocol agnostico (`genome.py`)
- [x] OllamaClient con cache (`client.py`)
- [x] ToolRegistry con @tool decorator (`registry.py`)
- [x] BaseAgent + OrchestratorAgent (`agents/`)
- [x] SurrealDB client async (`surrealdb.py`)
- [x] Script ingestion ArXiv + LLM4EC (`ingestion/`)

### 🔴 DA FARE
- [ ] PyMoo come Tool supervisionato
- [ ] Quality-Diversity / MAP-Elites integration
- [ ] Strategy/Analysis/RAG Agents
- [ ] Evolution Coordinator
- [ ] SurrealDB dedicato per questo framework

---

## 📊 RICERCHE MODERNE SULLA PIPELINE EVOLUTIVA

### Pattern Chiave da LLM4EC Survey

| Pattern | Paper | Cosa Fa | Applicazione |
|---------|-------|---------|--------------|
| **OPRO** | DeepMind 2023 | LLM genera soluzioni dal prompt con history | Orchestrator suggerisce nuovi genomi |
| **LMX** | Meyerson 2023 | LLM fa crossover semantico | CrossoverAgent |
| **LEO** | Liu 2024 | Chain-of-thought per selezione multi-obiettivo | SelectionAgent con reasoning |
| **FunSearch** | DeepMind 2023 | Evolve programmi, LLM genera codice | Auto-genera fitness functions |
| **EvoAgent** | Yuan 2024 | Multi-agent che evolvono per specializzarsi | Il nostro sistema di agenti |

### Quality-Diversity (MAP-Elites) - NON ANCORA INTEGRATO

**Problema**: NSGA-II trova Pareto front ma non esplora tutto lo spazio.

**Soluzione**: MAP-Elites crea archivio di soluzioni diverse + performanti.

```python
# Da integrare: pymoo ha CVT-MAP-Elites
from pymoo.algorithms.moo.cvtmapelites import CVTMAPELITES

# Behavior descriptors per bowl:
# - dim1: frequenza fondamentale
# - dim2: rapporto quinta
# - dim3: sustain
```

### Surrogate-Assisted Evolution - NON ANCORA INTEGRATO

**Problema**: FEM è lento (~1-5 sec per genome).

**Soluzione**: Neural surrogate predice fitness, FEM solo per validazione.

---

## 🔧 PYMOO COME TOOL SUPERVISIONATO

PyMoo **NON** deve essere il cuore dell'evoluzione.
PyMoo è un **TOOL** come CNC Virtual - l'LLM lo orchestra.

### Architettura Target

```
OrchestratorAgent (LLM 32B)
    │
    ├─→ "Usa NSGA-II per 20 generazioni"
    │       │
    │       └─→ PymooTool.run_nsga2(pop=50, gen=20)
    │               │
    │               └─→ Risultato: Pareto front
    │
    ├─→ "Analizza Pareto, suggerisci direzione"
    │       │
    │       └─→ AnalysisAgent
    │
    └─→ "Fai LMX crossover tra top 2 soluzioni"
            │
            └─→ CrossoverAgent (LLM-guided)
```

### Tool Registration per PyMoo

```python
@tool(
    name="pymoo_nsga2",
    description="Run NSGA-II multi-objective optimization",
    domain="evolution",
    cost="high",
)
async def pymoo_nsga2_tool(
    problem: Dict,           # {objectives, n_var, bounds}
    population_size: int,
    n_generations: int,
    seed: int = None,
) -> Dict:
    """
    Execute NSGA-II and return Pareto front.
    
    Returns:
        {
            "pareto_front": [[obj1, obj2, ...], ...],
            "pareto_solutions": [[x1, x2, ...], ...],
            "n_evaluations": int,
            "hypervolume": float,
        }
    """
    ...
```

---

## 🗄️ SURREALDB: SÌ, DEDICATO

**Decisione**: Creare database separato per LLM-Evolution.

```bash
# Database esistente (Golden Studio)
surreal start file:~/.config/surrealdb/research.db
# NS: research, DB: knowledge

# Nuovo database (LLM-Evolution) - SEPARATO
surreal start file:~/.config/surrealdb/evolution.db
# NS: evolution, DB: knowledge
```

**Perché separato?**
1. Paper diversi (LLM4EC vs vibroacustica)
2. Schema diverso (genome cache, LTM distillation)
3. Evita conflitti
4. Più facile backup/deploy

---

## 📋 TASK ATOMICI PER COMPOSER AGENTS

### 🔵 COMPOSER 1: PyMoo Tool Wrapper

**File**: `llm-evolution/src/tools/evolution/pymoo_wrapper.py`

**Task Atomico**:
```
1. Crea src/tools/evolution/pymoo_wrapper.py
2. Wrap NSGA-II, NSGA-III, MOEAD come @tool
3. Input: problem dict (n_var, n_obj, bounds, constraints)
4. Output: pareto_front, solutions, metrics
5. Test con Rastrigin multi-objective
```

**Codice Skeleton**:
```python
from ...tools.registry import tool
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

@tool(
    name="pymoo_nsga2",
    description="Multi-objective NSGA-II optimization",
    domain="evolution",
    cost="high",
)
def pymoo_nsga2_tool(
    n_var: int,
    n_obj: int,
    xl: list,  # lower bounds
    xu: list,  # upper bounds
    fitness_fn_name: str,  # registered fitness function
    pop_size: int = 50,
    n_gen: int = 100,
) -> dict:
    """Run NSGA-II, return Pareto front."""
    ...
```

---

### 🟢 COMPOSER 2: Strategy Agent

**File**: `llm-evolution/src/agents/strategy.py`

**Task Atomico**:
```
1. Crea src/agents/strategy.py
2. Eredita da BaseAgent
3. Analizza convergence curve
4. Suggerisce: mutation_rate, inject_diversity, when_to_use_llm_crossover
5. Input: fitness_history, generation, stall_count
6. Output: StrategyDecision dataclass
```

**Metodi Richiesti**:
```python
class StrategyAgent(BaseAgent):
    async def analyze_convergence(self, history: List[float]) -> Dict
    async def suggest_hyperparameters(self, context: Dict) -> Dict
    async def should_use_llm_operator(self, context: Dict) -> bool
```

---

### 🟡 COMPOSER 3: Analysis Agent

**File**: `llm-evolution/src/agents/analysis.py`

**Task Atomico**:
```
1. Crea src/agents/analysis.py
2. Eredita da BaseAgent (model: llama3.2:3b - veloce)
3. Interpreta fitness values
4. Detecta anomalie (Pokayoke style)
5. Genera summary per human-in-the-loop
```

**Metodi Richiesti**:
```python
class AnalysisAgent(BaseAgent):
    async def interpret_fitness(self, genome: Dict, fitness: Dict) -> str
    async def detect_anomaly(self, genome: Dict, context: Dict) -> Optional[str]
    async def summarize_generation(self, pop_stats: Dict) -> str
```

---

### 🟣 COMPOSER 4: RAG Agent

**File**: `llm-evolution/src/agents/rag.py`

**Task Atomico**:
```
1. Crea src/agents/rag.py
2. NON eredita da BaseAgent (usa solo embeddings)
3. Semantic search su SurrealDB
4. Inject papers rilevanti nel context degli altri agenti
5. Cache embeddings localmente
```

**Metodi Richiesti**:
```python
class RAGAgent:
    async def get_relevant_papers(self, query: str, limit: int = 5) -> List[Paper]
    async def inject_context(self, agent: BaseAgent, topic: str)
    async def get_domain_knowledge(self, domain: str) -> str
```

---

### 🔴 COMPOSER 5: Evolution Coordinator

**File**: `llm-evolution/src/core/coordinator.py`

**Task Atomico**:
```
1. Crea src/core/coordinator.py
2. Midollo spinale: traduce decisioni LLM → azioni GA
3. Loop principale: generation → fitness → LLM analysis → next gen
4. Gestisce budget token
5. Logging strutturato
```

**Classe Principale**:
```python
class EvolutionCoordinator:
    def __init__(self, orchestrator, strategy, analysis, rag, tools):
        ...
    
    async def run(
        self,
        domain_spec: DomainSpec,
        max_generations: int,
        token_budget: int,
    ) -> EvolutionResult:
        ...
```

---

### ⚪ COMPOSER 6: SurrealDB Setup + MCP Server

**Files**: 
- `llm-evolution/scripts/setup_surrealdb.sh`
- `llm-evolution/src/knowledge/mcp_server.py`

**Task Atomico**:
```
1. Script bash per inizializzare database evolution
2. MCP server dedicato per questo framework
3. Tools: search_papers, get_llm4ec_context, store_ltm
4. Testare con Cursor
```

---

## 📅 ORDINE ESECUZIONE

```
Settimana 1 (Parallelo):
├── COMPOSER 1: PyMoo Tool Wrapper
├── COMPOSER 2: Strategy Agent
├── COMPOSER 3: Analysis Agent
└── COMPOSER 6: SurrealDB Setup

Settimana 2 (Parallelo):
├── COMPOSER 4: RAG Agent
└── COMPOSER 5: Evolution Coordinator

Settimana 3:
└── Integration Test + Bowl Example
```

---

## ✅ CHECKLIST PRE-TASK

Prima di assegnare ai Composer, verificare:

- [ ] Branch `llm-evolution` è aggiornato
- [ ] Ollama running con modelli
- [ ] SurrealDB nuovo creato (`evolution.db`)
- [ ] Script ingestion eseguiti
- [ ] Test import: `from llm_evolution.src.core import *`

---

*Created: 2025-01-09*
*Per delegazione a Cursor Composer Agents*
