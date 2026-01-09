# 📋 ATOMIC TASK PROMPTS - Per Delegazione a Modelli Economici

> **Opus 4.5**: Solo orchestrazione e review
> **GPT-4.5 / Sonnet 4.5 / Composer**: Task atomici sotto

---

## ✅ COMPLETATI (dagli altri agenti)

- [x] Struttura cartelle base
- [x] `genome.py` - Genome Protocol
- [x] `base.py` - BaseAgent
- [x] `mutation.py` - MutationAgent (parziale)

---

## 🔵 TASK 1: PyMoo Tool Wrapper

**Modello consigliato**: Sonnet 4.5 (code generation)
**Tempo stimato**: 15 min
**File output**: `llm-evolution/src/tools/evolution/pymoo_wrapper.py`

### PROMPT DA COPIARE:

```
Crea il file `llm-evolution/src/tools/evolution/pymoo_wrapper.py`.

CONTESTO:
- Questo è un framework LLM-Evolution dove PyMoo è un TOOL supervisionato, non il core
- I tool vengono registrati con il decorator @tool dal file `src/tools/registry.py`

REQUISITI:
1. Import pymoo (NSGA2, NSGA3, MOEAD)
2. Crea 3 funzioni @tool:
   - `pymoo_nsga2_tool`: NSGA-II optimization
   - `pymoo_nsga3_tool`: NSGA-III con reference directions
   - `pymoo_moead_tool`: MOEA/D decomposition

3. Ogni tool deve accettare:
   - n_var: int (numero variabili)
   - n_obj: int (numero obiettivi)
   - xl: List[float] (lower bounds)
   - xu: List[float] (upper bounds)
   - evaluate_fn: Callable (fitness function)
   - pop_size: int = 50
   - n_gen: int = 100

4. Ogni tool deve ritornare Dict con:
   - pareto_F: List[List[float]] (objective values)
   - pareto_X: List[List[float]] (decision variables)
   - n_evals: int
   - time_seconds: float

5. Usa try/except per graceful fallback se pymoo non installato

ESEMPIO STRUTTURA:
```python
from ..registry import tool

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

@tool(
    name="pymoo_nsga2",
    description="Multi-objective NSGA-II optimization",
    domain="evolution",
    cost="high",
)
def pymoo_nsga2_tool(...) -> Dict:
    ...
```

Non aggiungere commenti oltre al docstring. Codice pulito e funzionale.
```

---

## 🟢 TASK 2: Strategy Agent

**Modello consigliato**: GPT-4.5 (reasoning)
**Tempo stimato**: 20 min
**File output**: `llm-evolution/src/agents/strategy.py`

### PROMPT DA COPIARE:

```
Crea il file `llm-evolution/src/agents/strategy.py`.

CONTESTO:
- Eredita da BaseAgent in `src/agents/base.py`
- Usa il modello "strategy" (14B) per decisioni su hyperparameters
- L'agente analizza la convergence curve e suggerisce modifiche

REQUISITI:
1. Classe StrategyAgent(BaseAgent):
   - DEFAULT_MODEL = "strategy"
   - AGENT_NAME = "StrategyAgent"

2. Metodi async:
   - `analyze_convergence(fitness_history: List[float]) -> Dict`
     Ritorna: trend, velocity, stall_detected, recommendation
   
   - `suggest_hyperparameters(context: Dict) -> Dict`
     Input context: generation, stall_count, diversity, mutation_rate
     Ritorna: mutation_rate_delta, crossover_rate_delta, inject_diversity
   
   - `should_invoke_llm_operator(context: Dict) -> bool`
     Decide quando usare LLM per mutation/crossover vs operatori standard

3. Il metodo `process(input_data, context)` deve gestire:
   - {"type": "analyze"} → analyze_convergence
   - {"type": "suggest"} → suggest_hyperparameters
   - {"type": "should_llm"} → should_invoke_llm_operator

4. System prompt deve spiegare che l'agente è "area motoria" del sistema nervoso

Usa lo stile del file base.py esistente. Niente commenti inline, solo docstring.
```

---

## 🟡 TASK 3: Analysis Agent

**Modello consigliato**: Sonnet 4.5 (fast)
**Tempo stimato**: 15 min
**File output**: `llm-evolution/src/agents/analysis.py`

### PROMPT DA COPIARE:

```
Crea il file `llm-evolution/src/agents/analysis.py`.

CONTESTO:
- Eredita da BaseAgent
- Usa modello "analysis" (3B, veloce) per interpretazione fitness
- Ruolo: "area sensoriale" - interpreta dati, detecta anomalie

REQUISITI:
1. Classe AnalysisAgent(BaseAgent):
   - DEFAULT_MODEL = "analysis"
   - AGENT_NAME = "AnalysisAgent"

2. Metodi async:
   - `interpret_fitness(genome: Dict, fitness: Dict) -> str`
     Spiega in linguaggio naturale cosa significano i valori
   
   - `detect_anomaly(genome: Dict, fitness: Dict, history: List) -> Optional[Dict]`
     Ritorna None se ok, altrimenti {type, severity, description}
   
   - `summarize_generation(stats: Dict) -> str`
     Genera summary per human-in-the-loop

3. Il metodo process gestisce i 3 tipi di richiesta

4. Per detect_anomaly, check:
   - fitness troppo basso/alto rispetto alla media
   - genome fuori dai bounds
   - sudden fitness drop

Codice conciso, niente commenti inline.
```

---

## 🟣 TASK 4: RAG Agent

**Modello consigliato**: Composer (no LLM, solo code)
**Tempo stimato**: 20 min
**File output**: `llm-evolution/src/agents/rag.py`

### PROMPT DA COPIARE:

```
Crea il file `llm-evolution/src/agents/rag.py`.

CONTESTO:
- NON eredita da BaseAgent (non usa LLM direttamente)
- Fa semantic search su SurrealDB
- Inietta papers nel context degli altri agenti

REQUISITI:
1. Classe RAGAgent:
   - __init__(db_client: SurrealDBClient = None)
   - Usa get_db_client() da src/knowledge/surrealdb.py

2. Metodi async:
   - `get_relevant_papers(query: str, limit: int = 5) -> List[Paper]`
     Semantic search (se embeddings) o text search
   
   - `inject_context(agent: BaseAgent, topic: str) -> None`
     Cerca papers su topic, chiama agent.inject_context(formatted_text)
   
   - `get_domain_knowledge(domain: str) -> str`
     Ritorna testo formattato per prompt LLM

3. Helper method:
   - `_format_papers_for_prompt(papers: List[Paper]) -> str`
     Formatta: "## Relevant Research\n- Paper1 (2023): abstract..."

4. Import da:
   - from ..knowledge.surrealdb import SurrealDBClient, Paper, get_db_client

Niente LLM calls, solo database queries e string formatting.
```

---

## 🔴 TASK 5: Evolution Coordinator

**Modello consigliato**: GPT-4.5 (complex logic)
**Tempo stimato**: 30 min
**File output**: `llm-evolution/src/core/coordinator.py`

### PROMPT DA COPIARE:

```
Crea il file `llm-evolution/src/core/coordinator.py`.

CONTESTO:
- "Midollo spinale" del sistema: traduce decisioni LLM → azioni GA
- Coordina tutti gli agenti
- Loop principale dell'evoluzione

REQUISITI:
1. Dataclass EvolutionState:
   - generation: int
   - population: List[DictGenome]
   - fitness_history: List[float]
   - best_genome: Optional[DictGenome]
   - best_fitness: float
   - stall_count: int

2. Dataclass EvolutionResult:
   - best_genome: DictGenome
   - best_fitness: float
   - generations_run: int
   - total_evaluations: int
   - agent_interventions: List[Dict]

3. Classe EvolutionCoordinator:
   - __init__(orchestrator, strategy, analysis, rag, tools)
   - Tutti gli agent sono opzionali (può funzionare senza LLM)

4. Metodo async principale:
   ```python
   async def run(
       self,
       schema: GenomeSchema,
       fitness_fn: Callable,
       max_generations: int = 100,
       population_size: int = 50,
       llm_intervention_frequency: int = 10,  # Ogni N gen chiedi all'LLM
   ) -> EvolutionResult:
   ```

5. Loop interno:
   - Genera popolazione iniziale (random_genome)
   - Per ogni generazione:
     a. Valuta fitness
     b. Se gen % llm_intervention_frequency == 0:
        - Chiedi a StrategyAgent suggerimenti
        - Applica modifiche
     c. Selezione (tournament)
     d. Crossover (uniform o LLM se attivato)
     e. Mutation (gaussian o LLM se attivato)
     f. Update state, check stall
   - Return EvolutionResult

6. Import da:
   - from .genome import GenomeSchema, DictGenome, random_genome, crossover_uniform, mutate_gaussian

Metodi helper: _tournament_select, _check_stall, _update_state
```

---

## ⚪ TASK 6: SurrealDB Setup Script

**Modello consigliato**: Composer (bash)
**Tempo stimato**: 10 min
**File output**: `llm-evolution/scripts/setup_surrealdb.sh`

### PROMPT DA COPIARE:

```
Crea il file `llm-evolution/scripts/setup_surrealdb.sh`.

REQUISITI:
1. Script bash che:
   - Verifica se surreal CLI è installato
   - Crea directory ~/.config/surrealdb/ se non esiste
   - Avvia SurrealDB con:
     - File: ~/.config/surrealdb/evolution.db
     - User: root, Pass: root
     - Log level: warn
   - Crea namespace "evolution", database "knowledge"

2. Output user-friendly con emoji

ESEMPIO:
```bash
#!/bin/bash
echo "🚀 Setting up SurrealDB for LLM-Evolution..."

# Check surreal installed
if ! command -v surreal &> /dev/null; then
    echo "❌ surreal CLI not found. Install: curl -sSf https://install.surrealdb.com | sh"
    exit 1
fi

# Create dir
mkdir -p ~/.config/surrealdb

# Start (background)
surreal start --log warn --user root --pass root \
    file:~/.config/surrealdb/evolution.db &

sleep 2
echo "✅ SurrealDB started on localhost:8000"
echo "   Namespace: evolution"
echo "   Database: knowledge"
```

Aggiungi anche creazione schema iniziale via curl.
```

---

## 📊 ORDINE ESECUZIONE

```
Parallelo (chiunque):
├── TASK 1: PyMoo Wrapper      → Sonnet 4.5
├── TASK 2: Strategy Agent     → GPT-4.5
├── TASK 3: Analysis Agent     → Sonnet 4.5
└── TASK 6: SurrealDB Script   → Composer

Sequenziale (dopo Task 2,3):
├── TASK 4: RAG Agent          → Composer
└── TASK 5: Coordinator        → GPT-4.5
```

---

## ✅ CHECKLIST REVIEW (Per Opus)

Dopo ogni task completato, Opus verifica:
- [ ] File creato nel path corretto
- [ ] Import corretti
- [ ] Eredita da classi giuste
- [ ] Metodi async dove richiesto
- [ ] Niente codice duplicato
- [ ] Test basico funziona

---

*Creato: 2025-01-09*
*Per delegazione a modelli economici*
