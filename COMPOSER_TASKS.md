# Cursor Composer - Task Assignment

## 🎯 Best Practices per Composer Agents Paralleli

### Regole Chiave (da [egghead.io](https://egghead.io/lessons/launch-multiple-cursor-composer-ai-agents-to-work-in-parallel))

1. **Git Worktree per Isolamento**
   ```bash
   # Crea worktree separati per ogni agent
   git worktree add ../llm-evo-agent1 -b agent1-infra
   git worktree add ../llm-evo-agent2 -b agent2-agents
   git worktree add ../llm-evo-agent3 -b agent3-ingestion
   ```

2. **Task Atomici**: Ogni task deve essere completabile in 1 sessione Composer
3. **No Dipendenze Circolari**: Tasks paralleli non devono toccare gli stessi file
4. **Merge Strategy**: Ogni agent lavora su branch, poi PR review

---

## 📋 TASK ASSIGNMENT

### 🔵 COMPOSER AGENT 1 - Infrastructure Lead

**Branch**: `agent1-core-infrastructure`

#### Task 1.1: Create Branch and Structure
```
Prompt per Composer:

"Crea un nuovo branch 'llm-evolution' e la struttura cartelle:

llm-evolution/
├── src/
│   ├── core/
│   ├── agents/
│   ├── llm/
│   └── ingestion/
├── examples/
├── tests/
└── requirements.txt

Copia da binaural_golden/src/core/ questi file:
- agnostic_evolution.py → src/core/genome.py (rinomina e semplifica)
- evolutionary_optimizer.py → src/core/evolution.py (rimuovi plate-specific)
- evolution_memory.py → src/core/memory.py
- evolution_pipeline.py → src/core/pipeline.py

Mantieni solo le interfacce generiche, rimuovi riferimenti a PlateGenome."
```

#### Task 2.1: BaseAgent Implementation
```
Prompt per Composer:

"Crea src/agents/base.py con:

1. BaseAgent class astratta:
   - __init__(llm_client, config)
   - abstract method: generate_prompt(context) -> str
   - abstract method: parse_response(response) -> Any
   - method: invoke(context) -> Any (calls LLM and parses)

2. AgentConfig dataclass:
   - model_name: str
   - temperature: float
   - max_tokens: int
   - retry_count: int

3. Logging e error handling

Usa typing hints e docstrings."
```

#### Task 2.4: MutationAgent
```
Prompt per Composer:

"Crea src/agents/mutation.py:

MutationAgent(BaseAgent):
- Analizza genome corrente e fitness breakdown
- Genera prompt che chiede 3 mutazioni specifiche
- Parsa risposta JSON con: {mutations: [{field, old_value, new_value, rationale}]}
- Applica mutazioni al genome

Include prompt template con few-shot examples."
```

---

### 🟢 COMPOSER AGENT 2 - LLM & Agents

**Branch**: `agent2-llm-agents`

#### Task 1.2: LLM Client
```
Prompt per Composer:

"Crea src/llm/client.py:

1. LLMClient class con supporto per:
   - Ollama (local): endpoint http://localhost:11434
   - OpenAI (cloud): con API key
   
2. Interface:
   - async def complete(prompt, model, temperature, max_tokens) -> str
   - async def complete_json(prompt, model, schema) -> dict
   
3. Features:
   - Retry con exponential backoff
   - Token counting (tiktoken per OpenAI, stima per Ollama)
   - Response caching (lru_cache)
   - Timeout handling

Usa httpx per async requests."
```

#### Task 2.2: CrossoverAgent (LMX)
```
Prompt per Composer:

"Crea src/agents/crossover.py implementando Language Model Crossover:

CrossoverAgent(BaseAgent):
- Input: parent_a, parent_b con loro fitness breakdown
- Prompt template che spiega:
  'Parent A è buono per X, Parent B è buono per Y.
   Combina creando offspring che eredita il meglio.'
- Output: nuovo genome valido

Reference paper: 'Language Model Crossover' (Meyerson 2023)
Include validation che offspring sia nel dominio valido."
```

#### Task 2.5: ExplainerAgent
```
Prompt per Composer:

"Crea src/agents/explainer.py:

ExplainerAgent(BaseAgent):
- Input: anomaly_type, generation, fitness_history, genome
- Genera spiegazione human-readable dell'anomalia
- Suggerisce azione: continue/pause/rollback

Integra con PokayokeObserver esistente.
Output: {explanation: str, severity: 1-10, action: str}"
```

---

### 🟡 COMPOSER AGENT 3 - Ingestion & Selection

**Branch**: `agent3-ingestion-selection`

#### Task 1.3: Prompt Templates
```
Prompt per Composer:

"Crea src/llm/prompts.py:

1. PromptTemplate class con:
   - template: str (Jinja2 format)
   - render(**kwargs) -> str
   
2. Templates predefiniti:
   - MUTATION_PROMPT: analisi genome + richiesta mutazioni
   - CROSSOVER_PROMPT: combinazione semantic parents
   - SELECTION_PROMPT: Pareto ranking con reasoning
   - STRATEGY_PROMPT: hyperparameter tuning
   - EXPLAIN_PROMPT: anomaly explanation

3. Ogni template include:
   - System message
   - Few-shot examples
   - Output schema JSON"
```

#### Task 2.3: SelectionAgent
```
Prompt per Composer:

"Crea src/agents/selection.py:

SelectionAgent(BaseAgent):
- Input: lista di (genome, objectives_dict)
- Chain-of-thought prompt per:
  1. Identificare Pareto front
  2. Rankare per weighted objectives
  3. Spiegare trade-offs
- Output: lista ordinata con rationale

Reference: LEO paper (Reasoning with Elitism)"
```

#### Task 3.1: arXiv Ingestion
```
Prompt per Composer:

"Crea src/ingestion/arxiv.py:

1. ArxivFetcher class:
   - search(query, max_results) -> List[Paper]
   - fetch_pdf(arxiv_id) -> bytes
   - extract_abstract(arxiv_id) -> str

2. Paper dataclass:
   - arxiv_id, title, authors, abstract, categories, published

3. SurrealDB integration:
   - save_to_surrealdb(paper) usando pattern da import_bibliography_to_surrealdb.py

Queries predefinite:
- 'evolutionary computation LLM'
- 'genetic algorithm neural network'
- 'multi-objective optimization'"
```

---

## 🔄 Merge Strategy

```bash
# Dopo che tutti gli agent completano:
git checkout llm-evolution
git merge agent1-core-infrastructure --no-ff
git merge agent2-llm-agents --no-ff
git merge agent3-ingestion-selection --no-ff

# Risolvi conflitti se necessario
# Test integration
pytest tests/
```

---

## ⚡ Quick Start Commands

```bash
# Terminal 1 - Agent 1
cd /Users/alessioivoycazzaniga/Rememberance
git worktree add ../llm-evo-1 -b agent1-core
cd ../llm-evo-1
# Apri Cursor qui

# Terminal 2 - Agent 2  
git worktree add ../llm-evo-2 -b agent2-llm
cd ../llm-evo-2
# Apri Cursor qui

# Terminal 3 - Agent 3
git worktree add ../llm-evo-3 -b agent3-ingest
cd ../llm-evo-3
# Apri Cursor qui
```

---

## 📊 Progress Tracking

| Task | Agent | Status | Branch |
|------|-------|--------|--------|
| 1.1 Structure | 1 | ⬜ TODO | agent1-core |
| 1.2 LLM Client | 2 | ⬜ TODO | agent2-llm |
| 1.3 Prompts | 3 | ⬜ TODO | agent3-ingest |
| 2.1 BaseAgent | 1 | ⬜ TODO | agent1-core |
| 2.2 CrossoverAgent | 2 | ⬜ TODO | agent2-llm |
| 2.3 SelectionAgent | 3 | ⬜ TODO | agent3-ingest |
| 2.4 MutationAgent | 1 | ⬜ TODO | agent1-core |
| 2.5 ExplainerAgent | 2 | ⬜ TODO | agent2-llm |
| 3.1 arXiv | 3 | ⬜ TODO | agent3-ingest |

---

*Last Updated: 2025-01-09*
