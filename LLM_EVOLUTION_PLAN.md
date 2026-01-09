# LLM-Enhanced Evolution - Master Plan

## 🎯 Objective
Create a standalone branch with LLM agents supervising evolutionary optimization, following [LLM4EC](https://github.com/wuxingyu-ai/LLM4EC) patterns.

---

## 📊 Current Status

### ✅ Already Implemented
- [x] Genetic Algorithm (tournament, crossover, mutation, elitism)
- [x] Evolution Pipeline (Pokayoke → ExciterGene → Physics → RDNN → LTM → Scoring)
- [x] RDNN Memory (LSTM/GRU trajectory prediction)
- [x] LTM Distillation (cross-run knowledge)
- [x] SurrealDB ingestion script (`import_bibliography_to_surrealdb.py`)
- [x] 70+ papers in knowledge base

### 🚧 To Implement
- [ ] LLM Agent infrastructure
- [ ] 5 specialized agents
- [ ] arXiv/LLM4EC paper ingestion
- [ ] Standalone branch

---

## 🔧 Recommended Local LLMs (January 2025)

| Model | Size | Use Case | Ollama Command |
|-------|------|----------|----------------|
| **DeepSeek R1 Distill** | 14B/32B | Reasoning, Strategy | `ollama pull deepseek-r1:14b` |
| **Qwen 2.5 Coder** | 32B | Code generation | `ollama pull qwen2.5-coder:32b` |
| **Llama 3.3** | 70B | General | `ollama pull llama3.3:70b` |

---

## 📁 Target Branch Structure

```
llm-evolution/
├── src/
│   ├── core/                    # Stripped from binaural_golden
│   │   ├── genome.py            # Generic genome protocol
│   │   ├── evolution.py         # GA core
│   │   ├── fitness.py           # Abstract evaluator
│   │   ├── memory.py            # RDNN + LTM
│   │   └── pipeline.py          # Evolution pipeline
│   │
│   ├── agents/                  # NEW: LLM Agents
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent with LLM client
│   │   ├── mutation.py          # Physics-aware mutations
│   │   ├── crossover.py         # LMX-style semantic crossover
│   │   ├── selection.py         # Pareto reasoning
│   │   ├── strategy.py          # Hyperparameter tuning
│   │   └── explainer.py         # Anomaly explanation
│   │
│   ├── llm/                     # LLM Infrastructure
│   │   ├── __init__.py
│   │   ├── client.py            # Ollama/OpenAI client
│   │   ├── prompts.py           # Prompt templates
│   │   └── parser.py            # JSON output parsing
│   │
│   └── ingestion/               # Knowledge ingestion
│       ├── arxiv.py             # arXiv paper fetcher
│       ├── surrealdb.py         # SurrealDB interface
│       └── embeddings.py        # Vector embeddings
│
├── examples/
│   ├── benchmark_functions.py   # Rastrigin, Rosenbrock
│   ├── tsp.py                   # Traveling Salesman
│   └── vibroacoustic.py         # Plate optimization
│
├── tests/
├── requirements.txt
└── README.md
```

---

## 🚀 ATOMIC TASKS FOR COMPOSER AGENTS

### Phase 1: Infrastructure Setup

#### TASK-1.1: Create Branch and Base Structure
**Agent**: Composer 1
**Atomic**: YES
**Dependencies**: None
```
- Create branch `llm-evolution` from main
- Create folder structure as above
- Copy core modules from binaural_golden/src/core/
- Strip plate-specific code, keep generic interfaces
```

#### TASK-1.2: LLM Client Infrastructure
**Agent**: Composer 2
**Atomic**: YES
**Dependencies**: None (parallel with 1.1)
```
- Create src/llm/client.py
- Support Ollama (local) and OpenAI (cloud)
- Async interface with retry logic
- Token counting and budgeting
```

#### TASK-1.3: Prompt Template System
**Agent**: Composer 3
**Atomic**: YES
**Dependencies**: None (parallel)
```
- Create src/llm/prompts.py
- Jinja2 or f-string templates
- Templates for: mutation, crossover, selection, strategy, explain
- Few-shot examples for each
```

---

### Phase 2: Agent Implementation

#### TASK-2.1: BaseAgent and MutationAgent
**Agent**: Composer 1
**Atomic**: YES
**Dependencies**: TASK-1.2, TASK-1.3
```
- Create src/agents/base.py with BaseAgent class
- Create src/agents/mutation.py
- Implement physics-aware mutation suggestions
- Test with mock LLM responses
```

#### TASK-2.2: CrossoverAgent (LMX)
**Agent**: Composer 2
**Atomic**: YES
**Dependencies**: TASK-1.2, TASK-1.3
```
- Create src/agents/crossover.py
- Implement Language Model Crossover (Meyerson 2023)
- Semantic understanding of parent strengths
- Output valid offspring genome
```

#### TASK-2.3: SelectionAgent
**Agent**: Composer 3
**Atomic**: YES
**Dependencies**: TASK-1.2, TASK-1.3
```
- Create src/agents/selection.py
- Chain-of-thought Pareto reasoning
- Multi-objective trade-off analysis
- Return ranked selection with rationale
```

#### TASK-2.4: StrategyAgent
**Agent**: Composer 1
**Atomic**: YES
**Dependencies**: TASK-2.1
```
- Create src/agents/strategy.py
- Analyze convergence curve
- Suggest hyperparameter changes
- Integrate with existing RDNN trajectory
```

#### TASK-2.5: ExplainerAgent
**Agent**: Composer 2
**Atomic**: YES
**Dependencies**: TASK-2.1
```
- Create src/agents/explainer.py
- Integrate with PokayokeObserver
- Natural language anomaly explanation
- Recommended actions
```

---

### Phase 3: Knowledge Ingestion

#### TASK-3.1: arXiv Ingestion Script
**Agent**: Composer 3
**Atomic**: YES
**Dependencies**: None (parallel)
```
- Create src/ingestion/arxiv.py
- Fetch papers from arXiv API
- Categories: cs.NE (neuroevolution), cs.AI, cs.LG
- Keywords: evolutionary, LLM, optimization
- Store in SurrealDB with embeddings
```

#### TASK-3.2: LLM4EC Papers Import
**Agent**: Composer 1
**Atomic**: YES
**Dependencies**: TASK-3.1
```
- Parse LLM4EC README table
- Fetch all referenced papers
- Create domain mappings for evolutionary LLM
- Add to knowledge base
```

#### TASK-3.3: Embedding Generation
**Agent**: Composer 2
**Atomic**: YES
**Dependencies**: TASK-3.1
```
- Create src/ingestion/embeddings.py
- Use local embedding model (e.g., nomic-embed-text)
- Generate embeddings for all papers
- Enable semantic search in agents
```

---

### Phase 4: Integration

#### TASK-4.1: Agent Coordinator
**Agent**: Composer 1
**Atomic**: YES
**Dependencies**: All Phase 2 tasks
```
- Create src/agents/coordinator.py
- Orchestrate all agents
- Budget token allocation
- Parallel vs sequential execution
```

#### TASK-4.2: Evolution Pipeline Integration
**Agent**: Composer 2
**Atomic**: YES
**Dependencies**: TASK-4.1
```
- Modify evolution pipeline to use agents
- Hook points: pre-mutation, post-crossover, selection, stall
- Configurable agent activation
```

#### TASK-4.3: Benchmarks and Tests
**Agent**: Composer 3
**Atomic**: YES
**Dependencies**: TASK-4.2
```
- Create benchmark suite
- Compare: vanilla GA vs LLM-enhanced
- Metrics: convergence speed, final fitness, diversity
- Ablation: which agent helps most
```

---

## 📅 Execution Order (Parallel Where Possible)

```
Week 1:
├── [Parallel] TASK-1.1, TASK-1.2, TASK-1.3
└── [Parallel] TASK-3.1

Week 2:
├── [Parallel] TASK-2.1, TASK-2.2, TASK-2.3
└── [Parallel] TASK-3.2, TASK-3.3

Week 3:
├── [Sequential] TASK-2.4, TASK-2.5 (depend on 2.1)
└── [Sequential] TASK-4.1, TASK-4.2, TASK-4.3
```

---

## 🔗 Key References

- [LLM4EC Survey](https://github.com/wuxingyu-ai/LLM4EC) - IEEE TEVC accepted
- [OPRO](https://arxiv.org/pdf/2309.03409) - LLM as optimizer (DeepMind)
- [LMX](https://arxiv.org/pdf/2302.12170) - Language Model Crossover
- [LEO](https://arxiv.org/pdf/2403.02054) - Reasoning with Elitism
- [EvoAgent](https://arxiv.org/pdf/2406.14228) - Multi-agent evolution

---

## 💡 Notes

1. **Token Budget**: ~50k tokens/run with caching
2. **Fallback**: If LLM fails, use traditional operator
3. **Caching**: Cache LLM responses for similar genomes
4. **Metrics**: Log agent contribution to fitness improvement

---

*Created: 2025-01-09*
*Author: Alessio Ivoy Cazzaniga*
