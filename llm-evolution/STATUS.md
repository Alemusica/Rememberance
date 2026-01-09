# 📊 LLM-Evolution - STATUS & CHECKLIST

> Ultimo aggiornamento: 2025-01-09

---

## 🔴 MANCANTE - DA FARE

### 1. Modelli LLM (Ollama)

```bash
# Ollama deve essere attivo
ollama serve

# Modelli da scaricare
ollama pull deepseek-r1:14b      # Orchestrator (reasoning)
ollama pull qwen2.5-coder:14b    # Strategy/Code
ollama pull llama3.2:3b          # Analysis (fast)
ollama pull qwen2.5:3b           # Fast decisions
ollama pull nomic-embed-text     # Embeddings per RAG
```

### 2. SurrealDB

```bash
# Installare se mancante
curl -sSf https://install.surrealdb.com | sh

# Eseguire setup
chmod +x llm-evolution/scripts/setup_surrealdb.sh
./llm-evolution/scripts/setup_surrealdb.sh
```

### 3. Ingestion Papers

```bash
# Dopo SurrealDB attivo
cd llm-evolution
python -m src.knowledge.ingestion.llm4ec
python -m src.knowledge.ingestion.arxiv
```

### 4. Dependencies Python

```bash
pip install -r llm-evolution/requirements.txt
```

---

## ✅ COMPLETATO

| Componente | File | Linee | Status |
|------------|------|-------|--------|
| Genome Protocol | `src/core/genome.py` | ~400 | ✅ |
| Ollama Client | `src/llm/client.py` | ~300 | ✅ |
| Tool Registry | `src/tools/registry.py` | ~250 | ✅ |
| BaseAgent | `src/agents/base.py` | ~200 | ✅ |
| OrchestratorAgent | `src/agents/orchestrator.py` | ~430 | ✅ |
| StrategyAgent | `src/agents/strategy.py` | ~360 | ✅ |
| AnalysisAgent | `src/agents/analysis.py` | ~200 | ✅ |
| RAGAgent | `src/agents/rag.py` | ~160 | ✅ |
| Coordinator | `src/core/coordinator.py` | ~620 | ✅ |
| PyMoo Wrapper | `src/tools/evolution/pymoo_wrapper.py` | ~330 | ✅ |
| SurrealDB Client | `src/knowledge/surrealdb.py` | ~350 | ✅ |
| ArXiv Ingestion | `src/knowledge/ingestion/arxiv.py` | ~220 | ✅ |
| LLM4EC Papers | `src/knowledge/ingestion/llm4ec.py` | ~240 | ✅ |
| Setup Script | `scripts/setup_surrealdb.sh` | ~150 | ✅ |

**Totale**: ~4000 linee di codice

---

## 📚 CONFRONTO CON STATO DELL'ARTE

### Repo Analizzati

| Repo | URL | Cosa Fa |
|------|-----|---------|
| **LLM4EC** | github.com/wuxingyu-ai/LLM4EC | Survey IEEE TEVC |
| **LLaMEA** | github.com/XAI-liacs/LLaMEA | LLM genera metaeuristiche |
| **EvoAgentX** | github.com/EvoAgentX/EvoAgentX | Multi-agent auto-evolutivi |
| **OpenEvolve** | (Google) | FunSearch per algorithm discovery |

### Feature Comparison

| Feature | LLM4EC | LLaMEA | EvoAgentX | **Nostro** |
|---------|--------|--------|-----------|------------|
| LLM as Optimizer | ✅ OPRO | ✅ | - | ✅ Orchestrator |
| LLM Crossover | ✅ LMX | ✅ | - | ⚠️ crossover.py (basic) |
| LLM Mutation | ✅ | ✅ | - | ⚠️ mutation non impl. |
| Multi-Objective | ✅ LEO | - | - | ✅ PyMoo tools |
| Self-Evolving Agents | - | - | ✅ | ❌ Non implementato |
| Quality-Diversity | - | - | - | ❌ MAP-Elites mancante |
| Surrogate Model | - | - | - | ❌ Non implementato |
| RAG Knowledge | - | - | - | ✅ SurrealDB |

### Cosa Manca vs Stato dell'Arte

1. **LMX Crossover Semantico** (da LLM4EC)
   - Nostro `crossover.py` è placeholder
   - Serve: prompt che capisce semantica dei genomi

2. **MutationAgent** (da LLM4EC)
   - Non implementato
   - Serve: LLM suggerisce quali geni mutare e come

3. **Self-Evolving Agents** (da EvoAgentX)
   - Agenti che migliorano i propri prompt
   - Non prioritario ora

4. **MAP-Elites / QD** (Quality-Diversity)
   - PyMoo ha `CVTMAPELITES`
   - Da aggiungere a pymoo_wrapper.py

5. **Surrogate Model**
   - Neural net che predice fitness
   - Risparmia chiamate FEM

---

## 📁 FILE DA ELIMINARE/CONSOLIDARE

### Documenti Duplicati

```
llm-evolution/
├── ATOMIC_PROMPTS.md      # Può essere eliminato (task completati)
├── COMPOSER_ATOMIC_TASKS.md  # Può essere eliminato (obsoleto)
└── README.md              # Tenere, aggiornare
```

**Azione**: Eliminare `ATOMIC_PROMPTS.md` e `COMPOSER_ATOMIC_TASKS.md`, tenere solo `README.md` e `STATUS.md`

### File Placeholder/Vuoti

```
./src/agents/crossover.py   # Solo placeholder, da implementare
./src/agents/explainer.py   # Placeholder
./examples/basic_usage.py   # Da verificare
```

---

## 🎯 PRIORITÀ IMMEDIATE

### P0 - Blockers (Oggi)

1. [ ] Avviare Ollama: `ollama serve`
2. [ ] Scaricare modelli: `ollama pull deepseek-r1:14b`
3. [ ] Avviare SurrealDB: `./scripts/setup_surrealdb.sh`
4. [ ] Eseguire ingestion: `python -m src.knowledge.ingestion.llm4ec`

### P1 - Core Missing (Settimana)

1. [ ] **MutationAgent** - LLM-guided mutation
2. [ ] **LMX Crossover** - Semantic crossover via LLM
3. [ ] **MAP-Elites** - Aggiungere a pymoo_wrapper

### P2 - Nice to Have (Dopo)

1. [ ] Surrogate Model (risparmio FEM)
2. [ ] Self-Evolving Agents
3. [ ] Web UI per monitoring

---

## 🧹 CLEANUP COMMAND

```bash
cd /Users/alessioivoycazzaniga/Rememberance/llm-evolution

# Rimuovi documenti obsoleti
rm -f ATOMIC_PROMPTS.md COMPOSER_ATOMIC_TASKS.md

# Verifica struttura pulita
tree -L 3 --dirsfirst
```

---

## 🔧 QUICK START (dopo setup)

```python
import asyncio
from src.agents import OrchestratorAgent
from src.core.coordinator import EvolutionCoordinator
from src.core.genome import GenomeSchema

async def main():
    # 1. Descrivi cosa vuoi ottimizzare
    orchestrator = OrchestratorAgent()
    response = await orchestrator.process("""
        Ottimizza una singing bowl tibetana:
        - Diametro: 18-25 cm
        - Materiale: bronzo
        - Frequenza: 432 Hz con quinta giusta
    """)
    
    schema = response.result.genome_schema
    
    # 2. Definisci fitness
    def fitness(genome):
        # Simulazione FEM o surrogate
        return {"error": 0.1, "sustain": 5.0}
    
    # 3. Evolvi
    coordinator = EvolutionCoordinator(orchestrator=orchestrator)
    result = await coordinator.run(
        schema=schema,
        fitness_fn=fitness,
        max_generations=50,
    )
    
    print(f"Best: {result.best_genome}")

asyncio.run(main())
```

---

*Documento di stato per LLM-Evolution framework*
