# 🔬 State of the Art Analysis - LLM-Guided Evolution

> Analisi comparativa dei repository di riferimento e feature da integrare

---

## 📚 Repository Analizzati

### 1. LLM4EC (Survey IEEE TEVC)
**URL**: `github.com/wuxingyu-ai/LLM4EC`

**Cosa fa**: Survey completo di 12+ pattern per integrare LLM in EC

| Pattern | Descrizione | Nostro Status |
|---------|-------------|---------------|
| **OPRO** | LLM come ottimizzatore (prompt optimization) | ⚠️ Orchestrator base |
| **LMX** | LLM Crossover semantico | ❌ Manca |
| **LEO** | LLM per multi-objective | ⚠️ PyMoo tools |
| **EoH** | Evolution of Heuristics | ❌ Manca |
| **AEL** | Algorithm Evolution via LLM | ❌ Manca |
| **FunSearch** | Function discovery | ❌ Manca |

**Da prendere**:
- Pattern **LMX**: crossover che comprende semantica
- Pattern **EoH**: LLM genera nuove euristiche
- Prompts template per mutation/crossover

---

### 2. LLaMEA (Leiden AI)
**URL**: `github.com/XAI-liacs/LLaMEA`

**Cosa fa**: LLM genera intere metaeuristiche (codice Python)

| Feature | Descrizione | Nostro Status |
|---------|-------------|---------------|
| Code Generation | LLM scrive optimizer da zero | ❌ Manca |
| Automatic Benchmarking | Test su BBOB/CEC | ❌ Manca |
| Iterative Improvement | Feedback loop | ⚠️ Parziale |

**Da prendere**:
- Struttura prompt per code generation
- Auto-evaluation framework
- BBOB integration per benchmark

---

### 3. EvoAgentX
**URL**: `github.com/EvoAgentX/EvoAgentX`

**Cosa fa**: Multi-agent che si auto-evolvono

| Feature | Descrizione | Nostro Status |
|---------|-------------|---------------|
| Self-Evolving Prompts | Agenti migliorano propri prompt | ❌ Manca |
| Agent Population | Pool di agenti che evolvono | ❌ Manca |
| Task-Adaptive | Agenti si specializzano | ❌ Manca |

**Da prendere**:
- Self-improvement loop per agenti
- Population-based agent training
- Non prioritario ora (troppo complesso)

---

### 4. OpenEvolve / FunSearch (Google)
**URL**: (interno Google, paper pubblico)

**Cosa fa**: Scopre nuovi algoritmi via evolution

| Feature | Descrizione | Nostro Status |
|---------|-------------|---------------|
| Function Discovery | LLM + EA trovano funzioni nuove | ❌ Manca |
| Island Model | Popolazioni parallele | ❌ Manca |
| Code Execution | Sandbox per testare codice | ❌ Manca |

**Da prendere**:
- Island model per diversità
- Sandbox execution per codice generato
- Scoring basato su execution

---

### 5. Awesome-Self-Evolving-Agents
**URL**: `github.com/Awesome-Self-Evolving-Agents`

**Cosa fa**: Curated list di self-evolving agent papers

**Da prendere**:
- Reference papers per LTM
- Architetture memory-augmented
- Benchmark metodologies

---

### 6. LLM_EA / LLM-Guided-Evolution
**URL**: Vari repo

**Cosa fa**: EA classico con LLM per mutation/selection

| Feature | Descrizione | Nostro Status |
|---------|-------------|---------------|
| Surrogate Fitness | LLM predice fitness | ❌ Manca |
| Adaptive Operators | Scelta operatore via LLM | ⚠️ Strategy base |
| Semantic Mutation | Mutation guidata da LLM | ❌ Manca |

**Da prendere**:
- Surrogate model LLM-based
- Adaptive operator selection
- Semantic mutation prompts

---

## 🎯 Priorità Integrazione

### P0 - CRITICO (Prossima settimana)

1. **LMX Crossover** (da LLM4EC)
   ```
   File: src/agents/crossover.py
   Descrizione: LLM comprende due genomi, genera figlio semanticamente coerente
   Prompt: "Given parent A={...} and parent B={...}, create offspring that combines..."
   ```

2. **Mutation Agent** (da LLM_EA)
   ```
   File: src/agents/mutation.py (NUOVO)
   Descrizione: LLM decide quali geni mutare e come
   Prompt: "This genome has fitness {f}. Suggest mutations to improve..."
   ```

3. **Surrogate Fitness** (da LLM_EA)
   ```
   File: src/tools/evolution/surrogate.py (NUOVO)
   Descrizione: LLM stima fitness senza FEM costoso
   Prompt: "Predict fitness for genome {...} based on previous evaluations..."
   ```

### P1 - IMPORTANTE (Prossimo mese)

4. **MAP-Elites/QD** (da literature)
   ```
   File: src/tools/evolution/mapelites.py (NUOVO)
   Descrizione: Quality-Diversity per esplorare behavior space
   ```

5. **Auto-Benchmark** (da LLaMEA)
   ```
   File: tests/benchmark_suite.py (NUOVO)
   Descrizione: BBOB/CEC test per validare
   ```

6. **Island Model** (da OpenEvolve)
   ```
   File: src/core/island_coordinator.py (NUOVO)
   Descrizione: Popolazioni parallele con migration
   ```

### P2 - NICE TO HAVE (Futuro)

7. **Self-Evolving Agents** (da EvoAgentX)
8. **Code Generation** (da LLaMEA)
9. **Function Discovery** (da FunSearch)

---

## 📋 Nuovi File da Creare

```
llm-evolution/
├── src/
│   ├── agents/
│   │   ├── mutation.py        # P0 - LLM-guided mutation
│   │   └── crossover.py       # P0 - Implementare LMX
│   └── tools/
│       └── evolution/
│           ├── surrogate.py   # P0 - Surrogate fitness LLM
│           └── mapelites.py   # P1 - Quality-Diversity
└── tests/
    └── benchmark_suite.py     # P1 - Auto-benchmarking
```

---

## 📊 Comparison Matrix

| Feature | LLM4EC | LLaMEA | EvoAgentX | Nostro | Gap |
|---------|--------|--------|-----------|--------|-----|
| LLM Orchestration | ✅ | ✅ | ✅ | ✅ | - |
| Semantic Crossover | ✅ LMX | ✅ | - | ❌ | HIGH |
| Semantic Mutation | ✅ | ✅ | - | ❌ | HIGH |
| Surrogate Model | ✅ | - | - | ❌ | HIGH |
| Multi-Objective | ✅ LEO | - | - | ✅ PyMoo | - |
| Self-Evolution | - | - | ✅ | ❌ | LOW |
| Code Generation | - | ✅ | - | ❌ | LOW |
| RAG Knowledge | - | - | - | ✅ | - |
| Quality-Diversity | - | - | - | ❌ | MED |

---

## 🔧 Implementazione Suggerita

### LMX Crossover (esempio prompt)

```python
LMX_PROMPT = """
You are a genetic algorithm crossover operator.

Parent A (fitness={fa}):
{genome_a}

Parent B (fitness={fb}):
{genome_b}

Problem context: {context}

Generate a child genome that:
1. Inherits beneficial traits from the fitter parent
2. Explores novel combinations
3. Maintains structural validity

Output JSON:
{{"child": {{...}}, "reasoning": "..."}}
"""
```

### Mutation Agent (esempio prompt)

```python
MUTATION_PROMPT = """
Genome to mutate:
{genome}

Current fitness: {fitness}
Target objectives: {objectives}
Fitness history: {history}

Suggest mutations:
1. Which parameters to change?
2. In which direction?
3. By how much?

Output JSON:
{{"mutations": [{{"param": "...", "delta": ...}}], "reasoning": "..."}}
"""
```

### Surrogate Fitness (esempio)

```python
SURROGATE_PROMPT = """
Training data (genome -> fitness):
{training_samples}

New genome to predict:
{genome}

Estimate fitness and confidence.

Output JSON:
{{"predicted_fitness": ..., "confidence": 0.0-1.0, "reasoning": "..."}}
"""
```

---

## 🎯 Prossimi Task Atomici

| # | Task | File | Model | Stima |
|---|------|------|-------|-------|
| 1 | Implementare LMX Crossover | `crossover.py` | Sonnet 4.5 | 30min |
| 2 | Creare MutationAgent | `mutation.py` | GPT-4.5 | 30min |
| 3 | Surrogate Fitness Tool | `surrogate.py` | Sonnet 4.5 | 45min |
| 4 | MAP-Elites wrapper | `mapelites.py` | Composer | 30min |
| 5 | Benchmark suite | `benchmark_suite.py` | Composer | 45min |

---

*Documento di analisi stato dell'arte - LLM-Evolution*
