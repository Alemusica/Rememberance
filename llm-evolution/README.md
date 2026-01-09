# 🧬 LLM-Evolution

**Agnostic Evolutionary Optimization with LLM Agents**

Un framework dove LLM agents supervisionano algoritmi evolutivi, creando una sinergia tra reasoning elastico (LLM) e execution rigida (neural networks + simulators).

## 🎯 La Metafora

```
┌─────────────────────────────────────────┐
│          HUMAN IN THE LOOP              │
│  "Voglio una bowl A432Hz, bronzo, 20cm" │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       CORTECCIA PREFRONTALE             │
│     OrchestratorAgent (LLM 32B)         │
│  Pianifica, istruisce, mantiene context │
└─────────────────┬───────────────────────┘
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│Strategy │  │Analysis │  │   RAG   │
│ Agent   │  │ Agent   │  │  Agent  │
│  (14B)  │  │  (7B)   │  │(embed)  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│          MIDOLLO SPINALE                │
│    Evolution Coordinator (Python)       │
│   Traduce LLM decisions → GA params     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│             MUSCOLI                     │
│  Neural Networks + Simulators + Tools   │
│  RDNN | FEM | CNC Virtual | Surrogate   │
└─────────────────────────────────────────┘
```

## ✨ Features

- **Agnostico**: Ottimizza qualsiasi cosa descrivibile con parametri
- **LLM-Guided**: Agents supervisionano mutation, crossover, selection
- **Tool Discovery**: Se manca un tool, il sistema lo richiede
- **Knowledge RAG**: Papers scientifici iniettati nel context
- **Human in the Loop**: Pokayoke per anomalie, approval per decisioni critiche

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/youruser/llm-evolution.git
cd llm-evolution

# Install
pip install -r requirements.txt

# Ensure Ollama is running with models
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:14b
ollama pull llama3.2:3b

# Run example
python -m src.domains.tibetan_bowl
```

## 📖 Usage

### 1. Descrivi cosa vuoi ottimizzare

```python
from src.agents import OrchestratorAgent

orchestrator = OrchestratorAgent()

# Descrizione naturale
description = """
Voglio ottimizzare una singing bowl tibetana.
- Diametro: 18-25 cm
- Materiale: bronzo (8% stagno)
- Frequenza fondamentale: 432 Hz (A4)
- Seconda armonica: quinta giusta (648 Hz)
- Produzione: 3D printing in cera persa
"""

# L'orchestrator genera tutto
response = await orchestrator.process(description)
domain_spec = response.result

print(domain_spec.genome_schema)  # Schema parametri
print(domain_spec.objectives)     # Obiettivi fitness
print(domain_spec.required_tools) # Tool necessari
```

### 2. L'evoluzione è agnostica

```python
from src.core import GenomeSchema, random_genome, mutate_gaussian

# Il core non sa cosa sia una "bowl"
# Sa solo che ci sono parametri con bounds

schema = domain_spec.genome_schema
population = [random_genome(schema) for _ in range(50)]

# Mutazione standard
mutated = mutate_gaussian(population[0], sigma=0.1)
```

### 3. Gli agenti supervisionano

```python
from src.agents import StrategyAgent

strategy = StrategyAgent()

# Quando l'evoluzione stalla...
context = {
    "generation": 50,
    "stall_count": 10,
    "fitness_history": [0.8, 0.81, 0.81, 0.81, 0.81],
}

response = await strategy.process(
    {"type": "suggest_adjustment"},
    context=context
)

# L'LLM suggerisce azioni
print(response.result)
# {"increase_mutation": True, "inject_diversity": True, ...}
```

## 🏗️ Architecture

```
llm-evolution/
├── src/
│   ├── core/           # Agnostic evolution (genome, operators)
│   ├── agents/         # LLM agents (orchestrator, strategy, analysis)
│   ├── memory/         # RDNN trajectory, LTM distillation
│   ├── tools/          # Simulators, neural surrogates
│   ├── llm/            # Ollama client
│   ├── knowledge/      # SurrealDB, RAG
│   └── domains/        # Example domains (bowl, plate, benchmark)
├── tests/
├── config/
└── requirements.txt
```

## 🧠 Modelli Consigliati

| Ruolo | Modello | RAM | Note |
|-------|---------|-----|------|
| Orchestrator | deepseek-r1:14b | 16GB | Reasoning complesso |
| Strategy | qwen2.5-coder:14b | 16GB | Code + parametri |
| Analysis | llama3.2:3b | 4GB | Veloce, interpretazione |
| Fast | qwen2.5:3b | 4GB | Decisioni semplici |

## 📚 Knowledge Base

Il sistema usa SurrealDB per:
- Papers scientifici (ArXiv, PubMed)
- Embeddings per RAG
- Long-Term Memory distillation

```bash
# Start SurrealDB
surreal start --log warn --user root --pass root file:~/.config/surrealdb/evolution.db

# Ingest papers
python -m src.knowledge.ingestion.llm4ec
```

## 🎵 Esempio: Tibetan Bowl A432Hz

```python
# L'umano dice:
"Voglio una bowl che faccia A432Hz con quinta giusta"

# L'orchestrator genera:
genome_schema = {
    "name": "tibetan_bowl",
    "genes": [
        {"name": "diameter_mm", "type": "float", "min": 150, "max": 300},
        {"name": "wall_thickness_mm", "type": "float", "min": 2, "max": 8},
        {"name": "wall_angle_deg", "type": "float", "min": 0, "max": 25},
        {"name": "bottom_thickness_mm", "type": "float", "min": 3, "max": 12},
        {"name": "rim_profile", "type": "categorical", "categories": ["flat", "rounded", "thickened"]},
    ]
}

objectives = [
    {"name": "freq_error_432", "type": "minimize", "weight": 0.4},
    {"name": "freq_error_648", "type": "minimize", "weight": 0.3},
    {"name": "sustain_s", "type": "maximize", "weight": 0.2},
    {"name": "mass_kg", "type": "minimize", "weight": 0.1},
]

required_tools = ["fem_modal_analysis", "mesh_generator"]
```

## 📖 Documentazione

| Doc | Descrizione |
|-----|-------------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Installazione completa passo-passo |
| [STATUS.md](STATUS.md) | Stato attuale e checklist |
| [STATE_OF_ART.md](STATE_OF_ART.md) | Confronto con repo di riferimento |

## 🔬 Research Basis

- [LLM4EC Survey](https://github.com/wuxingyu-ai/LLM4EC) - IEEE TEVC
- [LLaMEA](https://github.com/XAI-liacs/LLaMEA) - LLM-guided metaheuristics
- [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) - Self-evolving agents
- [OpenEvolve/FunSearch](https://arxiv.org/abs/2312.03130) - Algorithm discovery

## 📝 License

MIT

---

*Created by Alessio Ivoy Cazzaniga*
