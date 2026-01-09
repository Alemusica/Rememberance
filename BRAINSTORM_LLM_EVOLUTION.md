# 🧠 Brainstorming: LLM-Supervised Evolutionary Tibetan Bowl Optimizer

## 🎯 L'IDEA CHIARA (come l'hai descritta)

**Obiettivo**: Una singing bowl tibetana che produce un **accordo di quinta** basato su A432Hz

### Agenti Specializzati (come per la bowl)

| Agente | Ruolo | Analogia Bowl |
|--------|-------|---------------|
| **OrchestratorAgent** | Tiene le fila del contesto globale | La mano che tiene la bowl |
| **ModalAgent** | Analisi modale delle frequenze | Gli armonici della bowl |
| **PokayokeAgent** | Evita errori catastrofici | Il cuscinetto che stabilizza |
| **HarmonyAgent** | Gestisce rapporti A432Hz, quinta giusta | L'intonazione perfetta |
| **PhysicsAgent** | Validazione fisica FEM | La forma del metallo |
| **HumanInTheLoop** | Tu, come supervisore finale | Chi ascolta e decide |

---

## 📊 CONFRONTO: Tuo Script Mirror7 vs Framework Esistenti

### Mirror7 `knowledge_ingest_mirror7.py` - PUNTI DI FORZA

```python
# Il tuo approccio già ha:
QUERY_SETS = {
    "binaural": {...},     # Query semantiche per dominio
    "neuro": {...},        # Categorizzazione intelligente
    "phi": {...},          # Golden ratio focus
    "implementation": {...} # Anche StackExchange!
}

# + Multi-source: ArXiv, PubMed, StackExchange
# + Auto-tagging semantico
# + Rate limiting built-in
# + Escape per SurrealDB
```

### Cosa manca vs LLaMEA/EvoAgentX

| Feature | Mirror7 | LLaMEA | EvoAgentX | Proposta |
|---------|---------|--------|-----------|----------|
| Embeddings | sentence-transformers | ✓ | ✓ | **Già c'è!** |
| Multi-source | ✓ ArXiv/PubMed/SE | Solo arxiv | - | **Superiore** |
| Tagging | Manuale QUERY_SETS | Auto-LLM | Auto | **Ibrido** |
| LLM Query | ❌ | ✓ GPT | ✓ Claude | **Aggiungere** |
| RAG Pipeline | Basico | ✓ | ✓ | **Upgrade** |

---

## 🏗️ ARCHITETTURA PROPOSTA: "Singing Bowl Evolution"

```
┌──────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT (LLM)                       │
│         "La mano che tiene la bowl" - Contesto globale           │
│     DeepSeek R1 14B / Qwen 2.5 32B (reasoning pesante)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ ModalAgent  │  │HarmonyAgent │  │PokayokeAgent│  │ RAGAgent │ │
│  │ FEM modes   │  │ A432Hz+5th  │  │ Error catch │  │ Papers   │ │
│  │ Qwen 7B    │  │ Llama 8B    │  │ Rules only  │  │ nomic    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                      EVOLUTION PIPELINE                           │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐ │
│  │ SEED   │→  │ SPROUT │→  │  GROW  │→  │ BLOOM  │→  │ MATURE │ │
│  │Gen 1-10│   │Gen11-30│   │Gen31-60│   │Gen61-90│   │Gen 90+ │ │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘ │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                       KNOWLEDGE LAYER                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  SurrealDB: 70+ papers + embeddings + LTM distillation      │ │
│  │  + arXiv LLM4EC papers + StackExchange implementation       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                    HUMAN IN THE LOOP                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  • Pokayoke: PAUSE quando anomaly → ask user                │ │
│  │  • Strategy: ogni N generazioni → review                    │ │
│  │  • Final: best genomes → approval                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎵 ESEMPIO CONCRETO: Tibetan Bowl A432Hz + Fifth

### Obiettivo Armonico

```
Frequenza base (A432):     432.00 Hz (A4 - La)
Quinta giusta (E648):      648.00 Hz (E5 - Mi) = 432 × 1.5
Ottava (A864):             864.00 Hz (A5 - La)

Rapporto quinta giusta: 3:2 = 1.5 (NON 1.4983 temperato!)
```

### Come l'LLM Orchestra

1. **ModalAgent** analizza i modi della bowl:
   ```json
   {
     "mode_1": {"freq": 432.3, "damping": 0.02, "shape": "(1,2)"},
     "mode_2": {"freq": 645.8, "damping": 0.03, "shape": "(2,2)"},
     "deviation_from_target": {"432": "+0.3Hz", "648": "-2.2Hz"}
   }
   ```

2. **HarmonyAgent** suggerisce correzione:
   ```json
   {
     "action": "increase_wall_thickness_bottom",
     "reason": "mode_2 troppo basso, aggiungere massa al fondo alza freq",
     "expected_delta": "+1.8Hz"
   }
   ```

3. **OrchestratorAgent** decide:
   ```json
   {
     "accept": true,
     "mutation_type": "guided",
     "genes_to_modify": ["wall_thickness_ratio", "bottom_curvature"],
     "confidence": 0.85
   }
   ```

4. **PokayokeAgent** verifica:
   ```json
   {
     "structural_ok": true,
     "acoustic_valid": true,
     "warnings": ["damping_mode_2 near threshold"]
   }
   ```

---

## 📚 INTEGRAZIONE REPO ESTERNI

### 1. LLaMEA - Cosa Prendere

```python
# Da integrare: Il pattern di "LLM guida l'operatore evolutivo"
class LLMGuidedMutation:
    def __init__(self, llm_client, genome_description):
        self.llm = llm_client
        self.genome_desc = genome_description  # Schema JSON del genome
    
    async def suggest_mutation(self, genome, fitness_history):
        prompt = f"""
        Genome: {genome.to_dict()}
        Fitness trend: {fitness_history[-10:]}
        
        Suggest ONE mutation that could improve fitness.
        Consider physical constraints from the genome description.
        """
        response = await self.llm.generate(prompt)
        return parse_mutation(response)
```

### 2. EvoAgentX - Cosa Prendere

```python
# Da integrare: Self-evolving agent ecosystem
class AgentEcosystem:
    """Pool di agenti che si auto-migliorano."""
    
    def __init__(self):
        self.agents = {
            "mutation": MutationAgent(),
            "crossover": CrossoverAgent(),
            "selection": SelectionAgent(),
        }
        self.performance_log = {}  # Track quale agente aiuta di più
    
    def evolve_agents(self):
        """Meta-evoluzione: agenti che funzionano male vengono modificati."""
        for name, agent in self.agents.items():
            if self.performance_log[name]["improvement_rate"] < 0.1:
                agent.update_prompt()  # LLM riscrive il proprio prompt!
```

### 3. OpenEvolve - Cosa Prendere

```python
# Da integrare: Autonomous algorithm discovery
class AlgorithmDiscovery:
    """LLM che scopre nuovi operatori evolutivi."""
    
    async def discover_crossover(self, parent_genomes, fitness_scores):
        prompt = f"""
        You are an expert in evolutionary algorithms.
        
        Given these parent genomes and their fitness:
        {parent_genomes}
        
        Invent a NEW crossover operator that:
        1. Preserves structural integrity
        2. Combines best features from both parents
        3. Is DIFFERENT from standard 1-point, 2-point, uniform
        
        Output: Python code for the crossover function.
        """
        code = await self.llm.generate(prompt, temperature=0.9)
        return compile_and_validate(code)
```

---

## 🔧 INGESTION UPGRADE: Da Mirror7 a LLM-Enhanced

### Nuovo Script: `ingest_llm4ec_papers.py`

```python
#!/usr/bin/env python3
"""
Ingest LLM4EC papers + auto-embedding + LLM-enhanced tagging.
Basato su knowledge_ingest_mirror7.py ma con LLM enhancement.
"""

# Query set specifico per LLM+Evolution
LLM4EC_QUERIES = {
    "llm_evolution": {
        "description": "LLM-guided evolutionary computation",
        "arxiv": [
            "large language model evolutionary algorithm",
            "LLM mutation operator optimization",
            "prompt optimization evolutionary",
            "LLM crossover genetic algorithm",
            "foundation model evolutionary computation",
        ],
        "semantic_scholar": [
            "neural network guided evolution",
            "transformer optimization algorithm",
        ],
        "tags": ["llm", "evolutionary", "optimization", "foundation-model"],
    },
    
    "self_evolving_agents": {
        "description": "Auto-miglioramento agenti",
        "arxiv": [
            "self-evolving AI agent",
            "autonomous agent improvement",
            "meta-learning agent optimization",
            "agent prompt evolution",
        ],
        "tags": ["agent", "self-evolving", "meta-learning"],
    },
}

async def enhance_with_llm(paper: dict) -> dict:
    """LLM arricchisce i metadati del paper."""
    prompt = f"""
    Paper: {paper['title']}
    Abstract: {paper['abstract'][:500]}
    
    Extract:
    1. Main methodology (GA, ES, LLM, hybrid?)
    2. Key innovation (1 sentence)
    3. Relevance to vibroacoustic optimization (0-10)
    4. Suggested tags (max 5)
    
    JSON output:
    """
    
    response = await llm.generate(prompt)
    enrichment = json.loads(response)
    paper.update(enrichment)
    return paper
```

---

## 🧪 ESPERIMENTO PROPOSTO: "Bowl Optimizer v0.1"

### Step 1: Setup Branch

```bash
git checkout -b llm-evolution-bowl
mkdir -p src/{agents,llm,ingestion,examples}
```

### Step 2: Core Files da Creare

```
llm-evolution-bowl/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent con LLM client
│   │   ├── orchestrator.py      # "La mano che tiene la bowl"
│   │   ├── modal.py             # Analisi FEM modi
│   │   ├── harmony.py           # A432Hz + quinta giusta
│   │   ├── pokayoke.py          # Error catching
│   │   └── rag.py               # Paper retrieval
│   │
│   ├── llm/
│   │   ├── client.py            # Ollama + fallback OpenAI
│   │   ├── prompts.py           # Template Jinja2
│   │   └── cache.py             # Response caching
│   │
│   ├── ingestion/
│   │   ├── arxiv.py             # Basato su Mirror7
│   │   ├── surrealdb.py         # Esistente, da copiare
│   │   └── llm_enhance.py       # NEW: LLM enrichment
│   │
│   └── examples/
│       ├── tibetan_bowl.py      # Bowl A432 + quinta
│       ├── vibroacoustic_plate.py  # DML therapy
│       └── benchmark.py         # Rastrigin, TSP
│
├── tests/
├── requirements.txt
└── README.md
```

### Step 3: Minimo Prodotto Funzionante

```python
# examples/tibetan_bowl.py
"""
Esempio: Ottimizzare una singing bowl per A432Hz + quinta.
"""

from agents import Orchestrator, ModalAgent, HarmonyAgent
from evolution import GeneticAlgorithm

# Definizione genome bowl
class BowlGenome:
    """Parametri fisici singing bowl."""
    diameter: float          # 15-30 cm
    wall_thickness: float    # 2-5 mm
    wall_angle: float        # 0-15 degrees (svasatura)
    bottom_thickness: float  # 3-8 mm
    bottom_curvature: float  # 0-0.1 (piatto→curvo)
    material: str            # "bronze", "brass", "bell_metal"

# Target armonico
TARGET = {
    "fundamental": 432.0,   # Hz (A4)
    "second_partial": 648.0,  # Hz (quinta giusta = 432 * 1.5)
    "tolerance": 1.0,  # Hz
}

# Fitness multi-obiettivo
def fitness(genome: BowlGenome) -> dict:
    """FEM simulation → modal frequencies."""
    modes = fem_simulate(genome)  # Simula con scikit-fem
    
    return {
        "freq_error_fundamental": abs(modes[0].freq - TARGET["fundamental"]),
        "freq_error_fifth": abs(modes[1].freq - TARGET["second_partial"]),
        "sustain": modes[0].damping,  # Più basso = meglio
        "mass": compute_mass(genome),  # Minimizzare
    }

# Setup agenti
orchestrator = Orchestrator(model="deepseek-r1:14b")
modal_agent = ModalAgent(model="qwen2.5:7b")
harmony_agent = HarmonyAgent(model="llama3.2:3b")

# Run evoluzione
ga = GeneticAlgorithm(
    genome_class=BowlGenome,
    fitness_fn=fitness,
    agents=[orchestrator, modal_agent, harmony_agent],
    population_size=30,
    generations=100,
)

best = ga.run()
print(f"Best bowl: {best.genome}")
print(f"Frequencies: {best.modes}")
```

---

## ❓ DOMANDE PER TE (Human in the Loop!)

1. **LLM Preferito**
   - DeepSeek R1 (14B) - ottimo reasoning, gratis
   - Qwen 2.5 Coder (32B) - migliore per code
   - Mix locale + cloud (fallback OpenAI/Claude)?

2. **Priorità Agenti**
   - Tutti 5 subito, o iniziare con 2-3?
   - Quale consideri più critico? (Orchestrator? Modal? Pokayoke?)

3. **Knowledge Base**
   - Usare SurrealDB esistente (research.db) o nuova istanza?
   - Importare tutti i paper LLM4EC (~150 papers)?

4. **Target Primo Esperimento**
   - Tibetan Bowl (semplice, 5 parametri)?
   - Piatto vibroacustico (complesso, quello esistente)?
   - Benchmark standard (Rastrigin, per validare)?

5. **Token Budget**
   - Aggressivo (ogni mutation chiede all'LLM)?
   - Conservativo (solo ogni N generazioni)?
   - Adaptive (più LLM quando stallo)?

---

## 📅 TIMELINE PROPOSTA

### Settimana 1: Foundation
- [ ] Branch `llm-evolution-bowl`
- [ ] Copia core da `binaural_golden/src/core/`
- [ ] Setup `src/llm/client.py` con Ollama
- [ ] Primo test: LLM risponde a prompt evoluzione

### Settimana 2: Agenti Base
- [ ] `BaseAgent` + `Orchestrator`
- [ ] `MutationAgent` (physics-aware)
- [ ] Test con Rastrigin benchmark

### Settimana 3: Knowledge Integration
- [ ] Ingest LLM4EC papers
- [ ] RAG pipeline per agents
- [ ] Embeddings automatici

### Settimana 4: Tibetan Bowl
- [ ] `BowlGenome` definizione
- [ ] FEM integration (scikit-fem)
- [ ] `HarmonyAgent` per A432Hz
- [ ] Demo funzionante!

---

*Brainstorm generato: 2025-01-09*
*Per: Alessio Ivoy Cazzaniga*
*Progetto: Golden Sound / Singing Bowl Optimizer*
