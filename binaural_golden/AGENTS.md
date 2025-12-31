# Agent Instructions - Golden Studio

## Knowledge Base Access

Questo progetto ha una **knowledge base SurrealDB** con 72 paper di ricerca su:
- Vibroacoustic therapy
- DML plate design  
- Multi-exciter optimization
- Acoustic Black Holes (ABH)
- Modal analysis

### Come Accedere

**Via MCP Tools** (se disponibili):
- `search_papers(query)` - Cerca per keyword
- `get_papers_by_section(section)` - Paper per sezione
- `get_paper_details(cite_key)` - Dettagli completi
- `get_key_papers()` - Paper fondamentali

**Via HTTP** (sempre disponibile):
```bash
curl -X POST "http://localhost:8000/sql" \
  -H "Authorization: Basic cm9vdDpyb290" \
  -H "surreal-ns: research" \
  -H "surreal-db: knowledge" \
  --data "SELECT * FROM paper WHERE title CONTAINS 'keyword'"
```

### Sezioni Disponibili

| Sezione | Papers | Contenuto |
|---------|--------|-----------|
| MULTI_EXCITER | 8 | Ottimizzazione multi-eccitatore |
| LUTHERIE | 12 | Liuteria, design strumenti |
| ACOUSTIC_BLACK | 7 | ABH, energy focusing |
| HUMAN_BODY | 8 | Risonanze corpo umano |
| VIBROACOUSTIC | 7 | Terapia vibroacustica |

### Paper Chiave

- `bai2004genetic` - NSGA-II genetic algorithm (BASE del nostro optimizer)
- `krylov2014abh` - Acoustic Black Holes theory
- `skille1989vibroacoustic` - VAT founder (30-120Hz)
- `griffin1990handbook` - Body resonance (spine 10-12Hz)

### Quando Consultare la KB

✅ Domande su fisica delle placche DML
✅ Ottimizzazione posizione eccitatori  
✅ Frequenze risonanza corpo umano
✅ Algoritmi genetici per audio
✅ Acoustic Black Holes / peninsulas

## New Modules (December 2024)

### Evolution Memory System

**File**: `src/core/evolution_memory.py`

Neural-network inspired memory for evolutionary optimization with:

- **ShortTermMemory**: Ring buffer with 3-5-10 generation windows for trajectory analysis
- **LongTermMemory**: Pattern storage, experience archive, distillation
- **EvolutionMemory**: Unified interface combining STM + LTM

```python
from src.core.evolution_memory import EvolutionMemory

memory = EvolutionMemory("./memory")
memory.record_generation(gen, fitnesses, objectives, best_genome)
analysis = memory.get_trajectory_analysis()  # Trend, velocity, recommendations
memory.finalize_run(final_fitness, final_objectives, genome_summary, "success")
```

### Freeform Cutouts (Topology Optimization)

**File**: `src/core/freeform_cutout.py`

Topology-style free-form shape generation for CNC-ready cutouts:

- **DensityField**: SIMP method ρ(x,y) ∈ [0,1] for continuous optimization
- **FreeformCutout**: Bezier/B-spline parametric shapes
- **FreeformCutoutOptimizer**: Combined topology + parametric optimization

```python
from src.core.freeform_cutout import FreeformCutoutOptimizer

optimizer = FreeformCutoutOptimizer(plate_dims=(0.6, 0.4), target_area=0.01)
cutout = optimizer.optimize_for_frequency(target_freq=150, physics_engine=physics)
contour = cutout.get_contour(n_points=100)  # CNC-ready
```

### Agnostic Evolution Framework

**Files**: 
- `src/core/agnostic_evolution.py` - Abstract base classes
- `src/core/plate_adapters.py` - Plate-specific implementations

Goal-independent evolutionary optimization that works for ANY vibroacoustic design:

```python
# Use with plates
from src.core.plate_adapters import create_plate_optimizer
from src.core.person import Person

person = Person(height_m=1.75, weight_kg=70)
optimizer = create_plate_optimizer(person)
result = optimizer.optimize()

# Use with singing bowls (example)
from src.core.agnostic_evolution import SingingBowlGenome
bowl = SingingBowlGenome(diameter=0.15, height=0.08)
mutated = bowl.mutate(0.1)
```

**Architecture**:
- `PhysicsEngine[G, P]` - Abstract physics simulation
- `FitnessEvaluator[G, P, F]` - Abstract fitness evaluation
- `GenomeFactory[G, C]` - Abstract genome creation
- `AgnosticEvolutionaryOptimizer` - Generic optimizer

### Physics Validation Tests

**File**: `tests/test_physics_validation.py`

11 tests verifying cutout placement follows modal physics:

```bash
pytest tests/test_physics_validation.py -v  # All 11/11 pass
```

Tests cover:
- Mode shape antinode detection
- ABH placement at corners/edges
- Spine zone avoidance
- Symmetry detection
- Multi-mode prioritization
