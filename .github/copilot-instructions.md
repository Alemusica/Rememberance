# Golden Studio - Copilot Instructions

## Project Overview

Golden Studio is a **vibroacoustic therapy application** that designs **DML plates (Distributed Mode Loudspeakers)** to deliver precise acoustic energy to specific body zones (ears, spine, etc.).

## Core Philosophy

### Peninsula = Acoustic Black Holes (ABH)
**DO NOT PENALIZE PENINSULAS** - They can focus acoustic energy like ABH (Krylov 2014).
- Edge/corner peninsulas are BENEFICIAL for energy focusing
- Net score: `abh_benefit * 0.6 + resonator_potential * 0.3 - structural_penalty * 0.5`
- Reference: Deng et al. 2019, Zhao 2014, Feurtado 2017

### Multi-Exciter Optimization
Based on Chinese research (Lu, Shen, Bai):
- Multiple exciters improve frequency response flatness
- Genetic algorithms for optimal placement (Bai & Liu 2004)
- Modal cross-coupling must be minimized (Sum & Pan 2000)

### Target Metrics
1. **Ear zones**: Flat frequency response (< 6dB variation), L/R uniformity > 90%
2. **Spine zone**: Maximum energy 0-300Hz, flat response (< 10dB variation)

## Architecture

```
src/
├── core/                    # Physics & algorithms
│   ├── fitness.py          # Multi-objective evaluation (ABH benefit!)
│   ├── structural_analysis.py  # Peninsula detection → ABH benefit
│   ├── evolutionary_optimizer.py  # NSGA-II genetic algorithm
│   ├── plate_genome.py     # Plate DNA (contour, exciters, thickness)
│   ├── plate_physics.py    # Modal analysis, frequency response
│   └── fem/                # Finite Element solvers
├── ui/
│   ├── plate_designer_tab.py  # Main tab (MVVM)
│   ├── viewmodels/         # Business logic
│   └── components/         # Reusable widgets
└── utils/
    ├── knowledge_base.py   # SurrealDB research queries
    └── research_assistant.py  # Distilled paper insights
```

## Key APIs

### PlateGenome
```python
genome = PlateGenome(
    width=0.4, height=0.6,  # meters
    contour=ContourType.RECTANGLE,
    exciters=[ExciterPosition(x, y, diameter)],  # NOT exciter_positions!
    thickness_variation=0.2  # 0=uniform, 0.2=20% taper (NOT thickness_profile enum!)
)
```

### FitnessEvaluator
```python
evaluator = FitnessEvaluator(person, weights)
result = evaluator.evaluate(genome)
# result.peninsula_net_score > 0 means peninsula is BENEFICIAL
# result.ear_uniformity should be > 0.9
```

## SurrealDB Knowledge Base
- **URL**: http://localhost:8000
- **Auth**: root:root (Basic: cm9vdDpyb290)
- **Headers**: `surreal-ns: research`, `surreal-db: knowledge`

### Tables
| Table | Content |
|-------|---------|
| `paper` | 72+ paper scientifici con cite_key, title, abstract, section, project_relevance |
| `algorithm` | Algoritmi validati con success_rate e paper_sources (tracciabilità ai paper ispiratori) |
| `concept` | Concetti distillati dalla ricerca |

### Query Esempi
```sql
-- Cerca paper
SELECT * FROM paper WHERE title CONTAINS 'genetic';

-- Algoritmi con successo > 80%
SELECT * FROM algorithm WHERE success_rate > 0.8;

-- Quali algoritmi derivano da un paper?
SELECT * FROM algorithm WHERE paper_sources CONTAINS 'bai2004genetic';

-- Dettagli algoritmo
SELECT * FROM algorithm:nsga2_plate_optimizer;
```

### MCP Tools Disponibili
- `search_papers(query)` - Cerca paper per keyword
- `get_algorithms(domain, min_success_rate)` - Algoritmi validati
- `get_algorithm_details(id)` - Dettagli con paper sources
- `find_algorithms_by_paper(cite_key)` - Algoritmi ispirati da un paper

## Tests
```bash
cd binaural_golden
pytest tests/test_evolution_validation.py -v
```
- Current ear_uniformity: 0.06 (target > 0.9) - OPTIMIZATION NEEDED
- Peninsula benefit test: PASSING (+0.27 for corner peninsula)

## Research Papers (Key)
| Topic | Authors | Key Finding |
|-------|---------|-------------|
| Multi-exciter | Lu 2012, Shen 2016 | Attached masses + multi-exciter improve flatness |
| Exciter placement | Bai & Liu 2004 | Genetic algorithm for optimal positions |
| ABH | Krylov 2014, Deng 2019 | Tapered regions focus energy (don't penalize!) |
| Modal coupling | Sum & Pan 2000 | Cross-coupling affects zone response |

## Common Mistakes to Avoid
1. ❌ Using `exciter_positions` → ✅ Use `exciters` with `ExciterPosition`
2. ❌ Using `thickness_profile` enum → ✅ Use `thickness_variation` float (0-1)
3. ❌ Penalizing all peninsulas → ✅ Calculate ABH benefit first
4. ❌ Ignoring L/R ear uniformity → ✅ It's the main optimization target

## Session History
See `docs/SESSION_HISTORY.md` for detailed conversation logs.
