# Golden Studio 🌀

> **Evolutionary optimization framework for vibroacoustic therapy plates using physics-informed multi-objective algorithms**

[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NSGA-II](https://img.shields.io/badge/optimizer-NSGA--II-green.svg)](https://pymoo.org/)

---

## 🎯 What is this?

Golden Studio is a **novel hybrid evolutionary framework** that combines:

- **Multi-objective NSGA-II** for Pareto optimization
- **Curriculum learning** via staged gene activation (SEED→SPROUT→GROW→BLOOM)
- **Neural memory (RDNN)** for trajectory prediction across optimization runs
- **Long-term knowledge distillation (LTM)** for cross-run learning
- **Physics-informed fitness** based on FEM modal analysis

The goal: **optimize DML (Distributed Mode Loudspeaker) plates** for vibroacoustic therapy, where the human body is the "string to tune".

---

## 🏗️ Architecture

```
binaural_golden/src/core/
├── OPTIMIZATION LAYER
│   ├── unified_optimizer.py    # Strategy pattern (GENETIC/NSGA2/SIMP/HYBRID)
│   ├── agnostic_evolution.py   # Domain-agnostic interfaces
│   ├── evolution_pipeline.py   # Full pipeline: Pokayoke → Physics → RDNN → LTM
│   └── exciter_gene.py         # GenePhase curriculum learning
│
├── PHYSICS LAYER
│   ├── plate_physics.py        # Analytical modal analysis
│   ├── jax_plate_fem.py        # JAX-accelerated FEM (autodiff)
│   └── materials.py            # Spruce, Baltic birch, MDF...
│
├── FITNESS LAYER
│   ├── fitness.py              # Multi-objective evaluator
│   ├── scorers/                # Modular scoring (ear_uniformity, spine_coupling...)
│   └── structural_analysis.py  # ABH detection, peninsula analysis
│
├── MEMORY LAYER
│   ├── evolution_memory.py     # Short-term (ring buffer) + Long-term
│   └── rdnn_memory.py          # LSTM/GRU for trajectory prediction
│
└── EXPORT LAYER
    ├── stl_export.py           # 3D STL for CNC
    └── virtual_cnc.py          # G-code generation
```

---

## 🔬 Research Foundation

Built on **70+ distilled research papers** covering:

| Domain | Papers | Key References |
|--------|--------|----------------|
| Multi-exciter DML | 8 | Lu 2012, Bai & Liu 2004 |
| Acoustic Black Holes | 7 | Krylov 2014, Deng 2019 |
| Lutherie | 12 | Schleske 2002, Woodhouse 2014 |
| Human Body Resonance | 8 | Griffin 1990 (spine 10-12Hz, chest 50-60Hz) |
| Vibroacoustic Therapy | 7 | Skille 1989 (30-120Hz), Boyd-Brewer 2003 |

Full bibliography: [`binaural_golden/docs/research/vibroacoustic_references.bib`](binaural_golden/docs/research/vibroacoustic_references.bib)

---

## ✨ Key Features

### 🧬 Curriculum Learning (GenePhase)
```python
class GenePhase(Enum):
    SEED = 0    # Position-only mutations (exploration)
    SPROUT = 1  # + diameter mutations
    GROW = 2    # + exciter count mutations
    BLOOM = 3   # Full mutation (position + diameter + count + emission)
```
Inspired by NEAT (Stanley 2002) and curriculum learning (Bengio 2009).

### 🧠 Neural Memory (RDNN)
```python
# LSTM/GRU maintains hidden state across optimization runs
memory = RDNNMemory(architecture=RDNNArchitecture.GRU, hidden_size=64)
memory.observe(fitness_trajectory, physics_features)
suggestions = memory.suggest()  # Adaptive mutation rate, search direction
```

### 📊 Multi-Objective Pareto
```python
objectives = ObjectiveVector(
    spine_flatness=0.85,      # 20-200Hz response at spine
    ear_lr_uniformity=0.97,   # L/R balance for binaural
    structural_safety=1.0,    # Deflection < 10mm
    abh_benefit=0.72,         # Acoustic Black Hole energy focusing
)
# NSGA-II finds Pareto-optimal trade-offs
```

### 🎻 Physics-Informed Fitness
- Modal analysis via analytical Kirchhoff or JAX-accelerated FEM
- Zone-specific response: spine (tactile 20-200Hz) vs head (audio 50-8kHz)
- ABH peninsula detection for energy focusing (Krylov 2014)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Alemusica/Rememberance.git
cd Rememberance/binaural_golden

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the GUI

```bash
python src/golden_studio.py
```

### Run Optimization (CLI)

```python
from src.core.person import Person
from src.core.unified_optimizer import OptimizationStrategy
from src.core.plate_unified import create_plate_optimization_system

# Create person model (the "string to tune")
person = Person(height_m=1.80, weight_kg=75)

# Create optimizer
optimizer, config = create_plate_optimization_system(
    person,
    strategy=OptimizationStrategy.NSGA2,
    use_memory=True
)

# Run evolution
result = optimizer.optimize(config)
print(f"Best fitness: {result.best_fitness:.4f}")
print(f"Pareto front size: {len(result.pareto_front)}")
```

---

## 📐 The Mathematics

### Golden Ratio Constants
```
φ (Phi)           = 1.618033988749895...  (Golden Ratio)
φ conjugate       = 0.618033988749895...  (1/φ = φ-1)
Golden angle      = 2π/φ² ≈ 137.5°
```

### Zone Frequency Targets

| Zone | Frequency Range | Purpose |
|------|-----------------|---------|
| Spine | 20-200 Hz | Tactile vibration therapy |
| Head/Ears | 50-8000 Hz | Binaural audio reproduction |
| Chest | 50-60 Hz | Resonance coupling |

### ABH (Acoustic Black Hole) Profile
```python
h(x) = h₀ × (x / x_abh)^m   # m ≥ 2 for wave trapping
```

---

## 🧪 Tests

```bash
cd binaural_golden

# Run all tests
pytest tests/ -v

# Run physics validation tests
pytest tests/test_physics_validation.py -v  # 11/11 pass

# Run evolution pipeline tests
pytest tests/test_evolution_pipeline.py -v
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](binaural_golden/CONTRIBUTING.md) for guidelines.

Areas of interest:
- [ ] Ray parallelization for distributed evaluation
- [ ] Quality-Diversity (MAP-Elites) for zone-specific exploration
- [ ] Additional domain adapters (singing bowls, speaker enclosures)
- [ ] Web-based UI

---

## 📚 Citation

If you use this work in research, please cite:

```bibtex
@software{golden_studio_2025,
  author = {Cazzaniga, Alessio Ivoy},
  title = {Golden Studio: Evolutionary Optimization for Vibroacoustic Plates},
  year = {2025},
  url = {https://github.com/Alemusica/Rememberance}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **pymoo** team for NSGA-II implementation
- **scikit-fem** for finite element solvers
- All researchers whose papers informed this work (see bibliography)

---

*"Geometry has two great treasures: one is the theorem of Pythagoras; the other, the division of a line into extreme and mean ratio (golden ratio). The first we may compare to a measure of gold; the second we may name a precious jewel."*
— Johannes Kepler

✦ φ ✦
