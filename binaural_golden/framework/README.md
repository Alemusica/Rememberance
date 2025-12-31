# Evolutionary Optimization Framework Template

## Overview

This directory contains domain-agnostic evolutionary optimization tools
that can be adapted for any multi-objective design problem.

## Structure

```
framework/
├── core/                      # Domain-agnostic optimization engine
│   ├── agnostic_evolution.py  # Abstract base classes (Genome, PhysicsEngine, etc.)
│   ├── evolution_pipeline.py  # Pipeline orchestration
│   ├── nsga2.py               # NSGA-II implementation
│   └── memory/                # STM + LTM components
│
├── adapters/                  # Domain-specific implementations
│   ├── __init__.py
│   └── dml_plate/             # DML plate adapter (current domain)
│
├── config/                    # Configuration templates
│   └── domain_config.yaml
│
└── examples/                  # Example domain implementations
```

## Usage

1. **Define your domain**: Create an adapter that implements:
   - `Genome` protocol (your design representation)
   - `PhysicsEngine` (your simulation/analysis)
   - `FitnessEvaluator` (your objectives)

2. **Configure**: Set your domain in `config/domain_config.yaml`

3. **Run**: Use `EvolutionPipeline` to optimize

## Key Abstractions

### Genome Protocol
```python
class Genome(Protocol):
    def mutate(self, sigma: float) -> 'Genome': ...
    def crossover(self, other: 'Genome') -> 'Genome': ...
    def to_dict(self) -> Dict[str, Any]: ...
```

### PhysicsEngine ABC
```python
class PhysicsEngine(ABC, Generic[G, P]):
    @abstractmethod
    def analyze(self, genome: G) -> P: ...
```

### FitnessEvaluator ABC
```python
class FitnessEvaluator(ABC, Generic[G, P, F]):
    @abstractmethod
    def evaluate(self, genome: G, physics_result: P = None) -> F: ...
```

## Supported Features

- ✅ Multi-objective NSGA-II optimization
- ✅ Curriculum learning (staged gene activation)
- ✅ Short-term memory (trajectory analysis)
- ✅ Long-term memory (cross-run knowledge)
- ✅ Anomaly detection (Pokayoke Observer)
- ✅ Vectorized fitness evaluation (188x speedup)

## Example Domains

| Domain | Description | Status |
|--------|-------------|--------|
| DML Plate | Vibroacoustic therapy bed | ✅ Implemented |
| Singing Bowl | Tibetan bowl optimization | 📋 Template |
| Speaker Box | Loudspeaker enclosure | 📋 Template |

## License

MIT License - See LICENSE file
