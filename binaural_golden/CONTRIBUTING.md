# Contributing to Golden Studio

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## 🎯 Areas of Interest

We're particularly interested in contributions in these areas:

### High Priority
- **Ray parallelization** for distributed fitness evaluation
- **Quality-Diversity algorithms** (MAP-Elites) for zone-specific exploration
- **Performance optimization** of FEM analysis

### Medium Priority
- Additional domain adapters (singing bowls, speaker enclosures, vibrating strings)
- Web-based UI (React/Vue + FastAPI)
- Better visualization of Pareto fronts

### Documentation
- Tutorials and examples
- Research paper summaries
- API documentation

## 🔧 Development Setup

```bash
# Clone and setup
git clone https://github.com/Alemusica/Golden-Studio.git
cd Golden-Studio/binaural_golden

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (including dev)
pip install -r requirements.txt
pip install pytest pytest-cov black ruff

# Run tests to verify setup
pytest tests/ -v
```

## 📝 Code Style

We use:
- **Black** for formatting (line length 100)
- **Ruff** for linting
- **Type hints** for all public functions
- **Docstrings** in Google style

```bash
# Format code
black src/ tests/ --line-length 100

# Lint code
ruff check src/ tests/
```

## 🧪 Testing

Please add tests for new features:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/core --cov-report=html

# Run specific test file
pytest tests/test_physics_validation.py -v
```

## 📊 Architecture Guidelines

### Adding a New Optimizer Strategy

1. Create adapter in `src/core/plate_unified.py`:

```python
class MyNewStrategy(OptimizationStrategy):
    def optimize(self, config: OptimizationConfig) -> OptimizationResult:
        # Implementation
        pass
```

2. Register in strategy factory:

```python
STRATEGY_REGISTRY["my_new"] = MyNewStrategy
```

### Adding a New Scorer

1. Create in `src/core/scorers/`:

```python
from .protocol import ScorerBase, ScorerResult

class MyNewScorer(ScorerBase):
    name = "my_new_scorer"
    
    def score(self, genome: PlateGenome, physics_result: dict) -> ScorerResult:
        # Calculate score
        return ScorerResult(
            score=calculated_score,
            details={"key": "value"},
            confidence=0.9
        )
```

2. Register in `src/core/scorers/__init__.py`

### Adding a New Domain Adapter

See `framework/examples/singing_bowl_adapter.py` for a template.

## 🔀 Pull Request Process

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/amazing-feature`
3. Make your changes with **tests**
4. Run tests: `pytest tests/ -v`
5. Format: `black src/ tests/`
6. Commit: `git commit -m 'feat: Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a **Pull Request**

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvement

## 📚 Research References

When adding physics-based features, please reference relevant papers from our bibliography:
- `docs/research/vibroacoustic_references.bib`

Key papers to cite:
- **NSGA-II**: Deb et al. 2002
- **Exciter placement**: Bai & Liu 2004
- **ABH theory**: Krylov 2014
- **Body resonance**: Griffin 1990

## ❓ Questions?

- Open an **Issue** for bugs or feature requests
- Start a **Discussion** for questions or ideas

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Golden Studio better! 🙏
