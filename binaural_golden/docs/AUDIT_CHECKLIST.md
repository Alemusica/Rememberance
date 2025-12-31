# Golden Studio - Architecture Audit Checklist

## Overview

This document provides specific audit tasks for subagents analyzing the Golden Studio codebase.
Each section is designed for an independent audit task.

---

## Audit Task 1: Code Quality & Maintainability

### Files to Analyze
- `src/core/fitness.py` (1554 lines)
- `src/core/plate_genome.py` (2059 lines)
- `src/core/modal_guidance.py` (1383 lines)

### Checklist
- [ ] **Cyclomatic Complexity**: Functions > 10 complexity
- [ ] **Function Length**: Functions > 50 lines
- [ ] **Parameter Count**: Functions with > 5 parameters
- [ ] **Dead Code**: Unreachable code, unused imports
- [ ] **Magic Numbers**: Hardcoded values without constants
- [ ] **Comments**: Missing docstrings, outdated comments
- [ ] **Error Handling**: Bare except, swallowed exceptions

### Metrics to Report
```
- LOC per function (avg, max)
- Cyclomatic complexity (avg, max)
- Test coverage % (if available)
- Number of TODO/FIXME comments
```

---

## Audit Task 2: Design Patterns & Architecture

### Files to Analyze
- `src/core/unified_optimizer.py` (933 lines)
- `src/core/agnostic_evolution.py` (728 lines)
- `src/core/plate_adapters.py` (615 lines)
- `src/core/plate_unified.py` (560 lines)

### Checklist
- [ ] **Strategy Pattern**: Verify correct implementation
- [ ] **Protocol/Interface Compliance**: Check type consistency
- [ ] **Dependency Injection**: Components properly injected
- [ ] **Single Responsibility**: Each class has one purpose
- [ ] **Open/Closed Principle**: Extension without modification
- [ ] **Factory Pattern**: Genome creation consistency

### Architecture Questions
1. Is the Strategy Registry scalable for new strategies?
2. Are Physics/Fitness/Factory interfaces minimal enough?
3. Can plate-specific code be easily replaced?

---

## Audit Task 3: Physics Correctness

### Files to Analyze
- `src/core/plate_physics.py` (457 lines)
- `src/core/structural_analysis.py` (1332 lines)
- `src/core/freeform_cutout.py` (905 lines)

### Reference Papers (SurrealDB)
- `schleske2002lutherie` - Violin mode analysis
- `krylov2014abh` - Acoustic Black Holes
- `fletcher1998physics` - Musical acoustics

### Checklist
- [ ] **Modal Frequency Formula**: Verify against Leissa (1969)
- [ ] **ABH Benefit Calculation**: Compare to Krylov 2014
- [ ] **SIMP Penalization**: p=3 standard, density interpolation
- [ ] **Cutout Frequency Shift**: Schleske ~5-15% range

### Physics Queries
```sql
-- Get reference formulas
SELECT * FROM paper WHERE cite_key = 'leissa1969vibration';
SELECT * FROM algorithm WHERE domain = 'modal_analysis';
```

---

## Audit Task 4: Memory System

### Files to Analyze
- `src/core/evolution_memory.py` (1292 lines)

### Checklist
- [ ] **STM Ring Buffer**: Correct FIFO behavior
- [ ] **LTM Distillation**: Meaningful pattern extraction
- [ ] **Experience Archive**: Proper ranking/selection
- [ ] **Memory Persistence**: JSON/pickle robustness
- [ ] **Performance**: Memory consumption reasonable

### Key Questions
1. Does STM trajectory analysis provide useful guidance?
2. Is LTM distillation extracting reusable patterns?
3. What's the memory footprint after 1000 generations?

---

## Audit Task 5: Test Coverage

### Files to Analyze
- `tests/test_physics_validation.py` (11 tests)
- `tests/test_evolution_validation.py`
- `tests/test_new_modules.py`

### Checklist
- [ ] **Coverage**: Critical paths covered
- [ ] **Edge Cases**: Boundary conditions tested
- [ ] **Mocking**: External dependencies mocked
- [ ] **Assertions**: Meaningful assertions, not just "doesn't crash"
- [ ] **Physics Tests**: Validate actual physics constraints

### Missing Test Areas
- Unified optimizer strategies
- Memory system persistence
- Freeform cutout optimization
- Multi-objective Pareto front

---

## Audit Task 6: Performance

### Files to Analyze
All files in `src/core/`

### Checklist
- [ ] **Numpy Vectorization**: Loops that could be vectorized
- [ ] **Caching**: Results that could be memoized
- [ ] **Redundant Computation**: Duplicate calculations
- [ ] **Memory Leaks**: Growing collections not cleared
- [ ] **JAX Usage**: GPU acceleration utilized

### Profiling Commands
```bash
python -m cProfile -o profile.prof -m pytest tests/
snakeviz profile.prof
```

---

## Audit Task 7: API Consistency

### Files to Analyze
All `__init__` methods and public APIs

### Known Inconsistencies (Fixed)
- ✅ `contour` vs `contour_type` (now `contour_type`)
- ✅ `genome.material` (now hardcoded to "birch_plywood")
- ✅ `FitnessEvaluator.evaluate(genome, physics)` (now just `genome`)
- ✅ `DensityField(nx, ny)` vs `resolution=(nx, ny)` (now tuple)
- ✅ `extract_contours(threshold)` vs `level` (now `level`)

### Checklist
- [ ] **Parameter Naming**: Consistent across modules
- [ ] **Return Types**: Documented and consistent
- [ ] **Error Messages**: Clear and actionable
- [ ] **Default Values**: Sensible defaults everywhere

---

## Audit Task 8: Deprecation Plan

### Files to Phase Out
| File | Lines | Replacement | Priority |
|------|-------|-------------|----------|
| `plate_optimizer.py` | 1165 | `unified_optimizer.py` | HIGH |
| `pymoo_optimizer.py` | 750 | `unified_optimizer.py` | HIGH |
| `evolutionary_optimizer.py` | 840 | `unified_optimizer.py` | MEDIUM |
| `iterative_optimizer.py` | 1012 | `unified_optimizer.py` | MEDIUM |

### Migration Steps
1. [ ] Add deprecation warnings to old files
2. [ ] Update all imports in UI/tests
3. [ ] Run full test suite
4. [ ] Archive old files to `_deprecated/`
5. [ ] Remove after 2 releases

---

## Running Audits

### Local (CLI)
```bash
# Run all tests
pytest tests/ -v

# Check specific physics validation
pytest tests/test_physics_validation.py -v --tb=short

# Profile optimizer
python -c "from src.core.plate_unified import create_plate_optimization_system; ..."
```

### Cloud (GitHub Copilot Coding Agent)
Create issues with specific audit tasks and assign to Copilot:
- Issue title: "Audit: Code Quality Analysis for fitness.py"
- Label: `audit`, `architecture`
- Assign: `@copilot`

---

## SurrealDB Session Storage

To save conversation history for future agents:

```sql
-- Create session table
DEFINE TABLE agent_session SCHEMAFULL;
DEFINE FIELD session_id ON agent_session TYPE string;
DEFINE FIELD timestamp ON agent_session TYPE datetime;
DEFINE FIELD project ON agent_session TYPE string;
DEFINE FIELD messages ON agent_session TYPE array;
DEFINE FIELD summary ON agent_session TYPE string;

-- Insert session
CREATE agent_session SET
  session_id = 'session_20241231_unified_optimizer',
  timestamp = time::now(),
  project = 'golden_studio',
  summary = 'Fixed modal grid resolution, created unified optimizer...',
  messages = [...];
```

Query for previous context:
```sql
SELECT * FROM agent_session 
WHERE project = 'golden_studio' 
ORDER BY timestamp DESC 
LIMIT 5;
```
