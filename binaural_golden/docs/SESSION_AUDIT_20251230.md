# 🔧 SESSION AUDIT - Golden Studio Plate Designer
**Data**: 30 Dicembre 2025
**Scopo**: Documentazione completa per continuità delle sessioni

---

## 📋 STATO ATTUALE DEL PROGETTO

### ✅ Bug Risolto Oggi
**File**: `src/ui/components/evolution_canvas.py` (linea 653)

**Problema**: Il visualizzatore crashava durante l'evoluzione con errore:
```
ValueError: The truth value of an array with more than one element is ambiguous
```

**Causa**: Il codice `if not control_points or len(control_points) < 4:` falliva quando `control_points` era un numpy array (impossibile valutare truthiness di array).

**Fix Applicato**:
```python
# PRIMA (BUGGY):
if not control_points or len(control_points) < 4:

# DOPO (FIXED):
if control_points is None:
    return self._generate_organic_points(...)
try:
    if len(control_points) < 4:
        return self._generate_organic_points(...)
except TypeError:
    return self._generate_organic_points(...)
```

---

## 🏗️ ARCHITETTURA PLATE DESIGNER

### Struttura MVVM
```
PlateDesignerTab (View)          → plate_designer_tab.py (947 righe)
    ↓
PlateDesignerViewModel           → viewmodels/plate_designer_viewmodel.py (680 righe)
    ↓
EvolutionaryOptimizer            → core/evolutionary_optimizer.py (583 righe)
    ↓
FitnessEvaluator                 → core/fitness.py (1227 righe)
    ↓
Components:
├── EvolutionCanvas              → components/evolution_canvas.py (2377 righe)
├── GoldenProgressBar
├── FitnessRadarChart
└── FitnessLineChart
```

### Flusso Dati
1. **UI** (PlateDesignerTab) raccoglie input utente:
   - Person: height_m, weight_kg, preset
   - Evolution: population_size (30), generations (50), mutation_rate (0.3)
   - Contour: RECTANGLE, GOLDEN_RECT, ELLIPSE, OVOID, SUPERELLIPSE, ORGANIC, ERGONOMIC, FREEFORM, AUTO
   - Zone weights: spine (70%) vs head (30%)

2. **ViewModel** gestisce stato e threading:
   - `start_evolution()` avvia thread background
   - `_run_evolution()` chiama optimizer con callback
   - `poll_updates()` notifica observers (50ms polling)

3. **Optimizer** (genetic algorithm):
   - Popolazione iniziale casuale
   - Selezione torneo (size=3)
   - Crossover (80%), Mutazione (30%)
   - Elite (2 individui preservati)
   - Convergence check dopo 40 generazioni

4. **Fitness** (multi-obiettivo):
   - `flatness_score`: risposta frequenza piatta (20-200 Hz)
   - `spine_coupling_score`: accoppiamento vibroacustico spina
   - `low_mass_score`: peso minimo
   - `manufacturability_score`: producibilità CNC
   - `structural_score`: deflection < 10mm
   - `ear_uniformity_score`: L/R balance (nuovo!)

---

## 📊 METRICHE FITNESS

### Pesi Obiettivi (ObjectiveWeights)
```python
flatness: 1.0          # Risposta frequenza
spine_coupling: 2.0    # Accoppiamento spina (priorità)
low_mass: 0.3          # Peso tavola
manufacturability: 0.5 # Facilità produzione
```

### Pesi Zone (ZoneWeights)
```python
spine: 0.70  # 70% priorità feeling tattile
head: 0.30   # 30% priorità audio binaurale
```

### Score Strutturali
- `max_deflection_mm`: limite 10mm sotto peso persona
- `stress_safety_factor`: deve essere > 2.0
- `peninsula_net_score`: ABH benefit - structural penalty (può essere positivo!)

---

## 🔬 CONTOUR TYPES

| Tipo | Descrizione | CNC |
|------|-------------|-----|
| RECTANGLE | Rettangolo fisso | ✅ |
| GOLDEN_RECT | Rapporto φ (1:1.618) | ✅ |
| ELLIPSE | Ellisse liscia | ✅ |
| OVOID | Forma uovo | ✅ |
| VESICA_PISCIS | Geometria sacra | ✅ |
| SUPERELLIPSE | Squircle (corners arrotondati) | ✅ |
| ORGANIC | Fourier-based (come chitarra) | ✅ |
| ERGONOMIC | Conformato al corpo | ✅ |
| FREEFORM | Spline evolvibile | ⚠️ |

---

## 🛠️ FILES PRINCIPALI

### Core Physics
- `core/fitness.py` - FitnessEvaluator, FitnessResult, scoring
- `core/evolutionary_optimizer.py` - EvolutionaryOptimizer, EvolutionConfig
- `core/plate_genome.py` - PlateGenome, ContourType, Cutout
- `core/plate_physics.py` - Modal analysis, MATERIALS
- `core/structural_analysis.py` - DeflectionResult, peninsula detection
- `core/pymoo_optimizer.py` - NSGA-II multi-objective (NUOVO)

### UI Components
- `ui/plate_designer_tab.py` - Tab principale
- `ui/viewmodels/plate_designer_viewmodel.py` - Business logic
- `ui/components/evolution_canvas.py` - Visualizzazione (BUGGATO poi FIXATO)

### Export
- `core/dsp_export.py` - Export per DSP agent
- `core/stl_export.py` - Export STL/OBJ/DXF per CNC

---

## ⚠️ PROBLEMI NOTI

### 1. Audio Stuttering (Harmonic Tree)
- **Status**: NON RISOLTO
- **Sintomo**: Audio frammentato nel tab Harmonic Tree
- **Tentato**: Buffer 1024→2048 (non ha funzionato)
- **Ipotesi**: Problema con DDJ-FLX4 sample rate (48kHz vs 44.1kHz)
- **File**: `core/audio_engine.py`

### 2. pymoo Non Installato
- **Status**: Avviso ma non bloccante
- **Comando**: `pip install pymoo>=0.6.0`
- **File**: `core/pymoo_optimizer.py` (nuovo, opzionale)

### 3. JAX Non Installato
- **Status**: Avviso ma non bloccante
- **Comando**: `pip install jax jaxlib`
- **File**: `core/jax_plate_fem.py` (accelerazione GPU)

---

## 📈 TEST STATUS

```bash
cd binaural_golden && python -m pytest tests/test_evolution_validation.py -v
```

**Risultato atteso**: 12 passed, 1 skipped

### Test Chiave
- `test_person_on_plate` - Person posizionata correttamente
- `test_fitness_evaluation` - FitnessEvaluator funziona
- `test_peninsula_abh_benefit` - ABH benefit peninsula (+0.27)
- `test_ear_uniformity_calculation` - L/R balance (target > 0.6)
- `test_optimized_beats_baseline` - Evoluzione migliora fitness

---

## 🚀 PROSSIMI PASSI

1. **Risolvere Audio Stuttering** (priorità alta)
   - Provare MacBook Pro Speakers
   - Controllare sample rate matching
   - Possibile problema threading tra UI e audio callback

2. **Installare pymoo** per ottimizzazione Pareto
   ```bash
   pip install pymoo>=0.6.0
   ```

3. **Migliorare ear_uniformity**
   - Target attuale: 0.75
   - Target desiderato: > 0.90
   - Potrebbe richiedere constraint esplicito nel genoma

4. **Test E2E Plate Designer**
   - Avviare evoluzione completa (50 gen)
   - Verificare export DSP
   - Verificare export STL/CNC

---

## 📂 STRUTTURA REPOSITORY

```
binaural_golden/
├── src/
│   ├── core/                 # Physics & algorithms
│   │   ├── fitness.py       # Multi-objective scoring
│   │   ├── evolutionary_optimizer.py
│   │   ├── plate_genome.py  # DNA tavola
│   │   ├── plate_physics.py # Modal analysis
│   │   └── pymoo_optimizer.py (NUOVO)
│   ├── ui/
│   │   ├── plate_designer_tab.py
│   │   ├── viewmodels/
│   │   │   └── plate_designer_viewmodel.py
│   │   └── components/
│   │       └── evolution_canvas.py (FIXATO)
│   └── golden_studio.py     # Entry point
├── tests/
│   └── test_evolution_validation.py
└── docs/
    ├── DISTILLED_RESEARCH.md
    └── SESSION_AUDIT_20251230.md (QUESTO FILE)
```

---

## 🔑 COMANDI UTILI

```bash
# Avvia app
cd binaural_golden/src && python golden_studio.py

# Run tests
cd binaural_golden && python -m pytest tests/ -v

# Install dependencies
pip install pymoo>=0.6.0 scipy>=1.10.0

# Check for issues
python -c "from core.fitness import FitnessEvaluator; print('OK')"
```

---

## 📚 RIFERIMENTI RICERCA

Vedi `docs/research/vibroacoustic_references.bib` (72 papers):
- ABH (Acoustic Black Holes): Krylov 2014, Deng 2019
- Multi-exciter: Lu 2012, Shen 2016, Bai & Liu 2004
- Modal coupling: Sum & Pan 2000
- DML plates: Borwick 2001, Harris 2001

---

**NOTA PER PROSSIMA SESSIONE**: 
Il bug del visualizzatore è stato fixato. Il problema principale ora è l'audio stuttering nel tab Harmonic Tree che non era presente prima. Controllare git diff se ci sono state modifiche recenti all'audio engine.
