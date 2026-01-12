# 🎯 Architettura Sistema Ottimizzazione Tavola Vibroacustica

## Visione d'Insieme

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GOLDEN BINAURAL PLATFORM                                 │
│                  Sistema Tavola Vibroacustica Ottimizzata                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  GUI/API    │───▶│  Optimizer  │───▶│  FEM Core   │───▶│  Physics    │  │
│  │  Layer      │    │  Engine     │    │  (JAX/SKFEM)│    │  Output     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Body Zones  │    │  SIMP/RAMP  │    │  Eigenvalue │    │  Density    │  │
│  │ Model       │    │  Interp.    │    │  Solver     │    │  Field      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Struttura Moduli

### Core Modules (`src/core/`)

| Modulo | LOC | Descrizione | Status |
|--------|-----|-------------|--------|
| `body_zones.py` | ~450 | Modello zone corporee con 3 preset | ✅ Completo |
| `coupled_system.py` | ~400 | Sistema 2-DOF tavola+corpo | ✅ Completo |
| `iterative_optimizer.py` | ~950 | SIMP/RAMP + OC optimizer | ✅ Completo |
| `jax_plate_fem.py` | ~500 | FEM differenziabile JAX | ✅ Completo |
| `plate_fem.py` | ~580 | FEM modal analysis (scikit-fem) | ✅ Completo |
| `plate_optimizer.py` | ~400 | API high-level | ✅ Completo |

## 🔬 Fisica del Sistema

### 1. Equazione Piastra (Kirchhoff-Love)

$$D \nabla^4 w = \rho h \frac{\partial^2 w}{\partial t^2}$$

dove:
- $D = \frac{Eh^3}{12(1-\nu^2)}$ rigidità flessionale
- $w$ = spostamento trasversale
- $\rho$ = densità
- $h$ = spessore

### 2. Interpolazione Materiale

**SIMP (Solid Isotropic Material with Penalization):**
$$E(\rho) = \varepsilon + (1-\varepsilon)\rho^p, \quad p=3$$

**RAMP (Rational Approximation of Material Properties):**
$$E(\rho) = \varepsilon + \frac{(1-\varepsilon)\rho}{1 + p(1-\rho)}$$

### 3. Sensitività Autovalori

$$\frac{\partial \lambda_i}{\partial \rho_e} = \phi_i^T \left(\frac{\partial K}{\partial \rho_e} - \lambda_i \frac{\partial M}{\partial \rho_e}\right) \phi_i$$

### 4. Funzione Obiettivo

$$J(\rho) = \sum_{i=1}^{N_{zones}} w_i \left(\frac{f_i(\rho) - f_i^{target}}{f_i^{target}}\right)^2$$

## 🎵 Zone Corporee

### Preset Chakra (7 zone)

| Chakra | Nome | Frequenza Centro | Range | Posizione |
|--------|------|------------------|-------|-----------|
| 1 | Root (Muladhara) | 37.5 Hz | 25-50 Hz | 0.15 |
| 2 | Sacral (Svadhisthana) | 52.5 Hz | 40-65 Hz | 0.25 |
| 3 | Solar Plexus (Manipura) | 75 Hz | 60-90 Hz | 0.35 |
| 4 | Heart (Anahata) | 110 Hz | 90-130 Hz | 0.50 |
| 5 | Throat (Vishuddha) | 165 Hz | 140-190 Hz | 0.70 |
| 6 | Third Eye (Ajna) | 220 Hz | 200-240 Hz | 0.85 |
| 7 | Crown (Sahasrara) | 400 Hz | 350-450 Hz | 0.95 |

### Preset VAT Therapy (5 zone)

| Zona | Frequenza | Applicazione |
|------|-----------|--------------|
| Deep Relax | 40 Hz | Rilassamento profondo |
| Muscle Release | 60 Hz | Tensione muscolare |
| Pain Relief | 80 Hz | Sollievo dolore |
| Circulation | 120 Hz | Circolazione |
| Neuro | 180 Hz | Stimolazione nervosa |

### Preset Body Resonance (8 zone)

| Zona | Frequenza | Struttura Corporea |
|------|-----------|-------------------|
| Cranio | 20 Hz | Risonanza cranica |
| Torace | 40-60 Hz | Gabbia toracica |
| Addome | 80 Hz | Cavità addominale |
| Colonna | 100-120 Hz | Vertebre |
| Tessuti molli | 150 Hz | Muscoli/organi |
| Ossa lunghe | 200 Hz | Femori, tibie |
| Articolazioni | 300 Hz | Ginocchia, anche |
| Microvibrazione | 500 Hz | Tessuti fini |

## 🔧 Pipeline Ottimizzazione

```python
# 1. Definisci zone target
from core.body_zones import BodyZoneModel
model = BodyZoneModel(preset="chakra")

# 2. Crea sistema accoppiato
from core.coupled_system import ZoneCoupledSystem
coupled = ZoneCoupledSystem(model.zones)

# 3. Ottimizza tavola
from core.plate_optimizer import zone_optimize_plate
result = zone_optimize_plate(
    preset="chakra",
    plate_dims=(2.0, 0.6, 0.015),
    n_iterations=50
)

# 4. Estrai risultato
optimal_density = result['optimal_density']
final_freqs = result['final_frequencies']
```

## 📊 Convergenza Tipica

```
Iterazione  Objective   Volume    Max Δρ    Freq Error
─────────────────────────────────────────────────────
    1       0.4523      0.500     0.100     45.2%
    5       0.2341      0.485     0.082     23.4%
   10       0.1205      0.467     0.054     12.0%
   20       0.0523      0.443     0.028      5.2%
   30       0.0234      0.431     0.015      2.3%
   50       0.0089      0.420     0.005      0.9%
```

## 🔄 Algoritmo OC (Optimality Criteria)

```
Per ogni iterazione k:
    1. Calcola frequenze f_i con FEM
    2. Calcola sensitività ∂J/∂ρ con auto-diff
    3. Applica filtro densità (raggio R)
    4. Aggiorna ρ con schema OC:
       ρ_new = ρ * (−∂J/∂ρ / λ)^η
    5. Proietta su [ρ_min, 1] con vincolo volume
    6. Verifica convergenza: ||ρ_new - ρ|| < tol
```

## 📈 Dipendenze

### Obbligatorie
- `numpy >= 1.24.0` - Array numerici
- `scipy >= 1.10.0` - Solver sparsi, eigsh
- `scikit-fem >= 8.0.0` - FEM elementi finiti
- `jax >= 0.4.0` - Auto-differenziazione
- `jaxlib >= 0.4.0` - Backend JAX

### Opzionali
- `matplotlib >= 3.7.0` - Visualizzazione
- `meshio >= 5.3.0` - Export mesh
- `pytest >= 7.0.0` - Testing

## 🧪 Test

```bash
# Installa dipendenze
pip install -r requirements.txt

# Verifica setup
python setup_verify.py

# Esegui test
pytest tests/test_plate_optimization.py -v

# Test rapido
python -c "from core.plate_optimizer import zone_optimize_plate; print(zone_optimize_plate(preset='vat', n_iterations=3))"
```

## 🎯 Next Steps

1. **GUI Integration**: Collegare al tab Plate Lab
2. **Visualization**: Aggiungere visualizzazione 3D densità
3. **Export**: Salvare risultati in formato CAD/STL
4. **Multi-objective**: Estendere per ottimizzazione multi-obiettivo Pareto
5. **Rust Core**: Implementare FEM in Rust per performance

---

*Documentazione generata automaticamente - Golden Binaural Platform v2.0*
