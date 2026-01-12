# Plate Designer - Architettura Sistema

## Obiettivo
Progettare una tavola vibroacustica ottimale per una persona specifica, visualizzando l'evoluzione della forma iterazione dopo iterazione.

## Input Utente
```
┌─────────────────────────────────────────┐
│ PARAMETRI PERSONA                       │
│   • Altezza: 1.50 - 2.10 m             │
│   • Peso: 45 - 120 kg                   │
├─────────────────────────────────────────┤
│ PARAMETRI TAVOLA (opzionali)            │
│   • Materiale: spruce, birch, etc.      │
│   • Spessore iniziale: auto / manuale   │
├─────────────────────────────────────────┤
│ OBIETTIVI (pesi configurabili)          │
│   • Risposta frequenza piatta           │
│   • Coupling spina dorsale              │
│   • Peso minimo tavola                  │
└─────────────────────────────────────────┘
```

## Output
- Forma contorno ottimizzata (vertici poligono)
- Mappa spessore variabile (se abilitato)
- Tagli interni (se abilitati)
- Grafico risposta frequenza con persona
- Animazione evoluzione forma

---

## Architettura Moduli

### 1. `core/person.py` (NUOVO)
```python
@dataclass
class Person:
    height_m: float
    weight_kg: float
    
    # Calcolati automaticamente
    @property
    def spine_length(self) -> float
    @property  
    def mass_distribution(self) -> Dict[str, float]
    @property
    def contact_area(self) -> float
```
**Responsabilità**: Modello antropometrico persona distesa.

### 2. `core/plate_genome.py` (NUOVO)
```python
@dataclass
class PlateGenome:
    """Rappresentazione genetica della tavola."""
    # Forma
    contour_points: np.ndarray  # (N, 2) vertici normalizzati
    
    # Spessore (opzionale)
    thickness_map: Optional[np.ndarray]  # (nx, ny) grid
    
    # Tagli interni (opzionale)
    cutouts: List[CutoutGene]
    
    def mutate(self, sigma: float) -> 'PlateGenome'
    def crossover(self, other: 'PlateGenome') -> 'PlateGenome'
    def to_mesh(self) -> Tuple[points, triangles]
```
**Responsabilità**: Codifica genetica della tavola per ottimizzazione evolutiva.

### 3. `core/fitness.py` (NUOVO)
```python
class FitnessEvaluator:
    """Valuta fitness di un PlateGenome."""
    
    def __init__(self, person: Person, objectives: ObjectiveWeights):
        ...
    
    def evaluate(self, genome: PlateGenome) -> FitnessResult:
        # 1. Genera mesh da genome
        # 2. Calcola modi con FEM (mass-loaded)
        # 3. Calcola risposta frequenza
        # 4. Calcola coupling spine
        # 5. Combina in fitness score
        ...
```
**Responsabilità**: Funzione obiettivo multi-criterio.

### 4. `core/evolutionary_optimizer.py` (NUOVO)
```python
class EvolutionaryOptimizer:
    """Ottimizzazione genetica forma tavola."""
    
    def __init__(
        self,
        person: Person,
        config: EvolutionConfig,
        fitness_evaluator: FitnessEvaluator
    ):
        ...
    
    def run(
        self,
        callback: Optional[Callable[[int, PlateGenome, float], None]] = None
    ) -> OptimizationResult:
        """
        Esegue ottimizzazione.
        
        callback viene chiamato ogni iterazione con:
        - iteration: numero iterazione
        - best_genome: miglior genoma corrente  
        - fitness: fitness corrente
        
        Per visualizzazione real-time nella GUI.
        """
        ...
```
**Responsabilità**: Algoritmo genetico / CMA-ES / Differential Evolution.

### 5. `core/mass_loaded_fem.py` (NUOVO)
```python
def modal_analysis_with_body(
    mesh: Tuple[points, triangles],
    material: Material,
    person: Person,
    n_modes: int = 15
) -> List[Mode]:
    """
    FEM modale con massa persona distribuita.
    
    La massa della persona viene aggiunta come massa localizzata
    sui nodi della mesh in base alla distribuzione corporea.
    """
    ...

def frequency_response_with_body(
    modes: List[Mode],
    person: Person,
    exciter_position: Tuple[float, float],
    freq_range: Tuple[float, float] = (20, 200)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Risposta in frequenza con persona distesa.
    
    Returns:
        frequencies: array Hz
        response: array dB (risposta media su spine)
    """
    ...
```
**Responsabilità**: FEM con massa aggiunta della persona.

---

## Moduli Esistenti da Usare

| Modulo | Cosa Fornisce | Da Modificare |
|--------|---------------|---------------|
| `coupled_system.py` | `PlatePhysics`, `HumanBody`, `CoupledSystem` | Integrare con `Person` |
| `plate_physics.py` | `Material`, `MATERIALS` | OK com'è |
| `body_zones.py` | `BodyZone`, `BodyResonance` | Usare per spine zones |
| `plate_fem.py` | FEM base | Estendere per mass-loading |
| `iterative_optimizer.py` | SIMP/RAMP, density filter | Mantenere per density opt |

---

## UI Components

### `ui/plate_designer/` (NUOVA directory)

```
plate_designer/
├── __init__.py
├── tab.py              # PlateDesignerTab main class (~200 righe)
├── person_panel.py     # Input altezza/peso (~100 righe)
├── objectives_panel.py # Pesi obiettivi (~100 righe)
├── evolution_view.py   # Canvas evoluzione forma (~200 righe)
├── results_panel.py    # Grafici finali (~150 righe)
└── animation.py        # Animazione evoluzione (~100 righe)
```

---

## Flusso Dati

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ PersonPanel  │────▶│    Person    │────▶│ FitnessEvaluator │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
┌──────────────┐     ┌──────────────┐              │
│ ObjectPanel  │────▶│  Objectives  │──────────────┤
└──────────────┘     └──────────────┘              │
                                                   ▼
                     ┌──────────────────────────────────────┐
                     │      EvolutionaryOptimizer           │
                     │                                      │
                     │  iteration 1: ●────────────────      │
                     │  iteration 2:   ●──────────────      │
                     │  iteration 3:     ●────────────      │
                     │  ...                                 │
                     │  iteration N:           ●── BEST     │
                     └──────────────┬───────────────────────┘
                                    │
                                    ▼ callback ogni iterazione
                     ┌──────────────────────────────────────┐
                     │         EvolutionView Canvas         │
                     │  ┌─────────────────────────────────┐ │
                     │  │     FORMA EVOLVE LIVE           │ │
                     │  │                                 │ │
                     │  │    ╭───────────────────╮       │ │
                     │  │    │   iter 47/100     │       │ │
                     │  │    │   fitness: 0.87   │       │ │
                     │  │    ╰───────────────────╯       │ │
                     │  └─────────────────────────────────┘ │
                     └──────────────────────────────────────┘
```

---

## Piano Implementazione

### Fase 1: Core Models (senza UI)
1. `core/person.py` - modello persona
2. `core/plate_genome.py` - codifica genetica
3. `core/fitness.py` - funzione obiettivo
4. Test unitari

### Fase 2: Optimizer
1. `core/evolutionary_optimizer.py`
2. `core/mass_loaded_fem.py`
3. Test con callback dummy

### Fase 3: UI
1. `ui/plate_designer/tab.py`
2. `ui/plate_designer/person_panel.py`
3. `ui/plate_designer/evolution_view.py`
4. Integrazione in golden_studio.py

### Fase 4: Polish
1. Animazione fluida
2. Export risultati (DXF, SVG)
3. Preset persona (bambino, adulto, anziano)

---

## Note Tecniche

### FEM con Massa Persona
```python
# Matrice massa con contributo persona
M_total = M_plate + M_person

# M_person è diagonale con masse localizzate:
# - Head: 8% massa @ posizione testa
# - Torso: 50% massa @ distribuito torso  
# - Legs: 32% massa @ posizione gambe
```

### Fitness Multi-Obiettivo
```python
fitness = (
    w1 * flatness_score +      # Risposta piatta
    w2 * spine_coupling +      # Accoppiamento spina
    w3 * (1 - mass_penalty) +  # Peso minimo
    w4 * manufacturability     # Producibilità
)
```

### Rappresentazione Genetica Forma
```python
# Contorno come spline con N punti di controllo
# Mutazione = rumore gaussiano sui punti
# Crossover = blend tra punti corrispondenti
```
