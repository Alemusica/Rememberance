---
description: Design iterativo tavola vibroacustica con SIMP e Deep Learning
agent: Plate Designer
---

# 🎨 Design Iterativo Tavola Vibroacustica

## Parametri Design (da personalizzare)

- **Dimensioni**: ${input:dimensions:2.0m x 0.8m x 18mm}
- **Materiale**: ${input:material:plywood_birch|aluminum|mdf|bamboo}
- **Zone Target**: ${input:zones:pelvis,heart,head}
- **Frequenze**: ${input:frequencies:auto|custom}
- **Volume Fraction**: ${input:volfrac:0.85}
- **Max Iterazioni**: ${input:max_iter:100}

## Obiettivo

Esegui ottimizzazione topologica iterativa per creare un design di tavola vibroacustica ottimizzato per le zone corporee specificate.

## Workflow Richiesto

### 1. Setup Problema
- Definisci mesh FEM (almeno 100x40 elementi)
- Configura materiale: E, ρ, ν
- Mappa zone corporee → posizioni sulla tavola

### 2. Analisi Modale Iniziale
- Calcola frequenze naturali configurazione iniziale
- Identifica modi che matchano target
- Visualizza mode shapes

### 3. Ottimizzazione SIMP
```python
# Parametri suggeriti
simp_params = {
    'penal': 3.0,          # Penalizzazione
    'rmin': 0.015,         # Filter radius (1.5% dimensione)
    'volfrac': ${volfrac}, # Volume target
    'max_iter': ${max_iter}
}
```

### 4. Iterazioni con Feedback
Per ogni iterazione:
1. Calcola sensitivities (∂f/∂x)
2. Applica density filter (avoid checkerboard)
3. Update densità con Optimality Criteria
4. Verifica convergenza e coupling score

### 5. Post-Processing
- Threshold densità (0/1 per manufacturing)
- Applica vincoli CNC (min feature size)
- Export DXF/STL

### 6. Visualizzazione Finale
- Heatmap densità con human overlay
- Mode shapes dei primi 10 modi
- Grafico coupling score vs iterazione

## File da Generare/Modificare

1. `src/core/plate_optimizer.py` - Core SIMP optimizer
2. `src/core/plate_physics.py` - FEM solver
3. `src/ui/plate_designer_tab.py` - GUI integration
4. `presets/optimized_plate.json` - Save design

## Criteri di Successo

- [ ] Coupling score > 0.8 per tutte le zone target
- [ ] Convergenza < 100 iterazioni
- [ ] Design manufacturabile (no isole, bordi solidi)
- [ ] Frequenze entro ±10% dal target

## Note

- Usa `evolutionary_optimizer.py` se SIMP converge a minimo locale
- Considera accelerazione con neural network dopo 20 iter
- Verifica sempre vincoli manufacturing prima di export
