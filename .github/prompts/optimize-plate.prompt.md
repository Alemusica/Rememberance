---
agent: Plate Expert
tools: ['codebase', 'runInTerminal', 'editFiles', 'problems']
description: Ottimizza tavola vibrante per frequenze target corpo
---

# 🎸 Optimize Plate for Body

Ottimizza la distribuzione materiale di una tavola vibrante per massimizzare l'accoppiamento con le zone corporee target.

## Parametri
- **Materiale**: ${input:material:Materiale (plywood_birch/mdf/aluminum)}
- **Dimensioni**: ${input:dimensions:Lunghezza x Larghezza in cm (es. 200x80)}
- **Spessore**: ${input:thickness:Spessore in mm (es. 18)}
- **Zone Focus**: ${input:zones:Zone corporee prioritarie (feet,pelvis,heart)}

## Target Frequencies per Zone
| Zona | Frequenza Target | Tolleranza |
|------|-----------------|------------|
| Feet | 40 Hz | ±5 Hz |
| Legs | 60 Hz | ±8 Hz |
| Pelvis | 80 Hz | ±10 Hz |
| Solar Plexus | 120 Hz | ±15 Hz |
| Heart | 128 Hz | ±10 Hz |
| Throat | 192 Hz | ±15 Hz |
| Head | 256 Hz | ±20 Hz |

## Workflow
1. Calcola frequenze modali per materiale/dimensioni
2. Identifica modi più vicini ai target
3. Esegui ottimizzazione SIMP
4. Valida risultato con analisi FEM
5. Visualizza distribuzione materiale
6. Esporta per CNC se richiesto

## Output
- Mappa densità materiale ottimizzata
- Report frequenze modali vs target
- Coupling score per zona
- File DXF (opzionale)
