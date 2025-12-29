---
agent: DSP Engineer
tools: ['codebase', 'editFiles', 'problems']
description: Aggiungi nuova frequenza terapeutica al sistema
---

# 🎵 Add Therapeutic Frequency

Aggiungi una nuova frequenza terapeutica al sistema Golden Studio.

## Parametri
- **Nome**: ${input:name:Nome della frequenza (es. theta_wave)}
- **Frequenza Base**: ${input:freq:Frequenza in Hz (es. 7.83)}
- **Categoria**: ${input:category:Categoria (solfeggio/chakra/brainwave/custom)}
- **Descrizione**: ${input:description:Descrizione effetto terapeutico}

## Files da Modificare
1. `src/core/golden_constants.py` - Aggiungi costante
2. `src/core/golden_math.py` - Aggiungi calcoli derivati PHI
3. `modules/vibroacoustic/__init__.py` - Integra nel sistema

## Template Costante
```python
# In golden_constants.py
{NAME}_FREQ = {freq}  # Hz - {description}

# Derivate PHI
{NAME}_PHI = {NAME}_FREQ * PHI      # Armonica superiore
{NAME}_INV = {NAME}_FREQ / PHI      # Armonica inferiore
```
