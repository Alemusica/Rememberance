---
agent: DSP Engineer
tools: ['codebase', 'editFiles', 'runInTerminal', 'problems']
description: Refactoring modulo audio con ottimizzazione performance
---

# 🔧 Audio Module Refactoring

Analizza il modulo audio specificato e:

1. **Identifica loop Python** da convertire in NumPy vectorizzato
2. **Trova allocazioni** nel path audio critico
3. **Verifica type hints** mancanti
4. **Estrai funzioni comuni** in utils
5. **Aggiungi docstring** con parametri audio

## Target File
${input:file:File da refactorare}

## Constraints
- Mantieni compatibilità API esistente
- dtype=np.float32 per tutti gli array audio
- Latency target: < 10ms per block

## Output
- Codice refactorato
- Breve report delle modifiche
