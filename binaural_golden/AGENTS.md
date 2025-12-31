# Agent Instructions - Golden Studio

## Knowledge Base Access

Questo progetto ha una **knowledge base SurrealDB** con 72 paper di ricerca su:
- Vibroacoustic therapy
- DML plate design  
- Multi-exciter optimization
- Acoustic Black Holes (ABH)
- Modal analysis

### Come Accedere

**Via MCP Tools** (se disponibili):
- `search_papers(query)` - Cerca per keyword
- `get_papers_by_section(section)` - Paper per sezione
- `get_paper_details(cite_key)` - Dettagli completi
- `get_key_papers()` - Paper fondamentali

**Via HTTP** (sempre disponibile):
```bash
curl -X POST "http://localhost:8000/sql" \
  -H "Authorization: Basic cm9vdDpyb290" \
  -H "surreal-ns: research" \
  -H "surreal-db: knowledge" \
  --data "SELECT * FROM paper WHERE title CONTAINS 'keyword'"
```

### Sezioni Disponibili

| Sezione | Papers | Contenuto |
|---------|--------|-----------|
| MULTI_EXCITER | 8 | Ottimizzazione multi-eccitatore |
| LUTHERIE | 12 | Liuteria, design strumenti |
| ACOUSTIC_BLACK | 7 | ABH, energy focusing |
| HUMAN_BODY | 8 | Risonanze corpo umano |
| VIBROACOUSTIC | 7 | Terapia vibroacustica |

### Paper Chiave

- `bai2004genetic` - NSGA-II genetic algorithm (BASE del nostro optimizer)
- `krylov2014abh` - Acoustic Black Holes theory
- `skille1989vibroacoustic` - VAT founder (30-120Hz)
- `griffin1990handbook` - Body resonance (spine 10-12Hz)

### Quando Consultare la KB

✅ Domande su fisica delle placche DML
✅ Ottimizzazione posizione eccitatori  
✅ Frequenze risonanza corpo umano
✅ Algoritmi genetici per audio
✅ Acoustic Black Holes / peninsulas
