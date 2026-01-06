# AGENTS.MD — Knowledge Base System-Wide

## 🧠 MEMORIA AGENTE CONDIVISA

Questo database contiene la **memoria persistente** condivisa tra tutti gli agenti AI.
Include paper scientifici, algoritmi validati e **cronologia delle conversazioni**.

**Documentazione completa**: `~/.config/surrealdb/AGENTS.md`

## Location Database

```
~/.config/surrealdb/knowledge.db/
```

## Scripts Disponibili

```
~/.config/surrealdb/scripts/
├── knowledge_ingest.py       # Multi-source: StackExchange (6 siti) + ArXiv
├── push_algorithm.py         # Registra algoritmi validati con code snippets
├── import_chats_graph_rag.py # Import chat sessions (GraphRAG)
└── import_chat_sessions.py   # Import sessioni Copilot legacy
```

### Nuove Fonti StackExchange
- `stackoverflow` - General programming DSP/audio
- `dsp` - DSP StackExchange (specializzato)
- `music` - Music theory & acoustics
- `physics` - Acoustics, waves, vibrations
- `electronics` - Audio electronics, speakers

## Avvio Server

```bash
# Manuale
surreal start --log info --user root --pass root file:~/.config/surrealdb/knowledge.db

# Automatico (LaunchAgent)
launchctl load ~/Library/LaunchAgents/com.surrealdb.knowledge.plist
```

## Query API (HTTP)

```bash
curl -X POST http://localhost:8000/sql \
  -H "Authorization: Basic cm9vdDpyb290" \
  -H "surreal-ns: research" \
  -H "surreal-db: knowledge" \
  -H "Accept: application/json" \
  -d "YOUR_QUERY_HERE"
```

## Contenuto Database (Aggiornato 2026-01-04)

| Tabella | Records | Descrizione |
|---------|---------|-------------|
| **paper** | 86 | Paper scientifici (acustica, DSP, psicoacustica) |
| **algorithm** | 6 | Algoritmi validati con success_rate, code_snippet |
| **knowledge** | 165+ | StackExchange (4 siti) + ArXiv content |
| **distilled_insight** | 4 | Conoscenza distillata persistente (cymatics, FEM, etc) |
| **chat_session** | 98 | Sessioni conversazionali complete |
| **chat_message** | 2615 | Messaggi Q&A da sessioni Copilot |
| **entity** | varies | Entità estratte (concepts, technologies) |

## 🔬 Distilled Insights (Conoscenza Persistente)

Query per accedere alla conoscenza distillata:
```sql
-- Insight per dominio
SELECT * FROM distilled_insight WHERE domain = 'vibroacoustic_therapy';

-- Insight più rilevanti
SELECT title, insight FROM distilled_insight ORDER BY relevance_score DESC;

-- Insight applicabili a un progetto
SELECT * FROM distilled_insight WHERE project_applicable CONTAINS 'golden-studio';
```

### Insight Attivi:
1. **Chladni Patterns = Body Zone Targeting** (95%) - Cimatica applicata al corpo
2. **FEM Modal Analysis per DML Plates** (92%) - Calcolo modi vibrazionali
3. **Risonanza e Standing Waves per Terapia** (90%) - Frequenze terapeutiche
4. **Knowledge Distillation per Fitness Evaluation** (88%) - Surrogate models
| **chat_session** | 98 | Sessioni conversazionali complete |
| **chat_message** | 2615 | Messaggi Q&A da sessioni Copilot |
| **entity** | varies | Entità estratte (concepts, technologies) |

## Knowledge Loop Workflow

```
1. INGEST    → knowledge_ingest.py (papers, Q&A da 6 fonti)
2. IMPLEMENT → Codifica algoritmo nel progetto
3. VALIDATE  → Esegui test, misura success_rate
4. PUSH      → push_algorithm.py (con paper_sources, code_snippet)
5. GRAPH     → Relazioni create automaticamente
```

## Architettura GraphRAG

- **NO pre-classificazione** - dati raw con metadata ricchi
- Agent inferisce contesto dal contenuto conversazionale
- **Full-text search con BM25** su user_text e assistant_text

## Query RAG per Conversazioni Passate

```surql
-- Cerca conversazioni su un argomento (RAG)
SELECT session_id, user_text, assistant_text 
FROM chat_message 
WHERE user_text CONTAINS 'keyword'
LIMIT 10;

-- Conversazioni con un modello specifico
SELECT * FROM chat_message 
WHERE model_id = 'copilot/claude-opus-4.5';

-- Sessioni recenti
SELECT session_id, title, created_at 
FROM chat_session 
ORDER BY created_at DESC 
LIMIT 20;

-- Cerca nella risposta dell'assistente
SELECT user_text, assistant_text 
FROM chat_message 
WHERE assistant_text CONTAINS 'SurrealDB';
```

## Query Paper Scientifici

```surql
-- Cerca paper
SELECT * FROM paper WHERE title CONTAINS 'acoustic';

-- Algoritmi con alto success rate
SELECT * FROM algorithm WHERE success_rate > 0.8;
```

## MCP Server Config

File: `~/Library/Application Support/Code/User/mcp.json`

```json
{
  "surrealdb-knowledge": {
    "command": "npx",
    "args": ["-y", "surrealdb-mcp-server"],
    "env": {
      "SURREALDB_URL": "ws://localhost:8000",
      "SURREALDB_NS": "research",
      "SURREALDB_DB": "knowledge"
    }
  }
}
```

## Scripts Manutenzione

- Import chat sessions: `~/.config/surrealdb/scripts/import_chats_graph_rag.py`
- Import papers: `~/Rememberance/scripts/import_papers.py`

## Last Update: 2026-01-04
