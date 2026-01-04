# AGENTS.md — Knowledge Base System-Wide

## 🧠 MEMORIA AGENTE CONDIVISA

Questo database contiene la **memoria persistente** condivisa tra tutti gli agenti AI.
Include paper scientifici, algoritmi validati e **cronologia delle conversazioni**.

## Location Database

```
~/.config/surrealdb/knowledge.db/
```

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
  -H "Content-Type: text/plain" \
  -d "YOUR_QUERY_HERE"
```

## Contenuto Database

| Tabella | Records | Descrizione |
|---------|---------|-------------|
| **paper** | 86+ | Paper scientifici con abstract |
| **algorithm** | 6+ | Algoritmi validati |
| **concept** | vari | Concetti distillati |
| **chat_session** | 91+ | Sessioni Copilot chat |
| **chat_message** | 2340+ | Messaggi Q&A completi |
| **chat_agent** | 5 | Agenti usati (agent, vscode, etc) |
| **chat_model** | 13 | Modelli LLM usati |

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
