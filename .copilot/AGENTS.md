# AGENTS.md — Knowledge Base Redirect

## ⚠️ DATABASE SPOSTATO

Il database SurrealDB con la knowledge base di ricerca è stato spostato in una location **system-wide** per renderlo accessibile da tutti i progetti.

## Nuova Location

```
~/.config/surrealdb/knowledge.db/
```

## Accesso al Database

### Avvio Server
```bash
surreal start --log info --user root --pass root --bind 0.0.0.0:8000 rocksdb:~/.config/surrealdb/knowledge.db
```

### Query Dirette
```bash
surreal sql --endpoint ws://localhost:8000 --username root --password root --namespace research --database knowledge
```

### MCP Server (per VS Code Copilot)
Il file `~/Library/Application Support/Code/User/mcp.json` è configurato per:
- **Server**: surrealdb-knowledge
- **Namespace**: research  
- **Database**: knowledge
- **Auth**: root:root

## Contenuto Database

| Tabella | Records | Descrizione |
|---------|---------|-------------|
| paper | 86+ | Paper scientifici con abstract, relevance |
| algorithm | 6+ | Algoritmi validati con success_metrics |
| concept | vari | Concetti distillati dalla ricerca |

## Query Utili

```surql
-- Cerca paper per keyword
SELECT * FROM paper WHERE title CONTAINS 'keyword' OR abstract CONTAINS 'keyword';

-- Algoritmi con alto success rate
SELECT * FROM algorithm WHERE success_rate > 0.8;

-- Paper collegati ad algoritmo
SELECT * FROM algorithm WHERE paper_sources CONTAINS 'cite_key';
```

## Note

- Per importare nuovi paper: usa gli script in `~/Rememberance/scripts/`
- Backup: `~/.config/surrealdb/knowledge.db.backup/`
- Last migration: $(date +%Y-%m-%d)

