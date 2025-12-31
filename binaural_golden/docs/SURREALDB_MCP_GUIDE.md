# 🗄️ SurrealDB + MCP + VS Code: Guida Completa

## Panoramica Architettura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SurrealDB Server (System-Wide)                         │
│                         localhost:8000                                       │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Namespace: "research"                                                  │  │
│  │   └── Database: "knowledge"                                           │  │
│  │         ├── paper (72 papers importati)                               │  │
│  │         ├── concept (future: concetti estratti)                       │  │
│  │         └── algorithm (future: algoritmi)                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Namespace: "projects" (future)                                         │  │
│  │   └── Database: "golden_studio"                                       │  │
│  │         ├── experiments                                                │  │
│  │         └── results                                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   VS Code    │  │    Claude    │  │   Scripts    │
    │   + Copilot  │  │   Desktop    │  │   Python     │
    │   + MCP      │  │              │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## 1. SurrealDB: Dove Vive?

### Installazione (macOS)
```bash
# Via Homebrew (consigliato)
brew install surrealdb/tap/surreal

# Oppure via script
curl -sSf https://install.surrealdb.com | sh
```

### Storage Persistente
SurrealDB può funzionare in due modi:

1. **In-Memory** (dati persi al riavvio):
   ```bash
   surreal start --bind 0.0.0.0:8000 --user root --pass root
   ```

2. **Persistente su File** (consigliato):
   ```bash
   # Crea directory
   mkdir -p ~/.surrealdb/data
   
   # Avvia con storage file
   surreal start --bind 0.0.0.0:8000 \
     --user root --pass root \
     file:~/.surrealdb/data/knowledge.db
   ```

### Avvio Automatico (LaunchAgent macOS)
```bash
# Crea il file di configurazione
cat > ~/Library/LaunchAgents/com.surrealdb.server.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.surrealdb.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/surreal</string>
        <string>start</string>
        <string>--bind</string>
        <string>127.0.0.1:8000</string>
        <string>--user</string>
        <string>root</string>
        <string>--pass</string>
        <string>root</string>
        <string>file:/Users/YOUR_USERNAME/.surrealdb/data/knowledge.db</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/surrealdb.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/surrealdb.error.log</string>
</dict>
</plist>
EOF

# Carica il servizio
launchctl load ~/Library/LaunchAgents/com.surrealdb.server.plist

# Verifica
launchctl list | grep surrealdb
```

## 2. MCP (Model Context Protocol): Come Funziona

### Cos'è MCP?
MCP è un protocollo standard (creato da Anthropic) che permette agli AI agents di:
- Accedere a **risorse** (file, database, API)
- Eseguire **tool** (funzioni)
- Usare **prompt** predefiniti

### Tipi di Server MCP

| Tipo | Trasporto | Uso |
|------|-----------|-----|
| **STDIO** | stdin/stdout | Locale, lanciato da VS Code |
| **SSE** | HTTP Server-Sent Events | Remoto, sempre attivo |
| **Streamable HTTP** | HTTP bidirectional | Remoto, nuovo standard |

### Server MCP Disponibili

**Built-in (con GitHub Copilot Pro Plus):**
- GitHub MCP Server (repos, issues, PRs)
- Container tools (Docker/Podman)
- Pylance Python tools

**Registri MCP:**
- https://github.com/modelcontextprotocol/servers (ufficiali)
- https://registry.modelcontextprotocol.io (registry)
- https://mcp.so (community)

## 3. Configurazione VS Code

### File: `.vscode/mcp.json`
Questo file configura i server MCP **locali** per il workspace:

```json
{
  "servers": {
    "surrealdb-knowledge": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.utils.surrealdb_mcp_server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

### File: `.vscode/settings.json`
Impostazioni ottimali per Copilot Pro Plus:

```json
{
  // MCP
  "chat.mcp.access": "all",
  "chat.mcp.discovery.enabled": {},
  "chat.mcp.autoStart": "newAndOutdated",
  
  // Agent Mode
  "chat.agent.enabled": true,
  "chat.agent.maxRequests": 50,
  
  // Custom Instructions
  "github.copilot.chat.codeGeneration.useInstructionFiles": true,
  "chat.useAgentsMdFile": true
}
```

### File: `.github/copilot-instructions.md`
Già presente nel progetto - contiene le istruzioni personalizzate per Copilot.

## 4. Tool MCP Disponibili per Golden Studio

Il server `surrealdb-knowledge` espone questi tool:

| Tool | Descrizione | Esempio |
|------|-------------|---------|
| `search_papers` | Cerca paper per keyword | "genetic algorithm", "ABH" |
| `get_papers_by_section` | Paper per sezione | "MULTI_EXCITER", "LUTHERIE" |
| `get_paper_details` | Dettagli completi paper | `cite_key="bai2004genetic"` |
| `get_key_papers` | Paper fondamentali | - |
| `get_knowledge_stats` | Statistiche KB | - |

### Uso in Chat
```
@workspace Cerca i paper su "acoustic black holes" nella knowledge base

> Tool: search_papers(query="acoustic black holes")
> Trovati 7 paper...
```

## 5. GitHub Pro + Copilot Pro Plus: Features

### Cosa hai incluso:

| Feature | Descrizione |
|---------|-------------|
| **Agent Mode** | AI può eseguire multi-step tasks autonomamente |
| **MCP Servers** | Nessuna restrizione policy |
| **GitHub MCP** | Access completo a repos/issues/PRs |
| **Custom Models** | Possibilità di aggiungere modelli OpenAI-compatible |
| **Unlimited chat** | Nessun limite mensile |

### Agent Mode Best Practices

1. **Usa checkpoint**: Abilita `chat.checkpoints.enabled` per rollback
2. **Auto-approve selettivo**: Approva solo comandi sicuri
3. **Instructions file**: Mantieni `.github/copilot-instructions.md` aggiornato
4. **MCP locale**: Usa server MCP per accesso strutturato ai dati

## 6. Workflow Consigliato

```
┌─────────────────────────────────────────────────────────────────┐
│                     TUO WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. AVVIA SurrealDB (se non automatico)                         │
│     $ surreal start file:~/.surrealdb/data/knowledge.db         │
│                                                                  │
│  2. APRI VS Code nel workspace Golden Studio                    │
│     $ code /Users/alessioivoycazzaniga/Rememberance             │
│                                                                  │
│  3. I Server MCP si avviano automaticamente                     │
│     - surrealdb-knowledge (locale)                              │
│     - GitHub MCP (remoto, built-in)                             │
│                                                                  │
│  4. USA Agent Mode per task complessi                           │
│     > "Analizza i paper su multi-exciter optimization           │
│        e suggerisci miglioramenti al fitness evaluator"         │
│                                                                  │
│  5. QUERY Knowledge Base                                         │
│     > "Quali paper parlano di NSGA-II?"                         │
│     > Tool call: search_papers("NSGA-II")                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 7. Troubleshooting

### SurrealDB non risponde
```bash
# Verifica se è attivo
curl http://localhost:8000/health

# Controlla log
tail -f /tmp/surrealdb.log

# Riavvia
surreal start file:~/.surrealdb/data/knowledge.db
```

### MCP Server non appare
1. Ricarica VS Code window (`Cmd+Shift+P` → "Developer: Reload Window")
2. Controlla `Output` → `MCP` per errori
3. Verifica il path in `mcp.json`

### Tool non funziona
```bash
# Test manuale del server
cd /Users/alessioivoycazzaniga/Rememberance/binaural_golden
python -m src.utils.surrealdb_mcp_server
```

## 8. Estensioni Raccomandate

```json
// .vscode/extensions.json
{
  "recommendations": [
    "GitHub.copilot",
    "GitHub.copilot-chat",
    "ms-python.python",
    "ms-python.pylance",
    "surrealdb.surrealdb"
  ]
}
```

## 9. Prossimi Passi

1. **Arricchisci la Knowledge Base**:
   - Aggiungi concetti estratti dai paper
   - Crea relazioni tra paper (citazioni)
   
2. **Espandi MCP Server**:
   - Tool per salvare risultati esperimenti
   - Tool per query algoritmi ottimizzazione

3. **Remote MCP** (opzionale):
   - Deploya su server per accesso multi-dispositivo
   - Usa autenticazione OAuth2

---

## Riferimenti

- [MCP Protocol Spec](https://modelcontextprotocol.io/)
- [SurrealDB Docs](https://surrealdb.com/docs)
- [VS Code Copilot Settings](https://code.visualstudio.com/docs/copilot/copilot-settings)
- [GitHub MCP Server](https://github.com/github/github-mcp-server)
