# 🚀 LLM-Evolution Setup Guide

## 1. Prerequisiti

### Ollama (OBBLIGATORIO)

```bash
# macOS
brew install ollama

# Oppure installer
curl -fsSL https://ollama.com/install.sh | sh

# Avvia il servizio
ollama serve
```

### SurrealDB (OBBLIGATORIO)

```bash
# macOS
brew install surrealdb

# Oppure curl
curl -sSf https://install.surrealdb.com | sh

# Verifica
surreal version
```

### Python 3.10+

```bash
python3 --version  # >= 3.10
```

---

## 2. Modelli LLM

### Modelli Necessari

| Modello | Ruolo | RAM | Comando |
|---------|-------|-----|---------|
| `deepseek-r1:14b` | Orchestrator (reasoning profondo) | ~10GB | `ollama pull deepseek-r1:14b` |
| `qwen2.5-coder:14b` | Strategy/Code gen | ~10GB | `ollama pull qwen2.5-coder:14b` |
| `llama3.2:3b` | Analysis (fast) | ~2GB | `ollama pull llama3.2:3b` |
| `qwen2.5:3b` | Fast decisions | ~2GB | `ollama pull qwen2.5:3b` |
| `nomic-embed-text` | Embeddings RAG | ~300MB | `ollama pull nomic-embed-text` |

### Script Automatico

```bash
#!/bin/bash
# scripts/pull_models.sh

MODELS=(
    "deepseek-r1:14b"
    "qwen2.5-coder:14b" 
    "llama3.2:3b"
    "qwen2.5:3b"
    "nomic-embed-text"
)

for model in "${MODELS[@]}"; do
    echo "📥 Pulling $model..."
    ollama pull "$model"
done

echo "✅ All models ready!"
ollama list
```

### Verifica

```bash
ollama list
# Output atteso:
# NAME                  SIZE
# deepseek-r1:14b       9.1 GB
# qwen2.5-coder:14b     9.1 GB
# llama3.2:3b           2.0 GB
# qwen2.5:3b            1.9 GB
# nomic-embed-text      274 MB
```

---

## 3. SurrealDB Setup

### Avvio Database

```bash
# Database dedicato per llm-evolution
surreal start --log info --user root --pass root \
    file:~/evolution.db

# In altra finestra, verifica connessione
surreal sql --conn http://localhost:8000 \
    --user root --pass root --ns evolution --db main \
    "INFO FOR DB;"
```

### Setup Schema (esegui setup script)

```bash
cd llm-evolution
chmod +x scripts/setup_surrealdb.sh
./scripts/setup_surrealdb.sh
```

---

## 4. Python Environment

```bash
cd llm-evolution

# Crea venv
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt

# Verifica
python -c "import httpx; import numpy; print('✅ OK')"
```

---

## 5. Ingestion Papers

### LLM4EC Papers (12 papers fondamentali)

```bash
# Con SurrealDB attivo
python -m src.knowledge.ingestion.llm4ec

# Verifica
surreal sql --conn http://localhost:8000 \
    --user root --pass root --ns evolution --db main \
    "SELECT count() FROM papers GROUP ALL;"
```

### ArXiv Papers (opzionale, per più context)

```bash
python -m src.knowledge.ingestion.arxiv
```

---

## 6. Test Rapido

```bash
# Test connessione Ollama
curl http://localhost:11434/api/tags

# Test connessione SurrealDB
curl -X POST http://localhost:8000/sql \
    -H "Accept: application/json" \
    -H "NS: evolution" \
    -H "DB: main" \
    --user "root:root" \
    --data "INFO FOR DB;"

# Test Python client
python -c "
from src.llm.client import OllamaClient
client = OllamaClient()
print(client.complete('Say hello', model='llama3.2:3b'))
"
```

---

## 7. Mapping Agent → Modello

| Agent | Modello Ollama | Perché |
|-------|----------------|--------|
| `OrchestratorAgent` | `deepseek-r1:14b` | Reasoning complesso, planning |
| `StrategyAgent` | `qwen2.5-coder:14b` | Genera codice, analizza convergenza |
| `AnalysisAgent` | `llama3.2:3b` | Fast, interpretazione fitness |
| `RAGAgent` | `qwen2.5:3b` + embeddings | Query KB, context retrieval |

---

## ❗ Troubleshooting

### "Ollama non raggiungibile"

```bash
# Verifica se è in esecuzione
pgrep -x ollama
# Se vuoto, avvia:
ollama serve &
```

### "Model not found"

```bash
# Lista modelli locali
ollama list
# Se manca, pull:
ollama pull <model-name>
```

### "SurrealDB connection refused"

```bash
# Verifica porta
lsof -i :8000
# Se occupata, usa altra porta:
surreal start --bind 0.0.0.0:8001 file:~/evolution.db
```

### Memoria insufficiente

```bash
# Con 16GB RAM, usa modelli più piccoli:
# deepseek-r1:7b invece di 14b
# qwen2.5-coder:7b invece di 14b

# Oppure quantizzati:
ollama pull deepseek-r1:14b-q4_0  # ~5GB invece di 9GB
```

---

## 🎯 Checklist Setup Completo

- [ ] Ollama installato e `ollama serve` attivo
- [ ] Modelli pullati (almeno `llama3.2:3b` per test)
- [ ] SurrealDB installato e attivo su `:8000`
- [ ] Schema creato con `setup_surrealdb.sh`
- [ ] Python venv attivo con dipendenze
- [ ] Ingestion papers eseguito
- [ ] Test rapidi OK

---

*Guida setup per LLM-Evolution framework*
