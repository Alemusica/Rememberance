#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LLM-Evolution - Pull Required Ollama Models
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  LLM-Evolution - Model Setup"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}❌ Ollama non raggiungibile!${NC}"
    echo "   Esegui: ollama serve"
    exit 1
fi

echo -e "${GREEN}✅ Ollama attivo${NC}"
echo ""

# Define models with roles
declare -A MODELS
MODELS["deepseek-r1:14b"]="Orchestrator (reasoning profondo)"
MODELS["qwen2.5-coder:14b"]="Strategy Agent (code generation)"
MODELS["llama3.2:3b"]="Analysis Agent (fast inference)"
MODELS["qwen2.5:3b"]="Fast decisions"
MODELS["nomic-embed-text"]="Embeddings per RAG"

# Alternative smaller models (for limited RAM)
declare -A LITE_MODELS
LITE_MODELS["deepseek-r1:7b"]="Orchestrator LITE"
LITE_MODELS["qwen2.5-coder:7b"]="Strategy LITE"
LITE_MODELS["llama3.2:1b"]="Analysis LITE"
LITE_MODELS["qwen2.5:1.5b"]="Fast decisions LITE"
LITE_MODELS["nomic-embed-text"]="Embeddings"

# Check RAM
RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
if [ -z "$RAM_GB" ]; then
    RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
fi

echo "📊 RAM disponibile: ${RAM_GB:-unknown} GB"
echo ""

# Choose models based on RAM
if [ "${RAM_GB:-0}" -lt 16 ]; then
    echo -e "${YELLOW}⚠️  Meno di 16GB RAM - uso modelli LITE${NC}"
    echo ""
    for model in "${!LITE_MODELS[@]}"; do
        role="${LITE_MODELS[$model]}"
        echo "📥 Pulling $model ($role)..."
        ollama pull "$model"
        echo ""
    done
else
    echo "💪 RAM sufficiente - uso modelli FULL"
    echo ""
    for model in "${!MODELS[@]}"; do
        role="${MODELS[$model]}"
        echo "📥 Pulling $model ($role)..."
        ollama pull "$model"
        echo ""
    done
fi

echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Setup modelli completato!${NC}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Modelli installati:"
ollama list
echo ""
echo "Prossimi step:"
echo "  1. Avvia SurrealDB: ./scripts/setup_surrealdb.sh"
echo "  2. Ingestion: python -m src.knowledge.ingestion.llm4ec"
echo "  3. Test: python examples/basic_usage.py"
