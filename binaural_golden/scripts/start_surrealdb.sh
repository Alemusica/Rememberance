#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# START SURREALDB - Golden Studio Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════
#
# Questo script avvia SurrealDB con il database di ricerca persistente.
# Il database contiene 80+ paper scientifici su:
#   - Vibroacoustic therapy
#   - DML plate design
#   - Multi-exciter optimization
#   - Evolutionary memory / transfer learning
#   - Acoustic Black Holes (ABH)
#
# USAGE:
#   ./scripts/start_surrealdb.sh         # Avvia in foreground
#   ./scripts/start_surrealdb.sh &       # Avvia in background
#   ./scripts/start_surrealdb.sh stop    # Ferma il database
#
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_ROOT/docs/research/surrealdb"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Stop command
if [ "$1" = "stop" ]; then
    echo -e "${YELLOW}Stopping SurrealDB...${NC}"
    pkill -f "surreal start" 2>/dev/null
    echo -e "${GREEN}✓ SurrealDB stopped${NC}"
    exit 0
fi

# Check if already running
if lsof -i :8000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ SurrealDB already running on port 8000${NC}"
    echo "  Use './scripts/start_surrealdb.sh stop' to stop it first"
    exit 1
fi

# Check database path exists
if [ ! -d "$DB_PATH" ]; then
    echo -e "${YELLOW}Creating database directory: $DB_PATH${NC}"
    mkdir -p "$DB_PATH"
fi

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Starting SurrealDB Knowledge Base${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Database: ${YELLOW}$DB_PATH${NC}"
echo -e "  URL:      ${YELLOW}http://localhost:8000${NC}"
echo -e "  Auth:     ${YELLOW}root:root${NC}"
echo -e "  NS/DB:    ${YELLOW}research/knowledge${NC}"
echo ""
echo -e "${GREEN}  Query example:${NC}"
echo '  surreal sql --endpoint http://127.0.0.1:8000 -u root -p root \'
echo '    --ns research --db knowledge <<< "SELECT * FROM paper LIMIT 5"'
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

# Start SurrealDB with RocksDB storage (persistent)
cd "$DB_PATH"
exec surreal start \
    --user root \
    --pass root \
    --bind 0.0.0.0:8000 \
    rocksdb:.
