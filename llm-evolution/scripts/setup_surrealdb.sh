#!/bin/bash
# 🚀 Setup script for SurrealDB - LLM-Evolution Knowledge Base

set -e

echo "🚀 Setting up SurrealDB for LLM-Evolution..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DB_DIR="$HOME/.config/surrealdb"
DB_FILE="$DB_DIR/evolution.db"
SURREAL_URL="http://localhost:8000"
SURREAL_USER="root"
SURREAL_PASS="root"
NAMESPACE="evolution"
DATABASE="knowledge"

# Check if surreal CLI is installed
if ! command -v surreal &> /dev/null; then
    echo -e "${RED}❌ surreal CLI not found.${NC}"
    echo -e "${YELLOW}Install with: curl -sSf https://install.surrealdb.com | sh${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SurrealDB CLI found${NC}"

# Create directory if it doesn't exist
if [ ! -d "$DB_DIR" ]; then
    echo -e "${YELLOW}📁 Creating directory: $DB_DIR${NC}"
    mkdir -p "$DB_DIR"
fi

# Check if SurrealDB is already running
if curl -s "$SURREAL_URL/health" &> /dev/null; then
    echo -e "${YELLOW}⚠️  SurrealDB appears to be already running on $SURREAL_URL${NC}"
    echo -e "${YELLOW}   Skipping server start...${NC}"
else
    # Start SurrealDB in background
    echo -e "${GREEN}🔧 Starting SurrealDB...${NC}"
    surreal start --log warn --user "$SURREAL_USER" --pass "$SURREAL_PASS" \
        file:"$DB_FILE" &
    
    SURREAL_PID=$!
    echo -e "${GREEN}   Started with PID: $SURREAL_PID${NC}"
    
    # Wait for server to be ready
    echo -e "${YELLOW}⏳ Waiting for SurrealDB to be ready...${NC}"
    for i in {1..30}; do
        if curl -s "$SURREAL_URL/health" &> /dev/null; then
            break
        fi
        sleep 1
    done
    
    if ! curl -s "$SURREAL_URL/health" &> /dev/null; then
        echo -e "${RED}❌ SurrealDB failed to start${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ SurrealDB is running on $SURREAL_URL${NC}"

# Create namespace and database via curl
echo -e "${GREEN}📦 Creating namespace '$NAMESPACE' and database '$DATABASE'...${NC}"

# Create namespace
curl -X POST "$SURREAL_URL/sql" \
    -u "$SURREAL_USER:$SURREAL_PASS" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d "USE NS $NAMESPACE;" &> /dev/null

# Create database
curl -X POST "$SURREAL_URL/sql" \
    -u "$SURREAL_USER:$SURREAL_PASS" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "surreal-ns: $NAMESPACE" \
    -d "USE DB $DATABASE;" &> /dev/null

echo -e "${GREEN}✅ Namespace and database created${NC}"

# Create initial schema
echo -e "${GREEN}📋 Creating initial schema...${NC}"

# Schema for 'paper' table (based on surrealdb.py Paper class)
curl -X POST "$SURREAL_URL/sql" \
    -u "$SURREAL_USER:$SURREAL_PASS" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "surreal-ns: $NAMESPACE" \
    -H "surreal-db: $DATABASE" \
    -d "DEFINE TABLE paper SCHEMAFULL;
        DEFINE FIELD title ON paper TYPE string;
        DEFINE FIELD authors ON paper TYPE array;
        DEFINE FIELD year ON paper TYPE int;
        DEFINE FIELD abstract ON paper TYPE string;
        DEFINE FIELD source ON paper TYPE string DEFAULT 'arxiv';
        DEFINE FIELD tags ON paper TYPE array DEFAULT [];
        DEFINE FIELD embedding ON paper TYPE array;
        DEFINE FIELD url ON paper TYPE string;
        DEFINE INDEX paper_title ON paper FIELDS title;" &> /dev/null

# Schema for 'ltm_insight' table
curl -X POST "$SURREAL_URL/sql" \
    -u "$SURREAL_USER:$SURREAL_PASS" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "surreal-ns: $NAMESPACE" \
    -H "surreal-db: $DATABASE" \
    -d "DEFINE TABLE ltm_insight SCHEMAFULL;
        DEFINE FIELD domain ON ltm_insight TYPE string;
        DEFINE FIELD insight ON ltm_insight TYPE string;
        DEFINE FIELD confidence ON ltm_insight TYPE float;
        DEFINE FIELD metadata ON ltm_insight TYPE object DEFAULT {};
        DEFINE FIELD created_at ON ltm_insight TYPE datetime DEFAULT time::now();
        DEFINE INDEX ltm_domain ON ltm_insight FIELDS domain;" &> /dev/null

echo -e "${GREEN}✅ Schema created${NC}"

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ SurrealDB Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}URL:${NC}      $SURREAL_URL"
echo -e "  ${YELLOW}Auth:${NC}     $SURREAL_USER:$SURREAL_PASS"
echo -e "  ${YELLOW}Namespace:${NC} $NAMESPACE"
echo -e "  ${YELLOW}Database:${NC}  $DATABASE"
echo -e "  ${YELLOW}Storage:${NC}   $DB_FILE"
echo ""
echo -e "${GREEN}📝 Query example:${NC}"
echo "  curl -X POST $SURREAL_URL/sql \\"
echo "    -u $SURREAL_USER:$SURREAL_PASS \\"
echo "    -H \"surreal-ns: $NAMESPACE\" \\"
echo "    -H \"surreal-db: $DATABASE\" \\"
echo "    -d \"SELECT * FROM paper LIMIT 5;\""
echo ""
echo -e "${GREEN}🎉 Ready to use!${NC}"
