#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  Build Golden Studio.app per macOS
# ═══════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Building Golden Studio.app for macOS                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Pulisci build precedenti
echo "🧹 Cleaning previous builds..."
rm -rf build dist

# Installa py2app se necessario
echo "📦 Installing py2app..."
pip3 install py2app --quiet

# Aggiungi src al PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Build dell'app
echo "🔨 Building app..."
python3 setup_app.py py2app --no-strip

# Verifica
if [ -d "dist/Golden Studio.app" ]; then
    echo ""
    echo "✅ Build completata con successo!"
    echo ""
    echo "📍 L'app si trova in: dist/Golden Studio.app"
    echo ""
    echo "Per installarla:"
    echo "  cp -r 'dist/Golden Studio.app' /Applications/"
    echo ""
    echo "Oppure trascinala nel Dock!"
    
    # Apri la cartella dist
    open dist/
else
    echo "❌ Build fallita!"
    exit 1
fi
