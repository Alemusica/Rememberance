#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Flash Rememberance to Raspberry Pi SD Card
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   1. Insert Pi SD card into Mac
#   2. Run: ./tools/flash_to_sd.sh
#
# This script:
#   - Detects mounted Pi rootfs partition
#   - Copies app to /home/pi/rememberance/
#   - Sets up autostart service
#   - Configures first-boot setup
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🎵 Rememberance - Flash to SD Card${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Find Pi SD card
# ─────────────────────────────────────────────────────────────────────────────

# Look for common Pi rootfs mount points
ROOTFS=""

# Check for "rootfs" volume (common name)
if [ -d "/Volumes/rootfs" ]; then
    ROOTFS="/Volumes/rootfs"
elif [ -d "/Volumes/root" ]; then
    ROOTFS="/Volumes/root"
else
    # Find any volume with /home/pi or /etc/raspberry
    for vol in /Volumes/*; do
        if [ -d "$vol/home/pi" ] || [ -f "$vol/etc/rpi-issue" ]; then
            ROOTFS="$vol"
            break
        fi
    done
fi

if [ -z "$ROOTFS" ]; then
    echo -e "${RED}❌ Pi SD card not found!${NC}"
    echo ""
    echo "Please:"
    echo "  1. Insert the Pi SD card into your Mac"
    echo "  2. Wait for it to mount"
    echo "  3. Run this script again"
    echo ""
    echo "Available volumes:"
    ls -la /Volumes/
    exit 1
fi

echo -e "${GREEN}✓ Found Pi rootfs at: $ROOTFS${NC}"

# Verify it's a Pi filesystem
if [ ! -d "$ROOTFS/home" ]; then
    echo -e "${RED}❌ This doesn't look like a Pi rootfs (no /home directory)${NC}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Create target directory
# ─────────────────────────────────────────────────────────────────────────────

PI_HOME="$ROOTFS/home/pi"
TARGET_DIR="$PI_HOME/rememberance"

# Check if pi user exists
if [ ! -d "$PI_HOME" ]; then
    echo -e "${YELLOW}⚠ /home/pi not found, creating...${NC}"
    sudo mkdir -p "$PI_HOME"
fi

echo -e "${BLUE}📁 Installing to: $TARGET_DIR${NC}"

# Create target directory
sudo mkdir -p "$TARGET_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Copy application files
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}📦 Copying application files...${NC}"

# Copy main source
sudo rsync -av --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '.DS_Store' \
    --exclude 'tools' \
    "$PROJECT_DIR/" "$TARGET_DIR/"

# Copy firmware scripts
if [ -d "$PROJECT_DIR/firmware" ]; then
    sudo cp -r "$PROJECT_DIR/firmware" "$TARGET_DIR/"
fi

echo -e "${GREEN}✓ Files copied${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# Create launcher script
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}🚀 Creating launcher script...${NC}"

sudo tee "$TARGET_DIR/run.sh" > /dev/null << 'LAUNCHER'
#!/bin/bash
# Rememberance Launcher
cd "$(dirname "$0")"
export DISPLAY=:0

# Check if running headless
if [ -z "$DISPLAY" ] && command -v Xvfb &> /dev/null; then
    Xvfb :0 -screen 0 1024x768x24 &
    export DISPLAY=:0
fi

# Run the app
python3 src/golden_studio.py "$@"
LAUNCHER

sudo chmod +x "$TARGET_DIR/run.sh"

# ─────────────────────────────────────────────────────────────────────────────
# Create systemd service for autostart
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}⚙️  Creating systemd service...${NC}"

SYSTEMD_DIR="$ROOTFS/etc/systemd/system"
sudo mkdir -p "$SYSTEMD_DIR"

sudo tee "$SYSTEMD_DIR/rememberance.service" > /dev/null << 'SERVICE'
[Unit]
Description=Rememberance Sound Therapy
After=graphical.target sound.target
Wants=graphical.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rememberance
ExecStart=/home/pi/rememberance/run.sh
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=graphical.target
SERVICE

# Enable the service (create symlink)
sudo mkdir -p "$ROOTFS/etc/systemd/system/graphical.target.wants"
sudo ln -sf ../rememberance.service "$ROOTFS/etc/systemd/system/graphical.target.wants/rememberance.service" 2>/dev/null || true

echo -e "${GREEN}✓ Systemd service created${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# Create first-boot setup script
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}🔧 Creating first-boot setup...${NC}"

sudo tee "$TARGET_DIR/first_boot_setup.sh" > /dev/null << 'FIRSTBOOT'
#!/bin/bash
# First boot setup for Rememberance
# Run this once after booting the Pi

set -e

echo "🎵 Rememberance First Boot Setup"
echo "================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update

# Install dependencies
echo "📦 Installing dependencies..."
sudo apt install -y \
    python3-pip \
    python3-tk \
    python3-numpy \
    python3-scipy \
    python3-pil \
    python3-pil.imagetk \
    portaudio19-dev \
    python3-pyaudio \
    libatlas-base-dev \
    libasound2-dev

# Install Python packages
echo "🐍 Installing Python packages..."
pip3 install --user sounddevice simpleaudio

# Set permissions
echo "🔑 Setting permissions..."
sudo usermod -a -G audio,gpio,i2c,spi pi

# Enable service
echo "⚙️ Enabling Rememberance service..."
sudo systemctl daemon-reload
sudo systemctl enable rememberance.service

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start manually: ~/rememberance/run.sh"
echo "To start as service: sudo systemctl start rememberance"
echo ""
echo "🔄 Reboot recommended: sudo reboot"
FIRSTBOOT

sudo chmod +x "$TARGET_DIR/first_boot_setup.sh"

# ─────────────────────────────────────────────────────────────────────────────
# Set ownership
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}👤 Setting ownership to pi:pi...${NC}"

# Get pi user/group IDs from the Pi filesystem
PI_UID=$(grep "^pi:" "$ROOTFS/etc/passwd" 2>/dev/null | cut -d: -f3 || echo "1000")
PI_GID=$(grep "^pi:" "$ROOTFS/etc/group" 2>/dev/null | cut -d: -f3 || echo "1000")

sudo chown -R "$PI_UID:$PI_GID" "$TARGET_DIR"

echo -e "${GREEN}✓ Ownership set${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# Check boot partition for config
# ─────────────────────────────────────────────────────────────────────────────

BOOT=""
if [ -d "/Volumes/boot" ]; then
    BOOT="/Volumes/boot"
elif [ -d "/Volumes/bootfs" ]; then
    BOOT="/Volumes/bootfs"
fi

if [ -n "$BOOT" ]; then
    echo -e "${BLUE}📝 Configuring boot partition...${NC}"
    
    # Enable I2C, SPI in config.txt
    if [ -f "$BOOT/config.txt" ]; then
        # Add audio configuration if not present
        if ! grep -q "dtparam=audio=on" "$BOOT/config.txt"; then
            echo "" | sudo tee -a "$BOOT/config.txt"
            echo "# Rememberance Audio Configuration" | sudo tee -a "$BOOT/config.txt"
            echo "dtparam=audio=on" | sudo tee -a "$BOOT/config.txt"
        fi
        
        # Enable I2C for potential DAC
        if ! grep -q "dtparam=i2c_arm=on" "$BOOT/config.txt"; then
            echo "dtparam=i2c_arm=on" | sudo tee -a "$BOOT/config.txt"
        fi
        
        echo -e "${GREEN}✓ Boot config updated${NC}"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ Rememberance flashed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Safely eject the SD card"
echo "  2. Insert into Raspberry Pi"
echo "  3. Boot the Pi"
echo "  4. Run first-boot setup:"
echo ""
echo -e "     ${BLUE}cd ~/rememberance && ./first_boot_setup.sh${NC}"
echo ""
echo "  5. After reboot, the app will autostart!"
echo ""
echo -e "${BLUE}Manual launch: ~/rememberance/run.sh${NC}"
echo ""
