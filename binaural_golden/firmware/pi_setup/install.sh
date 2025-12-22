#!/bin/bash
###############################################################################
# Rememberance - Raspberry Pi Setup Script
# 
# Automated installation for Raspberry Pi 5 + JAB4 audio processor
# 
# This script will:
# 1. Update system packages
# 2. Install Python dependencies
# 3. Configure ALSA for JAB4 (I2S/Bluetooth)
# 4. Install Flask web server
# 5. Set up systemd service for autostart
# 6. Configure firewall for web access
# 
# Usage:
#   chmod +x install.sh
#   sudo ./install.sh
###############################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════╗"
echo "║   REMEMBERANCE - Raspberry Pi Setup            ║"
echo "║   Vibroacoustic Therapy System                 ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Detect Pi model
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    echo "📟 Detected: $PI_MODEL"
else
    echo "⚠️ Not running on Raspberry Pi - continuing anyway for testing"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 1: Update System Packages"
echo "════════════════════════════════════════════════"
apt-get update
apt-get upgrade -y

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 2: Install System Dependencies"
echo "════════════════════════════════════════════════"

# Audio libraries
apt-get install -y \
    alsa-utils \
    libasound2-dev \
    portaudio19-dev \
    python3-pyaudio

# Python development
apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git

# Network tools
apt-get install -y \
    avahi-daemon \
    avahi-utils

echo "✓ System dependencies installed"

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 3: Create Python Virtual Environment"
echo "════════════════════════════════════════════════"

# Create app directory
APP_DIR="/opt/rememberance"
mkdir -p $APP_DIR
cd $APP_DIR

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

echo "✓ Virtual environment created at $APP_DIR/venv"

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 4: Install Python Dependencies"
echo "════════════════════════════════════════════════"

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install numpy==1.24.3
pip install flask==2.3.2
pip install flask-cors==4.0.0
pip install pyalsaaudio==0.10.0

echo "✓ Python packages installed"

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 5: Copy Application Files"
echo "════════════════════════════════════════════════"

# Copy source files (assumes script is run from repo root)
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
echo "Repository: $REPO_DIR"

cp -r $REPO_DIR/core $APP_DIR/
cp -r $REPO_DIR/modules $APP_DIR/
cp -r $REPO_DIR/interfaces $APP_DIR/

echo "✓ Application files copied"

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 6: Configure ALSA for JAB4"
echo "════════════════════════════════════════════════"

# Create ALSA config for JAB4 (I2S or Bluetooth)
cat > /etc/asound.conf <<'EOF'
# Rememberance ALSA Configuration
# JAB4 WONDOM AA-JA33285 Audio Processor

pcm.!default {
    type plug
    slave.pcm "jab4"
}

pcm.jab4 {
    type hw
    card 0
    device 0
}

ctl.!default {
    type hw
    card 0
}
EOF

echo "✓ ALSA configured for JAB4"

# Test audio devices
echo ""
echo "Available audio devices:"
aplay -l

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 7: Create Systemd Service"
echo "════════════════════════════════════════════════"

cat > /etc/systemd/system/rememberance.service <<EOF
[Unit]
Description=Rememberance Vibroacoustic Therapy Server
After=network.target sound.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
Environment="AUDIO_BACKEND=alsa"
ExecStart=$APP_DIR/venv/bin/python interfaces/web/app.py
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

echo "✓ Systemd service created"

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 8: Configure Avahi (mDNS)"
echo "════════════════════════════════════════════════"

# Enable Avahi for .local hostname resolution
systemctl enable avahi-daemon
systemctl start avahi-daemon

echo "✓ Avahi configured - accessible at rememberance.local"

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 9: Configure Firewall"
echo "════════════════════════════════════════════════"

# Allow Flask port (5000)
if command -v ufw &> /dev/null; then
    ufw allow 5000/tcp
    echo "✓ Firewall configured (port 5000 open)"
else
    echo "⚠️ UFW not installed - skipping firewall config"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "STEP 10: Enable and Start Service"
echo "════════════════════════════════════════════════"

systemctl enable rememberance.service
systemctl start rememberance.service

echo "✓ Service enabled and started"

echo ""
echo "════════════════════════════════════════════════"
echo "INSTALLATION COMPLETE! 🎉"
echo "════════════════════════════════════════════════"
echo ""
echo "📱 Access the web interface:"
echo "   Local:   http://localhost:5000"
echo "   Network: http://$(hostname -I | awk '{print $1}'):5000"
echo "   mDNS:    http://rememberance.local:5000"
echo ""
echo "🔧 Manage the service:"
echo "   Status:  sudo systemctl status rememberance"
echo "   Stop:    sudo systemctl stop rememberance"
echo "   Restart: sudo systemctl restart rememberance"
echo "   Logs:    sudo journalctl -u rememberance -f"
echo ""
echo "🔊 Audio output:"
echo "   Check:   aplay -l"
echo "   Test:    speaker-test -c 2 -t sine"
echo ""
echo "════════════════════════════════════════════════"

# Show service status
echo ""
echo "Service status:"
systemctl status rememberance.service --no-pager

echo ""
echo "✓ Setup complete! Reboot recommended."
echo "  sudo reboot"
