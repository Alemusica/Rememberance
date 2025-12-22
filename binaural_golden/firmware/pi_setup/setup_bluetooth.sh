#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# REMEMBERANCE - Bluetooth Audio Receiver Setup
# Configures Raspberry Pi as Bluetooth speaker/receiver (A2DP Sink)
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     REMEMBERANCE - Bluetooth Audio Setup                  ║"
echo "║     Receive music from your phone via Bluetooth           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. INSTALL BLUETOOTH AUDIO PACKAGES
# ══════════════════════════════════════════════════════════════════════════════

echo "📦 Installing Bluetooth audio packages..."

sudo apt update
sudo apt install -y \
    bluez \
    bluez-tools \
    pulseaudio \
    pulseaudio-module-bluetooth \
    bluealsa

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURE BLUETOOTH FOR A2DP SINK
# ══════════════════════════════════════════════════════════════════════════════

echo "🔧 Configuring Bluetooth as audio receiver..."

# Add user to bluetooth group
sudo usermod -a -G bluetooth $USER

# Configure BlueALSA for A2DP sink
sudo tee /etc/systemd/system/bluealsa.service > /dev/null << 'EOF'
[Unit]
Description=BluALSA Service
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStart=/usr/bin/bluealsa -p a2dp-sink -p a2dp-source

[Install]
WantedBy=multi-user.target
EOF

# ══════════════════════════════════════════════════════════════════════════════
# 3. AUTO-ACCEPT BLUETOOTH CONNECTIONS
# ══════════════════════════════════════════════════════════════════════════════

echo "🤝 Setting up auto-pairing..."

# Create bluetooth agent for auto-accept
sudo tee /usr/local/bin/bt-agent-auto << 'EOF'
#!/bin/bash
# Auto-accept Bluetooth pairing and connections
bt-agent -c NoInputNoOutput &
sleep 2
bluetoothctl << BTCMD
power on
discoverable on
pairable on
agent NoInputNoOutput
default-agent
BTCMD
EOF

sudo chmod +x /usr/local/bin/bt-agent-auto

# Create systemd service for auto-accept
sudo tee /etc/systemd/system/bt-agent.service > /dev/null << EOF
[Unit]
Description=Bluetooth Auto-Accept Agent
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStart=/usr/local/bin/bt-agent-auto
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIGURE PULSEAUDIO FOR BLUETOOTH
# ══════════════════════════════════════════════════════════════════════════════

echo "🔊 Configuring PulseAudio..."

# Enable PulseAudio Bluetooth module
mkdir -p ~/.config/pulse
tee ~/.config/pulse/default.pa > /dev/null << 'EOF'
.include /etc/pulse/default.pa

# Automatically switch to Bluetooth when connected
load-module module-switch-on-connect

# Enable Bluetooth discovery
load-module module-bluetooth-discover
load-module module-bluetooth-policy
EOF

# Allow PulseAudio to run as system service
sudo tee /etc/systemd/system/pulseaudio.service > /dev/null << EOF
[Unit]
Description=PulseAudio Sound Server
After=sound.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/bin/pulseaudio --daemonize=no
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ══════════════════════════════════════════════════════════════════════════════
# 5. SET BLUETOOTH DEVICE NAME
# ══════════════════════════════════════════════════════════════════════════════

echo "📱 Setting Bluetooth name..."

# Set friendly Bluetooth name
sudo sed -i 's/#Name = .*/Name = Rememberance/' /etc/bluetooth/main.conf
sudo sed -i 's/#Class = .*/Class = 0x200414/' /etc/bluetooth/main.conf  # Audio device class
sudo sed -i 's/#DiscoverableTimeout = .*/DiscoverableTimeout = 0/' /etc/bluetooth/main.conf  # Always discoverable

# ══════════════════════════════════════════════════════════════════════════════
# 6. CREATE HELPER SCRIPTS
# ══════════════════════════════════════════════════════════════════════════════

echo "🛠️ Creating helper scripts..."

# Script to pair new device
sudo tee /usr/local/bin/rememberance-bt-pair << 'EOF'
#!/bin/bash
echo "🔵 Rememberance Bluetooth Pairing"
echo "=================================="
echo ""
echo "1. Make sure Bluetooth is enabled on your phone"
echo "2. Search for 'Rememberance' in Bluetooth settings"
echo "3. Tap to pair"
echo ""
echo "Waiting for connections..."
echo "(Press Ctrl+C to exit)"
echo ""

bluetoothctl << BTCMD
power on
discoverable on
pairable on
scan on
BTCMD

# Keep running and show connections
journalctl -u bluetooth -f
EOF

sudo chmod +x /usr/local/bin/rememberance-bt-pair

# Script to check Bluetooth status
sudo tee /usr/local/bin/rememberance-bt-status << 'EOF'
#!/bin/bash
echo "🔵 Bluetooth Status"
echo "==================="
echo ""
echo "📡 Controller:"
bluetoothctl show | grep -E "Name|Powered|Discoverable|Pairable"
echo ""
echo "📱 Paired Devices:"
bluetoothctl devices Paired
echo ""
echo "🔊 Connected Audio:"
bluetoothctl devices Connected
echo ""
echo "🎵 Audio Status:"
pactl list short sinks
EOF

sudo chmod +x /usr/local/bin/rememberance-bt-status

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENABLE ALL SERVICES
# ══════════════════════════════════════════════════════════════════════════════

echo "🚀 Enabling services..."

sudo systemctl daemon-reload
sudo systemctl enable bluetooth
sudo systemctl enable bluealsa
sudo systemctl enable bt-agent
sudo systemctl enable pulseaudio

# Start services now
sudo systemctl restart bluetooth
sudo systemctl start bluealsa
sudo systemctl start bt-agent
sudo systemctl start pulseaudio

# ══════════════════════════════════════════════════════════════════════════════
# 8. DONE!
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              BLUETOOTH SETUP COMPLETE! 🎵                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  📱 FROM YOUR PHONE:                                       ║"
echo "║     1. Open Bluetooth settings                             ║"
echo "║     2. Search for 'Rememberance'                           ║"
echo "║     3. Tap to pair and connect                             ║"
echo "║     4. Play music - it comes out of the Pi!                ║"
echo "║                                                            ║"
echo "║  🎛️ HELPER COMMANDS:                                       ║"
echo "║     rememberance-bt-pair    # Pair new device              ║"
echo "║     rememberance-bt-status  # Check status                 ║"
echo "║                                                            ║"
echo "║  🔊 AUDIO ROUTING:                                         ║"
echo "║     Phone Bluetooth → Pi → DAC → Transducers               ║"
echo "║                                                            ║"
echo "║  💡 TIP: Bluetooth audio + Web therapy can work together!  ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

read -p "🔄 Reboot now to complete setup? [y/N]: " REBOOT
if [[ $REBOOT == "y" || $REBOOT == "Y" ]]; then
    sudo reboot
fi
