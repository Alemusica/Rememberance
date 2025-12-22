#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# REMEMBERANCE - Raspberry Pi Auto-Start Setup
# Makes the therapy system boot-ready and accessible via phone
# ══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SERVICE_NAME="rememberance"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     REMEMBERANCE - Auto-Start Configuration               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. CREATE SYSTEMD SERVICE (auto-start at boot)
# ══════════════════════════════════════════════════════════════════════════════

echo "📦 Creating systemd service..."

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Rememberance Vibroacoustic Therapy Server
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${PROJECT_DIR}/interfaces/web
Environment=PORT=80
ExecStart=${PROJECT_DIR}/venv/bin/python ${PROJECT_DIR}/interfaces/web/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

echo "✅ Service created and enabled"

# ══════════════════════════════════════════════════════════════════════════════
# 2. ALLOW PORT 80 WITHOUT ROOT
# ══════════════════════════════════════════════════════════════════════════════

echo "🔓 Configuring port 80 access..."
sudo setcap 'cap_net_bind_service=+ep' ${PROJECT_DIR}/venv/bin/python

# ══════════════════════════════════════════════════════════════════════════════
# 3. CONFIGURE NETWORK OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🌐 Network Configuration Options:"
echo ""
echo "  1) Local WiFi only (Pi joins your home network)"
echo "  2) Pi Hotspot (Pi creates its own WiFi: 'Rememberance')"
echo "  3) Both (Hotspot + can connect to local WiFi)"
echo ""
read -p "Choose [1/2/3]: " NETWORK_CHOICE

case $NETWORK_CHOICE in
    2|3)
        echo ""
        echo "📶 Setting up WiFi Hotspot..."
        
        # Install hostapd and dnsmasq
        sudo apt install -y hostapd dnsmasq
        
        # Configure hostapd
        sudo tee /etc/hostapd/hostapd.conf > /dev/null << EOF
interface=wlan0
driver=nl80211
ssid=Rememberance
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=golden432
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

        # Point to config file
        sudo sed -i 's|#DAEMON_CONF=""|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd
        
        # Configure dnsmasq for DHCP
        sudo tee /etc/dnsmasq.d/rememberance.conf > /dev/null << EOF
interface=wlan0
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
address=/rememberance.local/192.168.4.1
EOF

        # Configure static IP for wlan0
        sudo tee -a /etc/dhcpcd.conf > /dev/null << EOF

# Rememberance Hotspot
interface wlan0
static ip_address=192.168.4.1/24
nohook wpa_supplicant
EOF

        # Enable services
        sudo systemctl unmask hostapd
        sudo systemctl enable hostapd
        sudo systemctl enable dnsmasq
        
        echo "✅ Hotspot configured!"
        echo ""
        echo "   📱 WiFi Name: Rememberance"
        echo "   🔑 Password:  golden432"
        echo "   🌐 URL:       http://192.168.4.1"
        echo "                 http://rememberance.local"
        ;;
    *)
        echo "✅ Using local WiFi only"
        ;;
esac

# ══════════════════════════════════════════════════════════════════════════════
# 4. CREATE QUICK ACCESS INFO
# ══════════════════════════════════════════════════════════════════════════════

# Get current IP
CURRENT_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE! 🎉                      ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  The system will start automatically on boot.              ║"
echo "║                                                            ║"
if [[ $NETWORK_CHOICE == "2" || $NETWORK_CHOICE == "3" ]]; then
echo "║  📱 HOTSPOT MODE:                                          ║"
echo "║     WiFi: Rememberance                                     ║"
echo "║     Pass: golden432                                        ║"
echo "║     URL:  http://192.168.4.1                               ║"
echo "║                                                            ║"
fi
echo "║  🏠 LOCAL NETWORK:                                         ║"
echo "║     URL:  http://${CURRENT_IP}                             ║"
echo "║                                                            ║"
echo "║  🔧 Commands:                                              ║"
echo "║     sudo systemctl start rememberance   # Start            ║"
echo "║     sudo systemctl stop rememberance    # Stop             ║"
echo "║     sudo systemctl status rememberance  # Status           ║"
echo "║     journalctl -u rememberance -f       # View logs        ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

read -p "🔄 Reboot now to activate? [y/N]: " REBOOT
if [[ $REBOOT == "y" || $REBOOT == "Y" ]]; then
    sudo reboot
fi
