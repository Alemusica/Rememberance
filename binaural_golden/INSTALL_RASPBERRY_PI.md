# 🍓 Rememberance - Raspberry Pi Installation Guide

## ✅ System Ready

Il codice è **pronto per Raspberry Pi** con:
- 🌐 Web interface mobile-first
- 🎵 4 modalità terapeutiche (EMDR, Vibroacoustic, Harmonic Tree, Spectral Sound)
- 🔧 Backend audio flessibile (ALSA/PyAudio)
- 🚀 Autostart con systemd
- 📱 Controllo da smartphone/tablet

---

## 📦 Hardware Necessario

### Configurazione Base (€190)
- **Raspberry Pi 5** (4GB) - €65
- **JAB4 WONDOM AA-JA33285** - 4-channel DSP amplifier - €44
- **4× Dayton DAEX25** - 25mm exciters - €48 (€12 cad.)
- **3× Alpha RV24AF-10** - Control potentiometers - €8
- **Power Supply** 12V 5A - €10
- **Enclosure** 150×100×40mm - €8
- **Misc** (cables, connectors) - €8

### Opzionale (Audiophile Quality)
- **HiFiBerry DAC+ ADC Pro** - €55
- **Linear power supply** - €45

**Vedi**: [`hardware/JAB4_DOCUMENTATION.md`](hardware/JAB4_DOCUMENTATION.md) per schemi di cablaggio completi.

---

## 🚀 Installazione Rapida

### Metodo 1: Script Automatico (Raccomandato)

```bash
# 1. Scarica repository su Raspberry Pi
git clone https://github.com/Alemusica/Rememberance.git
cd Rememberance/binaural_golden

# 2. Esegui installer automatico
sudo chmod +x firmware/pi_setup/install.sh
sudo firmware/pi_setup/install.sh

# 3. Riavvia
sudo reboot
```

**Lo script automatico installa**:
- ✅ Dipendenze sistema (ALSA, Python, avahi)
- ✅ Environment virtuale Python
- ✅ Flask + dependencies (numpy, flask-cors, pyalsaaudio)
- ✅ Copia file applicazione
- ✅ Configura ALSA per JAB4
- ✅ Crea servizio systemd
- ✅ Abilita mDNS (rememberance.local)
- ✅ Configura firewall

**Dopo il riboot**:
```
✨ Web interface disponibile su:
   http://rememberance.local:5000
   http://[IP-RASPBERRY]:5000
```

---

### Metodo 2: Installazione Manuale

#### 1. Aggiorna Sistema
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

#### 2. Installa Dipendenze
```bash
# Audio
sudo apt-get install -y alsa-utils libasound2-dev portaudio19-dev python3-pyaudio

# Python
sudo apt-get install -y python3-pip python3-dev python3-venv git

# Network
sudo apt-get install -y avahi-daemon avahi-utils
```

#### 3. Crea Virtual Environment
```bash
mkdir -p /opt/rememberance
cd /opt/rememberance
python3 -m venv venv
source venv/bin/activate
```

#### 4. Installa Pacchetti Python
```bash
pip install --upgrade pip
pip install numpy==1.24.3
pip install flask==2.3.2
pip install flask-cors==4.0.0
pip install pyalsaaudio==0.10.0
```

#### 5. Copia File Applicazione
```bash
# Dal repository clonato
cp -r ~/Rememberance/binaural_golden/core /opt/rememberance/
cp -r ~/Rememberance/binaural_golden/modules /opt/rememberance/
cp -r ~/Rememberance/binaural_golden/interfaces /opt/rememberance/
```

#### 6. Configura ALSA
```bash
sudo nano /etc/asound.conf
```

Inserisci:
```
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
```

#### 7. Crea Servizio Systemd
```bash
sudo nano /etc/systemd/system/rememberance.service
```

Inserisci:
```ini
[Unit]
Description=Rememberance Vibroacoustic Therapy
After=network.target sound.target

[Service]
Type=simple
User=pi
Group=audio
WorkingDirectory=/opt/rememberance
Environment="PATH=/opt/rememberance/venv/bin"
Environment="AUDIO_BACKEND=alsa"
Environment="FLASK_APP=interfaces/web/app.py"
ExecStart=/opt/rememberance/venv/bin/python interfaces/web/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 8. Abilita e Avvia Servizio
```bash
sudo systemctl daemon-reload
sudo systemctl enable rememberance.service
sudo systemctl start rememberance.service
```

#### 9. Verifica Status
```bash
sudo systemctl status rememberance.service
```

---

## 🎛️ Configurazione JAB4

### Collegamento Hardware

```
JAB4 Outputs → Exciters:
  CH1 (Left Shoulder)  → DAEX25 #1
  CH2 (Right Shoulder) → DAEX25 #2
  CH3 (Left Hip)       → DAEX25 #3
  CH4 (Right Hip)      → DAEX25 #4

Potentiometers:
  POT1 → Master Volume (10kΩ)
  POT2 → Bass Boost (10kΩ)
  POT3 → Treble (10kΩ)
```

### Test Audio
```bash
# Test speaker-test con ALSA
speaker-test -D jab4 -c 2 -t wav

# Test Flask API
curl http://localhost:5000/api/test
```

---

## 📱 Accesso Web Interface

### Dall'interno della rete locale:

1. **Via mDNS** (se avahi funziona):
   ```
   http://rememberance.local:5000
   ```

2. **Via IP address**:
   ```bash
   # Trova IP del Raspberry Pi
   hostname -I
   
   # Apri browser su:
   http://192.168.1.XXX:5000
   ```

3. **Da smartphone/tablet**:
   - Connetti alla stessa rete Wi-Fi
   - Apri browser
   - Inserisci `http://rememberance.local:5000`

---

## 🧪 Test Modalità

### 1. EMDR Bilateral Stimulation
- Seleziona tab "🧠 EMDR"
- Scegli frequenza (432Hz raccomandato)
- Mode: "Golden Phase φ"
- Start → sentirai panning L/R

### 2. Vibroacoustic Panning
- Seleziona tab "🎵 Vibro"
- Carrier: 432 Hz
- Modulation: 0.1 Hz
- Start → vibrazione che si muove lungo il corpo

### 3. Harmonic Tree (NUOVO!)
- Seleziona tab "🌳 Harmonic"
- Fundamental: 432 Hz
- Harmonics: 5-13
- Mode: Fibonacci (φ ratios)
- Growth: 1-60 minuti
- Start → armonici emergono progressivamente

### 4. Spectral Sound (NUOVO!)
- Seleziona tab "⚛️ Spectral"
- Element: Hydrogen/Oxygen/Helium
- Phase: Golden Angle
- Start → righe spettrali atomiche

---

## 🔧 Troubleshooting

### Server non si avvia
```bash
# Check logs
sudo journalctl -u rememberance.service -f

# Verifica dipendenze
/opt/rememberance/venv/bin/python -c "import flask, numpy, alsaaudio"
```

### Audio non funziona
```bash
# Lista device audio
aplay -l

# Test ALSA
speaker-test -D hw:0,0 -c 2

# Verifica backend
curl http://localhost:5000/api/test
# Output dovrebbe mostrare: "backend": "ALSABackend"
```

### Non riesco a raggiungere l'interfaccia web
```bash
# Verifica che Flask sia in ascolto
sudo netstat -tulpn | grep 5000

# Controlla firewall
sudo ufw status

# Test locale
curl http://localhost:5000/api/status
```

### JAB4 non funziona
```bash
# Verifica I2S enabled in raspi-config
sudo raspi-config
# → Interfacing Options → I2C → Enable

# Check /boot/config.txt contiene:
dtparam=i2s=on
dtoverlay=i2s-mmap
```

---

## 🎯 Performance Tips

### Ridurre latenza audio
In `/opt/rememberance/core/audio_backend.py`:
```python
# Line ~85
self.stream = self.pa.open(
    ...
    frames_per_buffer=512,  # Ridurre da 1024 a 512
    ...
)
```

### CPU-friendly per Pi Zero/3
```python
# In modules/harmonic_tree/__init__.py line ~120
SAMPLE_RATE = 44100  # Ridurre da 48000
```

### Auto-restart on crash
Il servizio systemd è già configurato con:
```ini
Restart=always
RestartSec=10
```

---

## 📊 Monitoraggio Sistema

```bash
# CPU temperature
vcgencmd measure_temp

# Memory usage
free -h

# Disk usage
df -h

# Service status
sudo systemctl status rememberance.service

# Live logs
sudo journalctl -u rememberance.service -f
```

---

## 🔄 Aggiornamento Software

```bash
# 1. Stop service
sudo systemctl stop rememberance.service

# 2. Pull updates
cd ~/Rememberance
git pull origin raspberry-pi-modular

# 3. Update app files
sudo cp -r binaural_golden/core /opt/rememberance/
sudo cp -r binaural_golden/modules /opt/rememberance/
sudo cp -r binaural_golden/interfaces /opt/rememberance/

# 4. Restart service
sudo systemctl start rememberance.service
```

---

## 🌐 Accesso Remoto (Opzionale)

### Via Cloudflare Tunnel (Raccomandato)
```bash
# Installa cloudflared
sudo apt install cloudflared

# Setup tunnel
cloudflared tunnel login
cloudflared tunnel create rememberance
cloudflared tunnel route dns rememberance rememberance.yourdomain.com

# Start tunnel
cloudflared tunnel run rememberance
```

### Via ngrok (Temporaneo)
```bash
# Installa ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz
tar xvzf ngrok-*.tgz
sudo mv ngrok /usr/local/bin

# Avvia tunnel
ngrok http 5000
```

---

## 📚 Documentazione Completa

- **Architettura**: [`README_PI_MODULAR.md`](README_PI_MODULAR.md)
- **Hardware JAB4**: [`hardware/JAB4_DOCUMENTATION.md`](hardware/JAB4_DOCUMENTATION.md)
- **API Reference**: [`README_PI_MODULAR.md#api-reference`](README_PI_MODULAR.md#api-reference)
- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md)

---

## 🆘 Supporto

**Issues**: https://github.com/Alemusica/Rememberance/issues

**Email**: [Il tuo contatto]

**Versione**: 2.0.0-pi (December 2025)

---

## ✨ Quick Reference Card

```
📍 Installazione:
   sudo firmware/pi_setup/install.sh

🌐 Web Interface:
   http://rememberance.local:5000

🎛️ Controlli:
   - EMDR: Bilateral audio therapy
   - Vibro: Physical soundboard panning
   - Harmonic: Fibonacci harmonic tree (1-60min growth)
   - Spectral: Atomic element sounds

🔧 Comandi Utili:
   sudo systemctl status rememberance    # Status
   sudo systemctl restart rememberance   # Restart
   sudo journalctl -u rememberance -f    # Logs
   
🧪 Test:
   curl http://localhost:5000/api/test
   speaker-test -D jab4 -c 2
```
