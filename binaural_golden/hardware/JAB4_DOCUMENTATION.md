# JAB4 Audio Processor Hardware Documentation

## Bill of Materials (BOM)

### Main Components

| Component | Model/Spec | Quantity | Unit Price (€) | Total (€) | Source |
|-----------|-----------|----------|----------------|-----------|---------|
| **Audio Processor** | WONDOM AA-JA33285 JAB4 | 1 | 44.00 | 44.00 | Amazon/Aliexpress |
| **Exciters** | Dayton Audio DAEX25 (25mm 40W 8Ω) | 4 | 12.00 | 48.00 | Parts Express |
| **Potentiometers** | Alpha RV24AF-10 10kΩ log (Moog-style) | 3 | 2.50 | 7.50 | Thomann |
| **Enclosure** | ABS project box 150×100×40mm | 1 | 8.00 | 8.00 | RS Components |
| **Raspberry Pi 5** | 4GB RAM (optional for advanced control) | 1 | 65.00 | 65.00 | Official distributors |
| **Power Supply** | 12V 5A DC adapter | 1 | 10.00 | 10.00 | Amazon |
| **Miscellaneous** | Wires, connectors, screws | - | 8.00 | 8.00 | - |
| **TOTAL** | | | | **€190.50** | |

_Without Raspberry Pi: **€125.50**_

---

## JAB4 WONDOM AA-JA33285 Specifications

### Audio Performance
- **DSP Chip**: ADAU1701 (SigmaStudio programmable)
- **Amplifier**: 4-channel Class D
- **Power Output**: 30W RMS per channel @ 8Ω
- **THD+N**: <0.1% @ 1kHz
- **Frequency Response**: 20Hz - 20kHz (±1dB)
- **SNR**: >95dB

### Input Options
- **Bluetooth 5.0**: Qualcomm aptX, AAC, SBC
- **I2S**: Direct digital input (RPi compatible)
- **AUX**: 3.5mm analog stereo input
- **USB**: Configuration and firmware updates

### DSP Features (ADAU1701)
- **Sample Rate**: 48kHz
- **Processing Power**: 28-bit fixed-point
- **Filters**: Parametric EQ, crossover, compressor, limiter
- **Routing**: Flexible 4×4 matrix mixer
- **Latency**: <5ms
- **Programs**: Store up to 4 presets, GPIO-switchable

### Physical
- **Dimensions**: 100mm × 68mm × 22mm
- **Power Input**: 12-24V DC, 2.1mm barrel jack
- **Outputs**: 4× spring terminals (4-8Ω speakers)
- **GPIO**: 4× programmable pins for control
- **LEDs**: Power, Bluetooth, signal indicators

### Operating Conditions
- **Temperature**: -10°C to +60°C
- **Humidity**: 10% - 90% non-condensing

---

## Wiring Diagram

```
                         ┌─────────────────────────┐
                         │   JAB4 AA-JA33285       │
                         │   (ADAU1701 DSP)        │
                         │                         │
                         │  [BT 5.0] [I2S] [AUX]  │
                         │                         │
┌─────────────┐          │  CH1  CH2  CH3  CH4    │
│ Raspberry   │  I2S     │  [+]  [+]  [+]  [+]    │
│ Pi 5        ├──────────┤  [-]  [-]  [-]  [-]    │
│             │          └────┬────┬────┬────┬─────┘
│  [GPIO]     │               │    │    │    │
│   ├─────────┼───────────────┘    │    │    │
│   ├─────────┼────────────────────┘    │    │
│   └─────────┼─────────────────────────┘    │
└─────────────┘                              │
                                             │
    Exciter Layout on Soundboard:            │
                                             │
    ┌───────────────────────────────────┐    │
    │ HEAD (Testa)                      │    │
    │                                   │    │
    │  [CH1 Left]          [CH2 Right] │←───┘
    │     8Ω 40W              8Ω 40W    │
    │                                   │
    │                                   │
    │    Listener Position              │
    │         (supino)                  │
    │                                   │
    │                                   │
    │  [CH3 Left]          [CH4 Right] │
    │     8Ω 40W              8Ω 40W    │
    │                                   │
    │ FEET (Piedi)                      │
    └───────────────────────────────────┘
    
    Soundboard: 1950mm × 600mm × 10mm Spruce
    Springs: 5× 15-20kg (4 corners + 1 center)
```

### Potentiometer Wiring

```
POT 1 (Volume Master):
  Pin 1 (CCW) → GND
  Pin 2 (Wiper) → JAB4 GPIO1 (analog read)
  Pin 3 (CW) → +3.3V

POT 2 (Frequency):
  Pin 1 → GND
  Pin 2 → JAB4 GPIO2
  Pin 3 → +3.3V

POT 3 (Speed/Modulation):
  Pin 1 → GND
  Pin 2 → JAB4 GPIO3
  Pin 3 → +3.3V
```

---

## Connection Steps

### 1. Power Supply
1. Connect 12V 5A power adapter to JAB4 barrel jack
2. Verify power LED illuminates

### 2. Raspberry Pi I2S Connection
| Pi 5 GPIO | Signal | JAB4 Pin |
|-----------|--------|----------|
| GPIO 18 (PCM_CLK) | BCLK | I2S_BCLK |
| GPIO 19 (PCM_FS) | LRCLK | I2S_LRCK |
| GPIO 20 (PCM_DIN) | DATA | I2S_DATA |
| GPIO GND | GND | I2S_GND |

### 3. Speaker Connections
- **CH1 (Head Left)**: Connect to HEAD-LEFT exciter
  - Red wire → CH1 [+]
  - Black wire → CH1 [-]
- **CH2 (Head Right)**: Connect to HEAD-RIGHT exciter
- **CH3 (Feet Left)**: Connect to FEET-LEFT exciter
- **CH4 (Feet Right)**: Connect to FEET-RIGHT exciter

**IMPORTANT**: 
- Use 18AWG speaker wire (minimum)
- Keep wire runs under 2 meters
- Secure connections with spring terminals

### 4. Control Potentiometers
- Mount 3× Alpha RV24AF-10 pots on front panel
- Wire per schematic above
- Label: VOLUME | FREQ | SPEED

---

## DSP Configuration (SigmaStudio)

### Install SigmaStudio
1. Download from Analog Devices website
2. Install on Windows PC
3. Connect JAB4 via USB

### Basic Configuration
1. Open SigmaStudio
2. Create new project: ADAU1701
3. Sample rate: 48kHz

### Signal Flow

```
Input (I2S/BT/AUX)
  │
  ├─► EQ (Parametric, 5-band)
  │
  ├─► Crossover (Optional: separate sub-bass)
  │
  ├─► Matrix Mixer (4×4 routing)
  │
  ├─► Volume Control (per channel)
  │
  ├─► Limiter (prevent clipping)
  │
  └─► Output (CH1-4)
```

### Suggested EQ Settings for EMDR
1. **Band 1 (50Hz)**: +3dB, Q=0.7 (body resonance)
2. **Band 2 (150Hz)**: +2dB, Q=1.0 (warmth)
3. **Band 3 (500Hz)**: -2dB, Q=0.5 (reduce mid-mud)
4. **Band 4 (2kHz)**: +1dB, Q=1.5 (clarity)
5. **Band 5 (8kHz)**: -3dB, Q=0.7 (reduce harshness)

### Limiter Settings
- **Threshold**: -3dBFS
- **Attack**: 1ms
- **Release**: 100ms
- **Ratio**: ∞:1 (brick-wall)

---

## Calibration Procedure

### Step 1: Level Matching
1. Generate 432Hz sine wave @ -20dBFS
2. Play through each channel individually
3. Measure SPL at listener position (use phone app)
4. Adjust per-channel gain in DSP to match ±1dB

### Step 2: Phase Alignment
1. Generate dual-tone test (432Hz + 528Hz)
2. Play stereo (L+R)
3. Use oscilloscope or audio analyzer
4. Adjust delay compensation in DSP if needed

### Step 3: Frequency Response
1. Sweep 20Hz - 20kHz
2. Measure with calibrated microphone
3. Apply corrective EQ in SigmaStudio
4. Target: Flat ±3dB 40Hz-10kHz

### Step 4: Distance Compensation
- **Head exciters**: 150mm from ears
- **Feet exciters**: 1800mm from ears
- Time difference: ~5.2ms
- Add delay to head channels if needed

---

## Troubleshooting

### No Sound
- ☑ Check power supply (12V connected?)
- ☑ Verify speaker connections (no shorts?)
- ☑ Test with Bluetooth (pair phone, play music)
- ☑ Check DSP program loaded (connect SigmaStudio)

### Distortion/Clipping
- ☑ Reduce master volume
- ☑ Check limiter threshold in DSP
- ☑ Verify 8Ω impedance (not 4Ω)
- ☑ Lower input gain if using AUX

### Imbalanced Channels
- ☑ Re-calibrate levels (see procedure above)
- ☑ Check wire connections (loose terminal?)
- ☑ Verify exciter impedance (multimeter test)

### Raspberry Pi I2S Issues
- ☑ Enable I2S in `raspi-config`
- ☑ Check `/boot/config.txt` for `dtparam=i2s=on`
- ☑ Test with `aplay -l` (should show I2S device)
- ☑ Verify wire connections (especially GND)

---

## Maintenance

### Weekly
- Check all wire connections for looseness
- Inspect exciters for damage
- Clean control pots (spray contact cleaner)

### Monthly
- Re-calibrate levels (channel matching)
- Test frequency response sweep
- Check power supply voltage (12V ±0.5V)

### Annually
- Replace potentiometers if noisy
- Check capacitors in power supply
- Update DSP firmware if available

---

## Advanced Modifications

### Add HiFiBerry DAC+ ADC Pro
- **Benefit**: Studio-grade audio quality (114dB SNR)
- **Cost**: ~€40
- **Wiring**: Connect to Pi GPIO, disable internal audio
- **Result**: Bypasses JAB4 for maximum fidelity

### External Control via MIDI
- **Hardware**: USB MIDI controller
- **Software**: Map MIDI CC to parameters via Python
- **Use Case**: Tactile faders for live therapy sessions

### Multi-Zone Support
- **Hardware**: Add second JAB4 unit
- **Wiring**: Parallel I2S feed from Pi
- **Use Case**: Dual soundboards for couples therapy

---

## Safety Warnings

⚠ **Electrical Safety**
- Unplug power before making connections
- Use insulated tools
- Keep liquids away from electronics

⚠ **Audio Safety**
- Start with low volume, increase gradually
- Prolonged exposure >85dB can damage hearing
- Use limiter to prevent sudden loud transients

⚠ **Medical Disclaimer**
- Not a substitute for professional medical care
- Consult healthcare provider for trauma/PTSD treatment
- Stop use if experiencing discomfort

---

## Support Resources

- **JAB4 Manual**: [Wondom website](https://www.wondom.com)
- **ADAU1701 Datasheet**: [Analog Devices](https://www.analog.com)
- **SigmaStudio Software**: [Free download](https://www.analog.com/en/design-center/evaluation-hardware-and-software/software/ss_sigst_02.html)
- **Raspberry Pi I2S Guide**: [Official docs](https://www.raspberrypi.com/documentation/)

---

_Last updated: 2024-01-20_
