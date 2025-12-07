#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TEST DI PRECISIONE - PHASE & BEATS                        ║
║                                                                              ║
║   Verifica matematica della precisione di:                                   ║
║   1. Phase cancellation (180° = silenzio quando sommato)                     ║
║   2. Beat frequency (differenza tra L e R)                                   ║
║   3. Phase offset in gradi                                                   ║
║   4. Golden angle (137.5°)                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import sys

# Costanti
PHI = 1.618033988749894848204586834365638117720309179805762862135
PHI_CONJUGATE = 0.618033988749894848204586834365638117720309179805762862135
GOLDEN_ANGLE_DEG = 360.0 / (PHI * PHI)  # ≈ 137.5077640500378546°
SAMPLE_RATE = 44100

def deg_to_rad(deg):
    return deg * np.pi / 180.0

def generate_stereo_signal(freq_left, freq_right, phase_deg, duration, sample_rate=SAMPLE_RATE):
    """
    Genera segnale stereo con controllo preciso della fase.
    
    Left:  sin(2π * freq_left * t)
    Right: sin(2π * freq_right * t + phase_rad)
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    
    phase_rad = deg_to_rad(phase_deg)
    
    left = np.sin(2 * np.pi * freq_left * t)
    right = np.sin(2 * np.pi * freq_right * t + phase_rad)
    
    return left, right, t

def test_phase_cancellation():
    """
    TEST 1: Phase cancellation a 180°
    
    Se freq_left == freq_right e phase = 180°,
    allora left + right ≈ 0 (cancellazione perfetta)
    """
    print("\n" + "="*70)
    print("TEST 1: PHASE CANCELLATION (180°)")
    print("="*70)
    
    freq = 440.0  # Stessa frequenza
    duration = 0.1
    
    # Test con 180°
    left, right, t = generate_stereo_signal(freq, freq, 180.0, duration)
    
    # Somma dei segnali (mono mix)
    mono_mix = left + right
    
    # RMS della somma (dovrebbe essere ~0)
    rms = np.sqrt(np.mean(mono_mix**2))
    max_abs = np.max(np.abs(mono_mix))
    
    print(f"  Frequenza: {freq} Hz (uguale su entrambi i canali)")
    print(f"  Fase: 180°")
    print(f"  RMS del mono mix: {rms:.10e}")
    print(f"  Max assoluto: {max_abs:.10e}")
    
    # Verifica
    passed = rms < 1e-10
    print(f"  ✅ PASSED" if passed else f"  ❌ FAILED (RMS dovrebbe essere < 1e-10)")
    
    # Test con 0°
    left0, right0, t = generate_stereo_signal(freq, freq, 0.0, duration)
    mono_mix0 = left0 + right0
    rms0 = np.sqrt(np.mean(mono_mix0**2))
    
    print(f"\n  Confronto con fase 0°:")
    print(f"  RMS mono mix (0°): {rms0:.6f}")
    print(f"  Rapporto cancellazione: {rms/rms0:.2e}" if rms0 > 0 else "  Rapporto: N/A")
    
    return passed

def test_beat_frequency():
    """
    TEST 2: Beat frequency
    
    Beat = |freq_right - freq_left|
    Il battimento dovrebbe avere esattamente questa frequenza.
    """
    print("\n" + "="*70)
    print("TEST 2: BEAT FREQUENCY")
    print("="*70)
    
    freq_left = 432.0
    freq_right = 440.0
    expected_beat = abs(freq_right - freq_left)  # 8 Hz
    duration = 2.0  # 2 secondi per vedere bene i battimenti
    
    left, right, t = generate_stereo_signal(freq_left, freq_right, 0.0, duration)
    
    # Il battimento si sente nel cervello, ma possiamo verificare
    # che la differenza di fase tra i canali oscilla alla frequenza beat
    
    # Metodo: analizza la correlazione incrociata
    # O più semplice: verifica che il prodotto L*R oscilli a freq_beat
    product = left * right
    
    # FFT del prodotto
    fft_product = np.fft.rfft(product)
    freqs = np.fft.rfftfreq(len(product), 1/SAMPLE_RATE)
    
    # Trova il picco
    magnitude = np.abs(fft_product)
    # Ignora DC e frequenze molto alte
    valid_range = (freqs > 1) & (freqs < 100)
    peak_idx = np.argmax(magnitude * valid_range)
    measured_beat = freqs[peak_idx]
    
    print(f"  Freq Left: {freq_left} Hz")
    print(f"  Freq Right: {freq_right} Hz")
    print(f"  Expected Beat: {expected_beat} Hz")
    print(f"  Measured Beat (from L*R product): {measured_beat:.2f} Hz")
    
    # La frequenza del prodotto di due sinusoidi è la somma e differenza
    # sin(a)*sin(b) = 0.5*[cos(a-b) - cos(a+b)]
    # Quindi dovremmo vedere un picco a |freq_right - freq_left| = 8 Hz
    
    error = abs(measured_beat - expected_beat)
    passed = error < 1.0  # Entro 1 Hz
    
    print(f"  Errore: {error:.4f} Hz")
    print(f"  ✅ PASSED" if passed else f"  ❌ FAILED")
    
    return passed

def test_phase_angles():
    """
    TEST 3: Verifica che gli angoli di fase siano precisi
    """
    print("\n" + "="*70)
    print("TEST 3: PHASE ANGLE PRECISION")
    print("="*70)
    
    test_angles = [0, 45, 90, 137.5, 180, 270, 360]
    freq = 100.0  # Bassa frequenza per vedere meglio
    duration = 0.01  # 1 ciclo a 100 Hz
    
    all_passed = True
    
    for angle in test_angles:
        left, right, t = generate_stereo_signal(freq, freq, angle, duration)
        
        # Al tempo t=0:
        # left[0] = sin(0) = 0
        # right[0] = sin(phase_rad)
        expected_right_at_0 = np.sin(deg_to_rad(angle))
        actual_right_at_0 = right[0]
        
        error = abs(expected_right_at_0 - actual_right_at_0)
        passed = error < 1e-10
        
        status = "✅" if passed else "❌"
        print(f"  {status} Angle {angle:6.1f}°: expected sin({angle}°) = {expected_right_at_0:+.6f}, got {actual_right_at_0:+.6f}, error = {error:.2e}")
        
        all_passed = all_passed and passed
    
    return all_passed

def test_golden_angle():
    """
    TEST 4: Verifica del Golden Angle
    """
    print("\n" + "="*70)
    print("TEST 4: GOLDEN ANGLE (φ)")
    print("="*70)
    
    # Golden angle = 360° / φ²
    calculated = 360.0 / (PHI * PHI)
    expected = 137.5077640500378546
    
    print(f"  φ (Golden Ratio): {PHI}")
    print(f"  φ² = {PHI * PHI}")
    print(f"  360° / φ² = {calculated}°")
    print(f"  Expected: {expected}°")
    
    error = abs(calculated - expected)
    passed = error < 1e-10
    
    print(f"  Error: {error:.2e}")
    print(f"  ✅ PASSED" if passed else f"  ❌ FAILED")
    
    # Verifica anche che φ² = φ + 1
    phi_squared = PHI * PHI
    phi_plus_one = PHI + 1
    identity_error = abs(phi_squared - phi_plus_one)
    identity_passed = identity_error < 1e-10
    
    print(f"\n  Verifica identità φ² = φ + 1:")
    print(f"  φ² = {phi_squared}")
    print(f"  φ + 1 = {phi_plus_one}")
    print(f"  Error: {identity_error:.2e}")
    print(f"  ✅ PASSED" if identity_passed else f"  ❌ FAILED")
    
    return passed and identity_passed

def test_frequency_precision():
    """
    TEST 5: Precisione della frequenza generata
    """
    print("\n" + "="*70)
    print("TEST 5: FREQUENCY PRECISION")
    print("="*70)
    
    test_freqs = [432.0, 440.0, 528.0, 639.0]
    duration = 1.0
    
    all_passed = True
    
    for expected_freq in test_freqs:
        left, right, t = generate_stereo_signal(expected_freq, expected_freq, 0, duration)
        
        # FFT per misurare la frequenza
        fft = np.fft.rfft(left)
        freqs = np.fft.rfftfreq(len(left), 1/SAMPLE_RATE)
        
        # Trova il picco
        peak_idx = np.argmax(np.abs(fft))
        measured_freq = freqs[peak_idx]
        
        error = abs(measured_freq - expected_freq)
        passed = error < 1.0  # Entro 1 Hz (limitazione FFT resolution)
        
        status = "✅" if passed else "❌"
        print(f"  {status} Expected: {expected_freq:.1f} Hz, Measured: {measured_freq:.1f} Hz, Error: {error:.4f} Hz")
        
        all_passed = all_passed and passed
    
    return all_passed

def test_stereo_independence():
    """
    TEST 6: Verifica che L e R siano indipendenti
    """
    print("\n" + "="*70)
    print("TEST 6: STEREO CHANNEL INDEPENDENCE")
    print("="*70)
    
    # Frequenze diverse su ogni canale
    freq_left = 200.0
    freq_right = 300.0
    duration = 1.0
    
    left, right, t = generate_stereo_signal(freq_left, freq_right, 0, duration)
    
    # FFT di ogni canale
    fft_left = np.fft.rfft(left)
    fft_right = np.fft.rfft(right)
    freqs = np.fft.rfftfreq(len(left), 1/SAMPLE_RATE)
    
    # Trova i picchi
    peak_left_idx = np.argmax(np.abs(fft_left))
    peak_right_idx = np.argmax(np.abs(fft_right))
    
    measured_left = freqs[peak_left_idx]
    measured_right = freqs[peak_right_idx]
    
    passed_left = abs(measured_left - freq_left) < 1.0
    passed_right = abs(measured_right - freq_right) < 1.0
    
    print(f"  Left channel: expected {freq_left} Hz, measured {measured_left:.1f} Hz {'✅' if passed_left else '❌'}")
    print(f"  Right channel: expected {freq_right} Hz, measured {measured_right:.1f} Hz {'✅' if passed_right else '❌'}")
    
    # Verifica che non ci sia leakage tra canali
    # Il picco nel canale sinistro non dovrebbe apparire nel destro e viceversa
    left_at_300 = np.abs(fft_left[np.argmin(np.abs(freqs - 300))])
    right_at_200 = np.abs(fft_right[np.argmin(np.abs(freqs - 200))])
    
    left_peak = np.abs(fft_left[peak_left_idx])
    right_peak = np.abs(fft_right[peak_right_idx])
    
    leakage_left = left_at_300 / left_peak if left_peak > 0 else 0
    leakage_right = right_at_200 / right_peak if right_peak > 0 else 0
    
    no_leakage = leakage_left < 0.01 and leakage_right < 0.01
    
    print(f"  Leakage L→R: {leakage_left*100:.2f}%")
    print(f"  Leakage R→L: {leakage_right*100:.2f}%")
    print(f"  {'✅ No significant leakage' if no_leakage else '❌ Leakage detected'}")
    
    return passed_left and passed_right and no_leakage

def test_molecular_phase():
    """
    TEST 7: Verifica che le fasi molecolari (angoli di legame) siano corrette
    """
    print("\n" + "="*70)
    print("TEST 7: MOLECULAR BOND ANGLES AS PHASE")
    print("="*70)
    
    # Angoli di legame famosi
    molecules = {
        "H₂O (Water)": 104.5,
        "NH₃ (Ammonia)": 107.3,
        "CH₄ (Methane)": 109.5,
        "CO₂ (Carbon Dioxide)": 180.0,
        "H₂S (Hydrogen Sulfide)": 92.1,
    }
    
    all_passed = True
    
    for name, angle in molecules.items():
        phase_rad = deg_to_rad(angle)
        
        # Verifica conversione
        back_to_deg = phase_rad * 180.0 / np.pi
        error = abs(back_to_deg - angle)
        passed = error < 1e-10
        
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {angle}° → {phase_rad:.6f} rad → {back_to_deg:.6f}° (error: {error:.2e})")
        
        all_passed = all_passed and passed
    
    return all_passed


def run_all_tests():
    """Esegue tutti i test"""
    print("\n" + "█"*70)
    print("█" + " "*20 + "PRECISION TEST SUITE" + " "*20 + "█")
    print("█"*70)
    
    results = {
        "Phase Cancellation": test_phase_cancellation(),
        "Beat Frequency": test_beat_frequency(),
        "Phase Angles": test_phase_angles(),
        "Golden Angle": test_golden_angle(),
        "Frequency Precision": test_frequency_precision(),
        "Stereo Independence": test_stereo_independence(),
        "Molecular Phase": test_molecular_phase(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Maximum precision achieved.")
    else:
        print("⚠️  SOME TESTS FAILED. Review and fix.")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
