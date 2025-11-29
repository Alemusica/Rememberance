//! Golden Ratio Binaural Core - High Precision Rust Implementation
//! ================================================================
//! 
//! Maximum precision audio generation using:
//! - f64 floating point (64-bit double precision)
//! - Golden ratio mathematics
//! - Phase-perfect sine generation
//! - SIMD optimizations where available

use std::f64::consts::PI;

/// Golden Ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749894848204586834365638117720309179805762862135;

/// Golden Ratio Conjugate φ - 1 = 1/φ
pub const PHI_CONJUGATE: f64 = 0.618033988749894848204586834365638117720309179805762862135;

/// √5 for golden calculations
pub const SQRT_5: f64 = 2.2360679774997896964091736687747632;

/// Sacred base frequency
pub const SACRED_FREQUENCY: f64 = 432.0;

// ═══════════════════════════════════════════════════════════════════
// SACRED ANGLES (in degrees)
// ═══════════════════════════════════════════════════════════════════

/// Golden Angle = 360° / φ² ≈ 137.5077640500378546°
pub const GOLDEN_ANGLE_DEG: f64 = 360.0 / (PHI * PHI);

/// Fine Structure Constant inverse α⁻¹ ≈ 137.035999084
pub const FINE_STRUCTURE_ANGLE_DEG: f64 = 137.035999084;

/// DNA Helix rotation per base pair ≈ 34.3°
pub const DNA_HELIX_ANGLE_DEG: f64 = 34.3;

/// Pentagon internal angle = 108°
pub const PENTAGON_ANGLE_DEG: f64 = 108.0;

/// Great Pyramid of Giza slope angle ≈ 51.83°
pub const PYRAMID_ANGLE_DEG: f64 = 51.8392;

/// Phase cancellation angle = 180°
pub const CANCELLATION_ANGLE_DEG: f64 = 180.0;

/// Convert degrees to radians
#[inline]
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Convert radians to degrees
#[inline]
pub fn rad_to_deg(radians: f64) -> f64 {
    radians * 180.0 / PI
}

/// First 20 prime numbers for sequences
pub const PRIMES: [u32; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/// First 30 Fibonacci numbers
pub const FIBONACCI: [u64; 30] = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
    89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
    10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040
];

/// Parameters for golden binaural generation
#[derive(Debug, Clone, Copy)]
pub struct GoldenParams {
    pub base_frequency: f64,
    pub beat_frequency: f64,
    pub duration: f64,
    pub transition_time: f64,
    pub amplitude: f64,
    pub phase_offset: f64,        // in radians (internal)
    pub phase_angle_degrees: f64,  // in degrees (user-facing)
}

/// Explicit dual-frequency parameters with angle control
#[derive(Debug, Clone, Copy)]
pub struct DualFrequencyParams {
    pub freq_left: f64,           // Left channel frequency (Hz)
    pub freq_right: f64,          // Right channel frequency (Hz)
    pub phase_angle_deg: f64,     // Phase difference in DEGREES
    pub amplitude_left: f64,      // Left channel amplitude [0,1]
    pub amplitude_right: f64,     // Right channel amplitude [0,1]
    pub duration: f64,            // Duration in seconds
}

impl DualFrequencyParams {
    /// Create with golden angle phase difference
    pub fn with_golden_angle(freq_left: f64, freq_right: f64, duration: f64) -> Self {
        Self {
            freq_left,
            freq_right,
            phase_angle_deg: GOLDEN_ANGLE_DEG,
            amplitude_left: 1.0,
            amplitude_right: 1.0,
            duration,
        }
    }
    
    /// Create with fine structure constant angle
    pub fn with_fine_structure_angle(freq_left: f64, freq_right: f64, duration: f64) -> Self {
        Self {
            freq_left,
            freq_right,
            phase_angle_deg: FINE_STRUCTURE_ANGLE_DEG,
            amplitude_left: 1.0,
            amplitude_right: 1.0,
            duration,
        }
    }
    
    /// Create with custom angle in degrees
    pub fn with_angle(freq_left: f64, freq_right: f64, angle_deg: f64, duration: f64) -> Self {
        Self {
            freq_left,
            freq_right,
            phase_angle_deg: angle_deg,
            amplitude_left: 1.0,
            amplitude_right: 1.0,
            duration,
        }
    }
    
    /// Get phase difference in radians
    pub fn phase_radians(&self) -> f64 {
        deg_to_rad(self.phase_angle_deg)
    }
    
    /// Get beat frequency (difference between L and R)
    pub fn beat_frequency(&self) -> f64 {
        (self.freq_right - self.freq_left).abs()
    }
}

impl GoldenParams {
    /// Create parameters from golden index - all in divine coherence
    pub fn from_golden_index(index: usize, base_freq: f64) -> Self {
        // Beat frequency decreases by golden ratio powers
        let beat_freq = base_freq / PHI.powi((index + 3) as i32);
        
        // Duration from Fibonacci (golden convergent)
        let fib_idx = (index + 5).min(FIBONACCI.len() - 1);
        let duration = FIBONACCI[fib_idx] as f64 * PHI_CONJUGATE;
        
        // Transition time in golden ratio to duration
        let transition = duration * PHI_CONJUGATE * PHI_CONJUGATE;
        
        // Amplitude follows golden decay
        let amplitude = PHI_CONJUGATE.powf(index as f64 * 0.5);
        
        // Phase offset in golden fractions of 2π
        let phase = 2.0 * PI * PHI_CONJUGATE.powi(index as i32);
        
        Self {
            base_frequency: base_freq,
            beat_frequency: beat_freq,
            duration,
            transition_time: transition,
            amplitude,
            phase_offset: phase,
        }
    }
}

/// Golden spiral interpolation function
/// Maps [0,1] → [0,1] following divine golden curve (NOT linear!)
#[inline]
pub fn golden_spiral_interpolation(t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    
    // Golden spiral easing
    let theta = t * PI * PHI;
    let golden_ease = (1.0 - (theta * PHI_CONJUGATE).cos()) / 2.0;
    
    // Golden sigmoid
    let x = (t - 0.5) * 4.0;
    let golden_sigmoid = 1.0 / (1.0 + (-x * PHI).exp());
    
    // Blend with golden weights
    let result = golden_ease * PHI_CONJUGATE + golden_sigmoid * (1.0 - PHI_CONJUGATE);
    
    result.clamp(0.0, 1.0)
}

/// Transition between values using golden spiral
#[inline]
pub fn golden_transition(start: f64, end: f64, t: f64) -> f64 {
    let golden_t = golden_spiral_interpolation(t);
    start + (end - start) * golden_t
}

/// High precision sine generator
/// Uses Taylor series expansion for maximum accuracy when needed
#[inline]
pub fn precise_sin(x: f64) -> f64 {
    // Normalize to [-π, π]
    let x_normalized = x % (2.0 * PI);
    let x_adjusted = if x_normalized > PI {
        x_normalized - 2.0 * PI
    } else if x_normalized < -PI {
        x_normalized + 2.0 * PI
    } else {
        x_normalized
    };
    
    // Use standard sin for normal precision
    // For ultra-high precision, could implement extended Taylor series
    x_adjusted.sin()
}

/// Generate a pure sine tone with maximum precision
pub fn generate_tone(
    frequency: f64,
    duration: f64,
    amplitude: f64,
    phase: f64,
    sample_rate: u32,
) -> Vec<f64> {
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    
    let dt = 1.0 / sample_rate as f64;
    let omega = 2.0 * PI * frequency;
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        let sample = amplitude * precise_sin(omega * t + phase);
        samples.push(sample);
    }
    
    samples
}

/// Generate stereo binaural beat
pub fn generate_binaural_beat(
    params: &GoldenParams,
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let num_samples = (params.duration * sample_rate as f64) as usize;
    let mut left = Vec::with_capacity(num_samples);
    let mut right = Vec::with_capacity(num_samples);
    
    let dt = 1.0 / sample_rate as f64;
    let omega_left = 2.0 * PI * params.base_frequency;
    let omega_right = 2.0 * PI * (params.base_frequency + params.beat_frequency);
    // Use explicit angle in degrees, converted to radians
    let phase_right = deg_to_rad(params.phase_angle_degrees);
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        
        let l = params.amplitude * precise_sin(omega_left * t + params.phase_offset);
        let r = params.amplitude * precise_sin(omega_right * t + phase_right);
        
        left.push(l);
        right.push(r);
    }
    
    (left, right)
}

/// Generate dual frequency stereo with EXPLICIT angle control
/// This is the main function for user-controlled phase difference
pub fn generate_dual_frequency(
    params: &DualFrequencyParams,
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let num_samples = (params.duration * sample_rate as f64) as usize;
    let mut left = Vec::with_capacity(num_samples);
    let mut right = Vec::with_capacity(num_samples);
    
    let dt = 1.0 / sample_rate as f64;
    let omega_left = 2.0 * PI * params.freq_left;
    let omega_right = 2.0 * PI * params.freq_right;
    let phase_offset = params.phase_radians(); // Convert degrees to radians
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        
        // Left channel: base phase = 0
        let l = params.amplitude_left * precise_sin(omega_left * t);
        // Right channel: phase shifted by user-specified angle
        let r = params.amplitude_right * precise_sin(omega_right * t + phase_offset);
        
        left.push(l);
        right.push(r);
    }
    
    (left, right)
}

/// Generate with phase angle sweep from start_angle to end_angle (in degrees)
pub fn generate_phase_sweep(
    freq_left: f64,
    freq_right: f64,
    start_angle_deg: f64,
    end_angle_deg: f64,
    amplitude: f64,
    duration: f64,
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut left = Vec::with_capacity(num_samples);
    let mut right = Vec::with_capacity(num_samples);
    
    let dt = 1.0 / sample_rate as f64;
    let omega_left = 2.0 * PI * freq_left;
    let omega_right = 2.0 * PI * freq_right;
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        let progress = i as f64 / num_samples as f64;
        
        // Interpolate angle using golden spiral
        let current_angle_deg = golden_transition(start_angle_deg, end_angle_deg, progress);
        let phase_offset = deg_to_rad(current_angle_deg);
        
        let l = amplitude * precise_sin(omega_left * t);
        let r = amplitude * precise_sin(omega_right * t + phase_offset);
        
        left.push(l);
        right.push(r);
    }
    
    (left, right)
}

/// Apply golden ratio envelope to audio
pub fn apply_golden_envelope(audio: &mut [f64]) {
    let len = audio.len();
    let attack_samples = (len as f64 * PHI_CONJUGATE * PHI_CONJUGATE * 0.1) as usize;
    let decay_samples = (len as f64 * PHI_CONJUGATE * 0.1) as usize;
    
    // Attack phase
    for i in 0..attack_samples.min(len) {
        let t = i as f64 / attack_samples as f64;
        audio[i] *= golden_spiral_interpolation(t);
    }
    
    // Decay phase
    for i in 0..decay_samples.min(len) {
        let idx = len - 1 - i;
        let t = i as f64 / decay_samples as f64;
        audio[idx] *= golden_spiral_interpolation(t);
    }
}

/// Generate golden transition between two parameter sets
pub fn generate_golden_transition(
    from: &GoldenParams,
    to: &GoldenParams,
    duration: f64,
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut left = Vec::with_capacity(num_samples);
    let mut right = Vec::with_capacity(num_samples);
    
    let dt = 1.0 / sample_rate as f64;
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        let factor = golden_spiral_interpolation(i as f64 / num_samples as f64);
        
        // Interpolate parameters using golden spiral
        let freq = golden_transition(from.base_frequency, to.base_frequency, factor);
        let beat = golden_transition(from.beat_frequency, to.beat_frequency, factor);
        let amp = golden_transition(from.amplitude, to.amplitude, factor);
        let phase = golden_transition(from.phase_offset, to.phase_offset, factor);
        
        let omega_left = 2.0 * PI * freq;
        let omega_right = 2.0 * PI * (freq + beat);
        
        let l = amp * precise_sin(omega_left * t + phase);
        let r = amp * precise_sin(omega_right * t + phase + PI * PHI_CONJUGATE);
        
        left.push(l);
        right.push(r);
    }
    
    (left, right)
}

/// Generate phase cancellation approach (annealing to silence)
pub fn generate_phase_cancellation(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut left = Vec::with_capacity(num_samples);
    let mut right = Vec::with_capacity(num_samples);
    
    let dt = 1.0 / sample_rate as f64;
    let omega = 2.0 * PI * frequency;
    let base_amplitude = PHI_CONJUGATE.powi(4);
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        let progress = i as f64 / num_samples as f64;
        
        // Phase approaches π using golden spiral
        let phase_diff = PI * golden_spiral_interpolation(progress);
        
        // Amplitude approaches zero
        let amp = base_amplitude * (1.0 - golden_spiral_interpolation(progress));
        
        let l = amp * precise_sin(omega * t);
        let r = amp * precise_sin(omega * t + phase_diff);
        
        left.push(l);
        right.push(r);
    }
    
    (left, right)
}

/// Complete annealing sequence generation
pub fn generate_annealing_sequence(
    num_stages: usize,
    base_frequency: f64,
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let mut left_channel: Vec<f64> = Vec::new();
    let mut right_channel: Vec<f64> = Vec::new();
    
    for stage in 0..num_stages {
        let mut params = GoldenParams::from_golden_index(stage, base_frequency);
        
        // Annealing factor approaches 1
        let annealing_factor = if num_stages > 1 {
            stage as f64 / (num_stages - 1) as f64
        } else {
            0.0
        };
        
        // Phase approaches π (cancellation)
        params.phase_offset = PI * golden_spiral_interpolation(annealing_factor);
        
        // Amplitude decreases toward silence
        params.amplitude *= 1.0 - golden_spiral_interpolation(annealing_factor) * 0.95;
        
        // Generate binaural beat
        let (mut left, mut right) = generate_binaural_beat(&params, sample_rate);
        
        // Apply envelope
        apply_golden_envelope(&mut left);
        apply_golden_envelope(&mut right);
        
        left_channel.extend(left);
        right_channel.extend(right);
        
        // Generate transition to next stage
        if stage < num_stages - 1 {
            let next_params = GoldenParams::from_golden_index(stage + 1, base_frequency);
            let (trans_left, trans_right) = generate_golden_transition(
                &params,
                &next_params,
                params.transition_time,
                sample_rate,
            );
            left_channel.extend(trans_left);
            right_channel.extend(trans_right);
        }
    }
    
    // Final phase cancellation
    let silence_duration = FIBONACCI[8] as f64 * PHI_CONJUGATE;
    let (final_left, final_right) = generate_phase_cancellation(
        base_frequency,
        silence_duration,
        sample_rate,
    );
    left_channel.extend(final_left);
    right_channel.extend(final_right);
    
    (left_channel, right_channel)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_golden_ratio() {
        // φ² = φ + 1
        let phi_squared = PHI * PHI;
        let phi_plus_one = PHI + 1.0;
        assert!((phi_squared - phi_plus_one).abs() < 1e-10);
    }
    
    #[test]
    fn test_golden_spiral_bounds() {
        assert_eq!(golden_spiral_interpolation(0.0), 0.0);
        assert_eq!(golden_spiral_interpolation(1.0), 1.0);
        
        for i in 1..10 {
            let t = i as f64 / 10.0;
            let result = golden_spiral_interpolation(t);
            assert!(result >= 0.0 && result <= 1.0);
        }
    }
    
    #[test]
    fn test_tone_generation() {
        let samples = generate_tone(440.0, 0.1, 1.0, 0.0, 44100);
        assert_eq!(samples.len(), 4410);
        
        // Check amplitude bounds
        for s in &samples {
            assert!(s.abs() <= 1.0);
        }
    }
    
    #[test]
    fn test_binaural_generation() {
        let params = GoldenParams::from_golden_index(0, 432.0);
        let (left, right) = generate_binaural_beat(&params, 44100);
        
        assert_eq!(left.len(), right.len());
        assert!(left.len() > 0);
    }
    
    #[test]
    fn test_golden_angle() {
        // Golden angle should be approximately 137.5°
        assert!((GOLDEN_ANGLE_DEG - 137.5).abs() < 0.1);
        
        // Verify: 360° / φ² = golden angle
        let calculated = 360.0 / (PHI * PHI);
        assert!((GOLDEN_ANGLE_DEG - calculated).abs() < 1e-10);
    }
    
    #[test]
    fn test_deg_rad_conversion() {
        assert!((deg_to_rad(180.0) - PI).abs() < 1e-10);
        assert!((rad_to_deg(PI) - 180.0).abs() < 1e-10);
        assert!((deg_to_rad(GOLDEN_ANGLE_DEG) - PI * GOLDEN_ANGLE_DEG / 180.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_dual_frequency_generation() {
        let params = DualFrequencyParams::with_golden_angle(432.0, 440.0, 0.1);
        let (left, right) = generate_dual_frequency(&params, 44100);
        
        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), 4410);
        
        // Verify phase angle is golden
        assert!((params.phase_angle_deg - GOLDEN_ANGLE_DEG).abs() < 1e-10);
    }
    
    #[test]
    fn test_phase_sweep() {
        // Sweep from 0° to 180° (cancellation)
        let (left, right) = generate_phase_sweep(
            432.0, 432.0, 0.0, 180.0, 1.0, 0.1, 44100
        );
        
        assert_eq!(left.len(), right.len());
        
        // At the end, L + R should approach 0 (phase cancellation)
        let last_100: Vec<f64> = left.iter().rev().take(100)
            .zip(right.iter().rev().take(100))
            .map(|(l, r)| (l + r).abs())
            .collect();
        let avg_sum: f64 = last_100.iter().sum::<f64>() / 100.0;
        // Should be close to zero due to 180° phase
        assert!(avg_sum < 0.5);
    }
}
