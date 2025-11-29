//! Python bindings for Golden Binaural Core
//! Exposes high-precision Rust functions to Python via PyO3

use pyo3::prelude::*;
use pyo3::types::PyList;

mod lib;
use lib::*;

/// Generate complete annealing sequence - returns (left_channel, right_channel)
#[pyfunction]
fn py_generate_annealing_sequence(
    num_stages: usize,
    base_frequency: f64,
    sample_rate: u32,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(generate_annealing_sequence(num_stages, base_frequency, sample_rate))
}

/// Golden spiral interpolation function
#[pyfunction]
fn py_golden_spiral_interpolation(t: f64) -> PyResult<f64> {
    Ok(golden_spiral_interpolation(t))
}

/// Generate single binaural beat segment
#[pyfunction]
fn py_generate_binaural_beat(
    base_freq: f64,
    beat_freq: f64,
    duration: f64,
    amplitude: f64,
    phase: f64,
    sample_rate: u32,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let params = GoldenParams {
        base_frequency: base_freq,
        beat_frequency: beat_freq,
        duration,
        transition_time: duration * PHI_CONJUGATE,
        amplitude,
        phase_offset: phase,
    };
    Ok(generate_binaural_beat(&params, sample_rate))
}

/// Generate phase cancellation (annealing to silence)
#[pyfunction]
fn py_generate_phase_cancellation(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(generate_phase_cancellation(frequency, duration, sample_rate))
}

/// Get golden ratio constant
#[pyfunction]
fn py_get_phi() -> PyResult<f64> {
    Ok(PHI)
}

/// Get golden ratio conjugate
#[pyfunction]
fn py_get_phi_conjugate() -> PyResult<f64> {
    Ok(PHI_CONJUGATE)
}

/// Python module
#[pymodule]
fn golden_binaural_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_generate_annealing_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_spiral_interpolation, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_binaural_beat, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_phase_cancellation, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_phi, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_phi_conjugate, m)?)?;
    Ok(())
}
