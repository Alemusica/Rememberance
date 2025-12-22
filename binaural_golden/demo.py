#!/usr/bin/env python3
"""
Rememberance - Quick Demo
==========================

Demonstrates the modular architecture without requiring full installation.
Tests EMDR and vibroacoustic modules with audio backends.

Usage:
    python demo.py [--backend {pyaudio|alsa|dummy}] [--mode {emdr|vibro}]

Examples:
    python demo.py --backend dummy --mode emdr     # Silent EMDR test
    python demo.py --backend pyaudio --mode vibro  # Audio vibroacoustic
    python demo.py                                  # Auto-detect, EMDR
"""

import sys
import os
import argparse
import time
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.audio_backend import create_audio_backend
from modules.emdr import EMDRGenerator, EMDRJourney, BilateralMode, ANNEALING_PROGRAMS
from modules.vibroacoustic import VibroacousticPanner, VibroacousticProgram


def demo_emdr(backend_type="dummy", duration=10):
    """
    Demo EMDR bilateral stimulation.
    
    Args:
        backend_type: Audio backend (pyaudio/alsa/dummy)
        duration: Demo duration in seconds
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║   EMDR Bilateral Stimulation Demo             ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    # Create backend
    print(f"🔊 Initializing {backend_type} backend...")
    backend = create_audio_backend(
        backend_type=backend_type,
        sample_rate=48000,
        channels=2,
        buffer_size=2048
    )
    
    # Create EMDR generator
    print("🧠 Creating EMDR generator...")
    emdr = EMDRGenerator(sample_rate=48000)
    
    # Configure parameters
    emdr.set_parameters(
        speed=1.0,  # 1 Hz bilateral
        freq=432,   # Sacred 432 Hz
        mode=BilateralMode.GOLDEN_PHASE,  # φ phase offset
        amplitude=0.5
    )
    
    print("\n✓ Configuration:")
    print(f"  Speed: {emdr.bilateral_speed} Hz")
    print(f"  Frequency: {emdr.carrier_freq} Hz")
    print(f"  Mode: {emdr.mode.value}")
    print(f"  Amplitude: {emdr.amplitude}")
    
    # Define callback
    def audio_callback(num_frames):
        return emdr.generate_frame(num_frames)
    
    # Start playback
    print(f"\n▶ Starting EMDR for {duration} seconds...")
    emdr.start()
    backend.start(audio_callback)
    
    # Run for duration
    try:
        for i in range(duration):
            time.sleep(1)
            print(f"  {i+1}/{duration} seconds elapsed...")
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
    
    # Stop
    emdr.stop()
    backend.stop()
    print("\n✓ Demo complete!")


def demo_emdr_journey(backend_type="dummy"):
    """
    Demo EMDR trauma annealing journey.
    
    Args:
        backend_type: Audio backend
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║   EMDR Journey Demo - Gentle Release          ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    # Create backend
    print(f"🔊 Initializing {backend_type} backend...")
    backend = create_audio_backend(
        backend_type=backend_type,
        sample_rate=48000,
        channels=2,
        buffer_size=2048
    )
    
    # Create EMDR generator
    print("🧠 Creating EMDR generator...")
    emdr = EMDRGenerator(sample_rate=48000)
    
    # Create journey
    program = ANNEALING_PROGRAMS["gentle_release"]
    print(f"\n🌅 Journey: {program['name']}")
    print(f"   Duration: {program['duration_min']} minutes")
    print(f"   Description: {program['description']}")
    print(f"   Phases: {len(program['phases'])}")
    
    journey = EMDRJourney(emdr, program)
    
    # Define callback
    def audio_callback(num_frames):
        return emdr.generate_frame(num_frames)
    
    # Start journey
    print(f"\n▶ Starting journey...")
    journey.start()
    backend.start(audio_callback)
    
    # Monitor progress
    try:
        while journey.running:
            journey.update()
            progress = journey.get_progress()
            
            if progress["running"]:
                print(f"  Phase: {progress['phase_name']} "
                      f"({progress['phase_progress']*100:.1f}% complete) "
                      f"| Overall: {progress['progress']*100:.1f}%")
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
        journey.stop()
    
    # Stop
    backend.stop()
    print("\n✓ Journey complete!")


def demo_vibroacoustic(backend_type="dummy", duration=10):
    """
    Demo vibroacoustic panning.
    
    Args:
        backend_type: Audio backend
        duration: Demo duration in seconds
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║   Vibroacoustic Panning Demo                   ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    # Create backend
    print(f"🔊 Initializing {backend_type} backend...")
    backend = create_audio_backend(
        backend_type=backend_type,
        sample_rate=48000,
        channels=2,
        buffer_size=2048
    )
    
    # Create panner
    print("🎵 Creating vibroacoustic panner...")
    panner = VibroacousticPanner(
        sample_rate=48000,
        num_channels=2,
        board_length_mm=1950,
        velocity_ms=5500,
        attenuation_alpha=0.4
    )
    
    # Create program
    program = VibroacousticProgram(
        panner=panner,
        frequency=432,
        modulation_freq=0.1  # 10 second cycle
    )
    
    print("\n✓ Configuration:")
    print(f"  Carrier frequency: 432 Hz")
    print(f"  Modulation: 0.1 Hz (10 sec cycle)")
    print(f"  Board length: 1950 mm")
    print(f"  Sound velocity: 5500 m/s")
    
    # Define callback
    def audio_callback(num_frames):
        return program.generate_frame(num_frames)
    
    # Start playback
    print(f"\n▶ Starting vibroacoustic for {duration} seconds...")
    backend.start(audio_callback)
    
    # Run for duration
    try:
        for i in range(duration):
            time.sleep(1)
            print(f"  {i+1}/{duration} seconds elapsed...")
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
    
    # Stop
    backend.stop()
    print("\n✓ Demo complete!")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description="Rememberance Quick Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --backend dummy --mode emdr
  python demo.py --backend pyaudio --mode vibro
  python demo.py --mode journey
        """
    )
    
    parser.add_argument(
        '--backend',
        choices=['pyaudio', 'alsa', 'dummy'],
        default=None,
        help='Audio backend (default: auto-detect)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['emdr', 'journey', 'vibro'],
        default='emdr',
        help='Demo mode (default: emdr)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Demo duration in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run demo
    try:
        if args.mode == 'emdr':
            demo_emdr(args.backend, args.duration)
        elif args.mode == 'journey':
            demo_emdr_journey(args.backend)
        elif args.mode == 'vibro':
            demo_vibroacoustic(args.backend, args.duration)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
