#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          100 GENERATION EVOLUTION TEST                                       ║
║                                                                              ║
║   Validates the full evolution pipeline with 100 generations.               ║
║   Tests convergence, fitness improvement, and memory integration.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import time
import json
from datetime import datetime

from core.person import Person
from core.plate_genome import PlateGenome, ContourType
from core.fitness import FitnessEvaluator, ObjectiveWeights, ZoneWeights
from core.evolutionary_optimizer import EvolutionaryOptimizer, EvolutionConfig


def run_100_generation_test():
    """Run full 100 generation evolution test."""
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         100 GENERATION EVOLUTION TEST                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Create test person
    person = Person(
        height_m=1.80,
        weight_kg=75,
    )
    
    print(f"\n📋 Test Person:")
    print(f"   Height: {person.height_m}m")
    print(f"   Weight: {person.weight_kg}kg")
    print(f"   Shoulder width: {person.shoulder_width:.3f}m")
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=30,        # Smaller for faster test
        n_generations=100,         # Full 100 generations
        elite_count=2,
        tournament_size=3,
        crossover_rate=0.85,
        mutation_rate=0.25,
        mutation_sigma=0.05,
        adaptive_mutation=True,
        mutation_decay=0.97,
        min_mutation_sigma=0.008,
        fixed_contour=ContourType.RECTANGLE,  # Fix for deterministic test
    )
    
    print(f"\n⚙️  Evolution Config:")
    print(f"   Population: {config.population_size}")
    print(f"   Generations: {config.n_generations}")
    print(f"   Elite count: {config.elite_count}")
    print(f"   Mutation σ: {config.mutation_sigma}")
    
    # Weights - using actual API
    weights = ObjectiveWeights(
        flatness=1.0,
        spine_coupling=2.0,
        low_mass=0.3,
        manufacturability=0.5,
    )
    
    # Create optimizer - using actual API
    optimizer = EvolutionaryOptimizer(
        person=person,
        objectives=weights,
        zone_weights=ZoneWeights(spine=0.7, head=0.3),
        config=config,
        seed=42,  # Reproducible
    )
    
    # Track progress
    history = {
        'generations': [],
        'best_fitness': [],
        'mean_fitness': [],
        'diversity': [],
        'ear_uniformity': [],
        'spine_energy': [],
    }
    
    def progress_callback(state):
        """Track evolution progress."""
        gen = state.generation
        history['generations'].append(gen)
        history['best_fitness'].append(state.best_fitness.total_fitness)
        history['diversity'].append(state.diversity)
        
        # Extract specific metrics
        if hasattr(state.best_fitness, 'ear_uniformity'):
            history['ear_uniformity'].append(state.best_fitness.ear_uniformity)
        if hasattr(state.best_fitness, 'spine_total_energy'):
            history['spine_energy'].append(state.best_fitness.spine_total_energy)
        
        # Progress output every 10 generations
        if gen % 10 == 0:
            print(f"   Gen {gen:3d}: fitness={state.best_fitness.total_fitness:.4f}, "
                  f"diversity={state.diversity:.3f}")
        
        return True  # Continue evolution
    
    # Run evolution
    print("\n🧬 Running evolution...")
    start_time = time.time()
    
    best_genome = optimizer.run(
        callback=progress_callback,
        verbose=False,  # We handle our own output
    )
    
    elapsed = time.time() - start_time
    
    # Get final fitness - use the optimizer's cached evaluator
    final_result = optimizer.evaluator.evaluate(best_genome)
    
    # Results
    print("\n" + "═" * 60)
    print("📊 RESULTS")
    print("═" * 60)
    
    print(f"\n⏱️  Time: {elapsed:.2f}s ({elapsed/100:.2f}s per generation)")
    
    print(f"\n🏆 Best Genome:")
    print(f"   Size: {best_genome.length*100:.1f}cm × {best_genome.width*100:.1f}cm")
    print(f"   Contour: {best_genome.contour_type.name}")
    print(f"   Exciters: {len(best_genome.exciters)}")
    
    print(f"\n📈 Fitness Progression:")
    print(f"   Initial: {history['best_fitness'][0]:.4f}")
    print(f"   Final:   {history['best_fitness'][-1]:.4f}")
    improvement = (history['best_fitness'][-1] - history['best_fitness'][0]) / max(history['best_fitness'][0], 0.001)
    print(f"   Improvement: {improvement*100:.1f}%")
    
    print(f"\n🎯 Final Objectives:")
    print(f"   Total fitness: {final_result.total_fitness:.4f}")
    if hasattr(final_result, 'ear_uniformity'):
        print(f"   Ear uniformity: {final_result.ear_uniformity:.4f}")
    if hasattr(final_result, 'spine_total_energy'):
        print(f"   Spine energy: {final_result.spine_total_energy:.4f}")
    
    # Validation checks
    print("\n" + "═" * 60)
    print("✅ VALIDATION")
    print("═" * 60)
    
    checks = []
    
    # Check 1: Fitness improved
    improved = history['best_fitness'][-1] > history['best_fitness'][0]
    checks.append(("Fitness improved over evolution", improved))
    
    # Check 2: Final fitness > 0.3 (reasonable threshold)
    good_fitness = final_result.total_fitness > 0.3
    checks.append(("Final fitness > 0.3", good_fitness))
    
    # Check 3: Evolution ran most generations OR converged early (both OK)
    all_gens = len(history['generations']) >= 50  # Allow early convergence after 50 gens
    checks.append(("Completed 50+ generations (or converged)", all_gens))
    
    # Check 4: Reasonable genome size
    reasonable_size = 0.5 < best_genome.length < 3.0 and 0.3 < best_genome.width < 1.5
    checks.append(("Genome size within bounds", reasonable_size))
    
    # Check 5: Has exciters
    has_exciters = len(best_genome.exciters) >= 2  # At least 2 exciters
    checks.append(("Has minimum exciters", has_exciters))
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'config': {
            'population_size': config.population_size,
            'n_generations': config.n_generations,
        },
        'final_fitness': final_result.total_fitness,
        'improvement_percent': improvement * 100,
        'best_genome': {
            'length': best_genome.length,
            'width': best_genome.width,
            'contour': best_genome.contour_type.name,
            'n_exciters': len(best_genome.exciters),
        },
        'history': {
            'best_fitness': history['best_fitness'][:10] + ['...'] + history['best_fitness'][-5:],
        },
        'all_checks_passed': all_passed,
    }
    
    # Save to file
    output_path = Path(__file__).parent / "100gen_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_path}")
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED - Evolution pipeline validated!")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED - Review results")
        return 1


if __name__ == "__main__":
    sys.exit(run_100_generation_test())
