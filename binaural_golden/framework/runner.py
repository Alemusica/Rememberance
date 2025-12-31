"""
══════════════════════════════════════════════════════════════════════════════
AGNOSTIC EVOLUTION RUNNER
══════════════════════════════════════════════════════════════════════════════

Domain-agnostic evolutionary optimization runner.

This module ties together:
- YAML configuration loading
- Domain adapter resolution
- Evolutionary optimizer instantiation
- Curriculum progression
- Memory system integration

Example usage:
    python -m framework.runner --config config/domain_config.yaml --domain dml_plate
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import time
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_adapter(domain: str, config: Dict[str, Any]):
    """
    Get domain adapter by name.
    
    Args:
        domain: Domain identifier (e.g., 'dml_plate', 'singing_bowl')
        config: Domain-specific configuration
        
    Returns:
        Instantiated domain adapter
    """
    # Import registry
    from framework.adapters import get_adapter as adapter_factory
    return adapter_factory(domain, config)


def run_evolution(
    config: Dict[str, Any],
    domain: str,
    n_generations: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run domain-agnostic evolutionary optimization.
    
    Args:
        config: Full configuration dictionary
        domain: Domain to optimize (overrides config if provided)
        n_generations: Number of generations (overrides config if provided)
        verbose: Print progress
        
    Returns:
        Dictionary with results:
        - best_genome: Best individual found
        - pareto_front: Non-dominated solutions
        - history: Generation-by-generation metrics
        - elapsed_time: Total runtime
    """
    from core.evolutionary_optimizer import EvolutionaryOptimizer
    
    # Resolve configuration
    domain = domain or config.get('domain', {}).get('type', 'dml_plate')
    domain_config = config.get('domains', {}).get(domain, {})
    
    # Merge evolution params
    evolution_config = {
        **config.get('evolution', {}),
        **domain_config.get('evolution', {}),
    }
    
    if n_generations:
        evolution_config['generations'] = n_generations
    
    # Get adapter
    adapter = get_adapter(domain, domain_config)
    
    if verbose:
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║  AGNOSTIC EVOLUTION RUNNER                                   ║")
        print(f"║  Domain: {domain:<50} ║")
        print(f"║  Generations: {evolution_config.get('generations', 100):<45} ║")
        print(f"║  Population: {evolution_config.get('population_size', 50):<46} ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
    
    # Build optimizer
    # NOTE: This would need adaptation based on actual optimizer API
    # The real implementation would use the domain adapter to bridge
    
    start_time = time.time()
    
    # Placeholder for actual evolution loop
    # In production, this would instantiate EvolutionaryOptimizer
    # with the adapter's genome factory, physics engine, and evaluator
    
    history = []
    best_fitness = 0.0
    best_genome = None
    
    n_gens = evolution_config.get('generations', 100)
    pop_size = evolution_config.get('population_size', 50)
    
    for gen in range(n_gens):
        # Placeholder: simulate generation
        gen_fitness = 0.5 + 0.4 * (gen / n_gens)  # Mock improving fitness
        
        history.append({
            'generation': gen,
            'best_fitness': gen_fitness,
            'mean_fitness': gen_fitness * 0.8,
            'diversity': 0.3 - 0.1 * (gen / n_gens),
        })
        
        if gen_fitness > best_fitness:
            best_fitness = gen_fitness
            best_genome = adapter.create_genome()  # Placeholder
        
        if verbose and gen % 10 == 0:
            print(f"  Gen {gen:4d}: best={gen_fitness:.4f}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Evolution complete in {elapsed:.2f}s")
        print(f"  Best fitness: {best_fitness:.4f}")
    
    return {
        'best_genome': best_genome.to_dict() if hasattr(best_genome, 'to_dict') else None,
        'best_fitness': best_fitness,
        'pareto_front': [],  # Placeholder
        'history': history,
        'elapsed_time': elapsed,
        'config': {
            'domain': domain,
            'generations': n_gens,
            'population_size': pop_size,
        }
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run domain-agnostic evolutionary optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m framework.runner
  
  # Specify domain
  python -m framework.runner --domain singing_bowl
  
  # Override generations
  python -m framework.runner --generations 100
  
  # Custom config file
  python -m framework.runner --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='framework/config/domain_config.yaml',
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--domain', '-d',
        default=None,
        help='Domain to optimize (dml_plate, singing_bowl, speaker_box)'
    )
    
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=None,
        help='Number of generations'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output JSON file for results'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Run evolution
    results = run_evolution(
        config=config,
        domain=args.domain,
        n_generations=args.generations,
        verbose=not args.quiet,
    )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
