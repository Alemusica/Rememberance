"""
Framework Adapters Package

Domain-specific implementations that connect the agnostic evolution
framework to specific design problems.

Available adapters:
- dml_plate: Distributed Mode Loudspeaker plates for vibroacoustic therapy
"""

from typing import Dict, Any

# Registry of available adapters
ADAPTER_REGISTRY = {
    'dml_plate': 'framework.adapters.dml_plate',
    # Future adapters:
    # 'singing_bowl': 'framework.adapters.singing_bowl',
    # 'speaker_box': 'framework.adapters.speaker_box',
}


def get_adapter(domain_name: str, config: Dict[str, Any]):
    """
    Get adapter instance for a domain.
    
    Args:
        domain_name: Name of the domain (e.g., 'dml_plate')
        config: Configuration dict for the domain
    
    Returns:
        Adapter instance
    """
    if domain_name not in ADAPTER_REGISTRY:
        available = ', '.join(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown domain: {domain_name}. Available: {available}")
    
    module_path = ADAPTER_REGISTRY[domain_name]
    module = __import__(module_path, fromlist=['create_adapter'])
    
    return module.create_adapter(config)


def list_domains() -> list:
    """List all available domains."""
    return list(ADAPTER_REGISTRY.keys())
