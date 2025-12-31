"""
Framework core - Protocol definitions and base classes.
"""

from framework.core.protocols import (
    # Protocols
    GenomeProtocol,
    ObjectiveResultProtocol,
    MemoryProtocol,
    DistillerProtocol,
    ObserverProtocol,
    
    # Abstract Base Classes
    PhysicsEngineABC,
    FitnessEvaluatorABC,
    GenomeFactoryABC,
    DomainAdapterABC,
)

__all__ = [
    # Protocols
    'GenomeProtocol',
    'ObjectiveResultProtocol',
    'MemoryProtocol',
    'DistillerProtocol',
    'ObserverProtocol',
    
    # ABCs
    'PhysicsEngineABC',
    'FitnessEvaluatorABC',
    'GenomeFactoryABC',
    'DomainAdapterABC',
]
