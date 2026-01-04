"""
Logging Configuration for Golden Studio Evolution System.

Provides centralized logging control with easily toggleable levels:
- DEBUG: Full diagnostic output (every score calculation)
- INFO: Key events only (start/end of operations, final scores)
- WARNING+: Errors and warnings only (production mode)

Usage:
    from core.logging_config import setup_logging, set_debug_mode
    
    # In main app:
    setup_logging(debug=False)  # Production mode
    
    # To enable debug temporarily:
    set_debug_mode(True)
    
    # To disable logging for performance:
    set_logging_level(logging.WARNING)
"""

import logging
import os

# Named loggers for different components
LOGGER_NAMES = [
    # Core
    'golden_studio.evolution',
    'src.core.fitness',
    'src.core.evolutionary_optimizer',
    
    # Scorers
    'src.core.scorers.jab_coherence',
    'src.core.scorers.ear_uniformity',
    'src.core.scorers.spine_coupling',
    'src.core.scorers.flatness',
    'src.core.scorers.exciter',
    'src.core.scorers.manufacturability',
    'src.core.scorers.structural',
]


def setup_logging(debug: bool = False, log_file: str = None):
    """
    Setup logging configuration for Golden Studio.
    
    Args:
        debug: If True, set DEBUG level. Otherwise INFO level.
        log_file: Optional file path to write logs to.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set level for all our loggers
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.setLevel(level)


def set_debug_mode(enabled: bool):
    """
    Enable or disable debug mode for all scorers and evolution.
    
    Args:
        enabled: True for DEBUG level, False for INFO level.
    """
    level = logging.DEBUG if enabled else logging.INFO
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.setLevel(level)


def set_logging_level(level: int):
    """
    Set logging level for all Golden Studio components.
    
    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.setLevel(level)


def disable_scorer_logging():
    """
    Disable scorer logging for maximum performance.
    Sets all scorers to WARNING level (only errors).
    """
    scorer_loggers = [n for n in LOGGER_NAMES if 'scorers' in n]
    for name in scorer_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)


def enable_scorer_logging(debug: bool = False):
    """
    Re-enable scorer logging.
    
    Args:
        debug: If True, enable DEBUG level. Otherwise INFO.
    """
    scorer_loggers = [n for n in LOGGER_NAMES if 'scorers' in n]
    level = logging.DEBUG if debug else logging.INFO
    for name in scorer_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(level)


# Environment variable control
if os.environ.get('GOLDEN_DEBUG', '').lower() in ('1', 'true', 'yes'):
    setup_logging(debug=True)
elif os.environ.get('GOLDEN_QUIET', '').lower() in ('1', 'true', 'yes'):
    setup_logging(debug=False)
    set_logging_level(logging.WARNING)
