"""
EntroPit - Probabilistic Dungeon Generation
============================================

A demonstration of thermodynamic computing for procedural content generation.

Main Functions:
    generate_thrml: Generate dungeons using THRML (Ising model)
    generate_traditional: Generate dungeons using classical algorithms
    analyze_dungeon: Analyze dungeon quality and connectivity
    
Example:
    >>> from entropit import generate_thrml, analyze_dungeon
    >>> dungeons, metadata = generate_thrml(grid_size=16, beta=2.0, seed=42)
    >>> metrics = analyze_dungeon(dungeons[0])
    >>> print(f"Connected: {metrics['is_connected']}")
"""

__version__ = "0.1.0"
__author__ = "EntroPit Team"
__license__ = "MIT"

from entropit.core import generate_thrml, analyze_dungeon
from entropit.traditional import generate_traditional
from entropit.analysis import (
    check_connectivity,
    calculate_playability_score,
    benchmark_generator,
)

__all__ = [
    "generate_thrml",
    "generate_traditional",
    "analyze_dungeon",
    "check_connectivity",
    "calculate_playability_score",
    "benchmark_generator",
]

