"""
EntroPit - Traditional Dungeon Generators
==========================================

Baseline algorithmic approaches for comparison with THRML-based generation.

This module implements classic procedural generation algorithms:
- Random: Baseline (purely random floor/wall placement)
- Cellular Automata: Game of Life-style organic caves
- BSP: Binary Space Partitioning with rooms and corridors
- Drunkard's Walk: Random walk corridor carving

These serve as benchmarks to demonstrate THRML's advantages.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import random
import time


def generate_traditional(
    method: str = "cellular_automata",
    grid_size: int = 24,
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Generate dungeon using traditional algorithmic methods.
    
    Args:
        method: One of ["random", "cellular_automata", "bsp", "drunkards_walk"]
        grid_size: Width and height of dungeon grid
        seed: Random seed for reproducibility (None = random)
        verbose: Print generation progress
        **kwargs: Method-specific parameters:
            - random: floor_probability (default: 0.5)
            - cellular_automata: initial_floor_prob, iterations, birth_limit, death_limit
            - bsp: min_room_size, max_room_size
            - drunkards_walk: floor_percentage
        
    Returns:
        dungeon: Boolean array [grid_size, grid_size] where True = floor
        metadata: Dict with generation info (method, time, seed, etc.)
        
    Example:
        >>> dungeon, meta = generate_traditional("cellular_automata", grid_size=24, seed=42)
        >>> print(f"Generated in {meta['generation_time']*1000:.1f}ms")
        
    Raises:
        ValueError: If method is not recognized
    """
    if seed is None:
        import random
        seed = random.randint(0, 999999)
    
    start_time = time.time()
    
    method = method.lower().replace(" ", "_")
    
    if verbose:
        print(f"[*] Generating {grid_size}x{grid_size} dungeon with {method}")
        print(f"    Seed: {seed}")
    
    # Map to generator functions
    generators = {
        "random": _random_dungeon,
        "cellular_automata": _cellular_automata_dungeon,
        "bsp": _bsp_dungeon,
        "drunkards_walk": _drunkards_walk_dungeon,
    }
    
    if method not in generators:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(generators.keys())}")
    
    gen_func = generators[method]
    
    # Call generator (most take width, height, seed as first args)
    if method == "random":
        dungeon = gen_func(grid_size, grid_size, kwargs.get('floor_probability', 0.5), seed)
    else:
        dungeon = gen_func(grid_size, grid_size, seed=seed, **kwargs)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"[+] Generated dungeon in {elapsed*1000:.2f}ms")
    
    # Build metadata
    metadata = {
        'method': method,
        'grid_size': grid_size,
        'generation_time': elapsed,
        'seed': seed,
        'parameters': kwargs
    }
    
    return dungeon, metadata


def _random_dungeon(width: int, height: int, floor_probability: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """
    Baseline: purely random floor/wall placement.
    
    Args:
        width: Grid width
        height: Grid height
        floor_probability: Probability each tile is a floor (0-1)
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        np.random.seed(seed)
    
    dungeon = np.random.random((height, width)) < floor_probability
    
    # Force edges to be walls
    dungeon[0, :] = False
    dungeon[-1, :] = False
    dungeon[:, 0] = False
    dungeon[:, -1] = False
    
    return dungeon


def _cellular_automata_dungeon(width: int, height: int, 
                               initial_floor_prob: float = 0.45,
                               iterations: int = 5,
                               birth_limit: int = 4,
                               death_limit: int = 3,
                               seed: Optional[int] = None) -> np.ndarray:
    """
    Cellular automata (similar to Conway's Game of Life).
    
    Classic roguelike technique - creates organic cave-like structures.
    
    Args:
        width: Grid width
        height: Grid height
        initial_floor_prob: Initial random floor density
        iterations: Number of CA steps
        birth_limit: Neighbors needed for wall->floor
        death_limit: Neighbors needed to stay floor
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize randomly
    dungeon = np.random.random((height, width)) < initial_floor_prob
    
    # Force edges to be walls
    dungeon[0, :] = False
    dungeon[-1, :] = False
    dungeon[:, 0] = False
    dungeon[:, -1] = False
    
    # Run cellular automata iterations
    for _ in range(iterations):
        new_dungeon = dungeon.copy()
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Count floor neighbors (8-connected)
                neighbors = np.sum(dungeon[i-1:i+2, j-1:j+2]) - dungeon[i, j]
                
                if dungeon[i, j]:  # Currently floor
                    # Stay floor if enough neighbors are floor
                    new_dungeon[i, j] = neighbors >= death_limit
                else:  # Currently wall
                    # Become floor if enough neighbors are floor
                    new_dungeon[i, j] = neighbors >= birth_limit
        
        dungeon = new_dungeon
    
    return dungeon


def _bsp_dungeon(width: int, height: int, 
                min_room_size: int = 4,
                max_room_size: int = 10,
                seed: Optional[int] = None) -> np.ndarray:
    """
    Binary Space Partitioning with rooms and corridors.
    
    Classic algorithm used in Nethack, ADOM, etc.
    Guaranteed connectivity but very rigid/geometric.
    
    Args:
        width: Grid width
        height: Grid height
        min_room_size: Minimum room dimension
        max_room_size: Maximum room dimension
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        random.seed(seed)
    
    dungeon = np.zeros((height, width), dtype=bool)
    
    # Simple BSP implementation
    def split_region(x, y, w, h, depth=0):
        """Recursively split regions and add rooms"""
        if depth > 3 or w < min_room_size * 2 or h < min_room_size * 2:
            # Create a room in this region
            max_room_w = min(max_room_size, w - 2)
            max_room_h = min(max_room_size, h - 2)
            
            if max_room_w < min_room_size or max_room_h < min_room_size:
                return None
            
            room_w = random.randint(min_room_size, max_room_w)
            room_h = random.randint(min_room_size, max_room_h)
            
            max_x_offset = max(0, w - room_w - 1)
            max_y_offset = max(0, h - room_h - 1)
            
            if max_x_offset == 0 or max_y_offset == 0:
                return None
                
            room_x = x + random.randint(1, max_x_offset)
            room_y = y + random.randint(1, max_y_offset)
            
            # Carve room
            dungeon[room_y:room_y + room_h, room_x:room_x + room_w] = True
            return (room_x + room_w // 2, room_y + room_h // 2)  # Center point
        
        # Split either horizontally or vertically
        if random.random() < 0.5:  # Vertical split
            split_x = x + random.randint(min_room_size, w - min_room_size)
            center1 = split_region(x, y, split_x - x, h, depth + 1)
            center2 = split_region(split_x, y, w - (split_x - x), h, depth + 1)
            
            # Connect with horizontal corridor
            if center1 and center2:
                y_corridor = random.randint(y + 1, y + h - 2)
                for cx in range(min(center1[0], center2[0]), max(center1[0], center2[0]) + 1):
                    dungeon[y_corridor, cx] = True
        else:  # Horizontal split
            split_y = y + random.randint(min_room_size, h - min_room_size)
            center1 = split_region(x, y, w, split_y - y, depth + 1)
            center2 = split_region(x, split_y, w, h - (split_y - y), depth + 1)
            
            # Connect with vertical corridor
            if center1 and center2:
                x_corridor = random.randint(x + 1, x + w - 2)
                for cy in range(min(center1[1], center2[1]), max(center1[1], center2[1]) + 1):
                    dungeon[cy, x_corridor] = True
        
        return center1 if center1 else center2
    
    split_region(0, 0, width, height)
    
    return dungeon


def _drunkards_walk_dungeon(width: int, height: int,
                            floor_percentage: float = 0.4,
                            seed: Optional[int] = None) -> np.ndarray:
    """
    Drunkard's walk algorithm.
    
    Start at center, randomly walk and carve floors until target percentage reached.
    Creates meandering organic corridors.
    
    Args:
        width: Grid width
        height: Grid height
        floor_percentage: Target floor coverage (0-1)
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        random.seed(seed)
    
    dungeon = np.zeros((height, width), dtype=bool)
    
    # Start at center
    x, y = width // 2, height // 2
    
    target_floors = int(width * height * floor_percentage)
    current_floors = 0
    
    # Random walk
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while current_floors < target_floors:
        # Carve current position
        if not dungeon[y, x]:
            dungeon[y, x] = True
            current_floors += 1
        
        # Random step
        dx, dy = random.choice(directions)
        new_x = max(1, min(width - 2, x + dx))
        new_y = max(1, min(height - 2, y + dy))
        
        x, y = new_x, new_y
    
    return dungeon

