#!/usr/bin/env python3
"""
EntroPit Interactive UI
=======================

Launch the Gradio web interface for interactive dungeon generation.

Features:
- Generate dungeons with different algorithms
- Adjust THRML parameters in real-time
- View connectivity and quality metrics
- Side-by-side comparisons

Run:
    python examples/interactive_ui.py
    
Then open http://localhost:7860 in your browser.
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropit.ui import launch_ui


if __name__ == "__main__":
    print("=" * 60)
    print("    EntroPit - Interactive Web UI")
    print("=" * 60)
    print()
    print("Starting Gradio interface...")
    print("Once launched, open http://localhost:7860 in your browser")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    launch_ui(share=False)

