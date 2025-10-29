# EntroPit - TODO List

## Immediate Tasks (This Week)

### High Priority
- [ ] **Test UI parameter controls** - Verify all sliders work correctly with new `generate_thrml` API
- [ ] **Add output/ to README** - Update documentation to mention output directory
- [ ] **Create example dungeons** - Generate a few good examples to include in docs
- [ ] **Test on fresh install** - Verify `pip install -r requirements.txt` works

### Code Quality
- [ ] **Add docstring examples** - Include usage examples in `entropit.py` docstrings
- [ ] **Type hints** - Add complete type hints to all public functions
- [ ] **Error handling** - Add proper error messages for invalid parameters
- [ ] **Unit tests** - Create basic tests for generate functions

## Short Term (Next 2 Weeks)

### Features from README Future List
- [ ] **Interactive constraint painting** - Click to set "must be wall/floor" tiles
  - Add `constraints` parameter to `generate_thrml()`
  - Update UI with paint tool
  - Implement as clamped blocks in THRML
  
- [ ] **Better connectivity** - Auto-retry if disconnected
  - Add `ensure_connected` parameter
  - Implement flood-fill + resample logic
  - Show connectivity in real-time during generation

- [ ] **Export functionality** - Save dungeons to game-ready formats
  - JSON export (universal)
  - Tilemap format for Godot
  - 2D array format for Unity
  - Add export button to UI

### Documentation
- [ ] **API Reference** - Create `docs/API.md` with all functions
- [ ] **Tutorial notebook** - Jupyter notebook walkthrough
- [ ] **Video demo** - Screen recording of UI in action
- [ ] **Blog post** - Write up explaining the THRML advantage

## Medium Term (Next Month)

### Advanced Features
- [ ] **Categorical nodes** - Beyond binary wall/floor
  - Door tiles (require keys)
  - Treasure tiles (loot placement)
  - Enemy spawn points
  - Special room types (boss, shop, puzzle)

- [ ] **Multi-floor dungeons** - 3D generation
  - Staircase placement
  - Vertical connectivity constraints
  - Per-floor energy functions

- [ ] **Path constraints** - Control dungeon layout
  - Minimum path length (spawn → exit)
  - Critical path through dungeon
  - Side area constraints

### Performance
- [ ] **Optimize sampling** - Reduce generation time
  - Profile bottlenecks
  - Smart initialization strategies
  - Adaptive annealing schedules
  
- [ ] **Batch generation** - Generate 100s of dungeons efficiently
  - Parallelize across seeds
  - Cache compiled JAX functions
  - Progress bar for long runs

### Analysis
- [ ] **Better metrics** - More sophisticated quality measures
  - Difficulty estimation
  - Exploration potential
  - Visual variety score
  - Metroidvania progression viability

## Long Term (Future)

### Research Directions
- [ ] **Metroidvania constraints** - Key/lock progression
  - Model reachability with items
  - Generate dungeons with intended progression
  - Balance difficulty curve

- [ ] **Style transfer** - Learn from existing dungeons
  - Train on roguelike levels
  - Extract style parameters
  - Generate in specific styles

- [ ] **Real-time editing** - Live parameter adjustment
  - Continuous resampling
  - Smooth transitions between configs
  - Constraint painting during generation

- [ ] **Multi-objective optimization** - Balance competing goals
  - Difficulty + explorable + aesthetics
  - Pareto-optimal dungeons
  - User preference learning

### Infrastructure
- [ ] **Package for PyPI** - `pip install entropit`
  - Setup.py configuration
  - Version management
  - CI/CD pipeline

- [ ] **Web deployment** - Public demo
  - Deploy Gradio UI to Hugging Face Spaces
  - Add gallery of generated dungeons
  - Community sharing features

- [ ] **Hardware integration** - When Extropic chips available
  - Benchmark on real hardware
  - Optimize for thermodynamic accelerators
  - Real-time generation demo

## Project Organization

### Option A: Keep Current Structure (Recommended for Now)
```
entropit/
├── entropit.py              # Main API
├── entropit_ui.py           # Web UI
├── entropit_quickstart.py   # Simple demo
├── traditional_generators.py
├── benchmark.py
├── output/                  # Generated files
├── README.md
├── ARCHITECTURE.md
└── TODO.md (this file)
```

**Pros**: Simple, flat, easy to navigate  
**Cons**: Will get messy with 20+ files

### Option B: Organized Structure (For Future Growth)
```
entropit/
├── src/
│   └── entropit/
│       ├── __init__.py
│       ├── core.py          # generate_thrml, generate_traditional
│       ├── traditional.py   # Traditional algorithms
│       ├── analysis.py      # Metrics and benchmarking
│       └── ui/
│           └── gradio_app.py
├── examples/
│   ├── quickstart.py
│   ├── demo.py
│   └── notebooks/
├── tests/
│   ├── test_generation.py
│   └── test_analysis.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── tutorials/
├── output/
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

**Pros**: Scalable, professional, testable  
**Cons**: More complex, requires refactoring

### Decision Point
- **Stay with Option A** until we have 10+ Python files
- **Move to Option B** when adding tests, packaging, or scaling up

## Immediate Next Steps (In Priority Order)

1. **Test the new structure** - Run all scripts, verify UI works
2. **Update README** - Mention output/ directory, new API
3. **Create simple tests** - Basic smoke tests for generate functions
4. **Add constraint painting** - Biggest UX improvement, demonstrates THRML power
5. **Write blog post** - Share the project, explain THRML advantages

## Notes

- **THRML Advantage**: Focus on features traditional methods can't do easily
  - Soft constraints (prefer but don't require)
  - Global constraints (connectivity, path length)
  - Multi-objective optimization
  - Real-time constraint editing

- **Benchmark-Driven Development**: Always compare against traditional methods
  - Every new feature → benchmark it
  - Quantify the THRML advantage
  - Build compelling demos

- **Documentation First**: Write docs before implementing complex features
  - Clarifies the API design
  - Easier to get feedback
  - Better end result

---

**Last Updated**: October 29, 2025  
**Current Phase**: Core implementation complete, moving to features  
**Next Milestone**: Interactive constraint painting + export functionality

