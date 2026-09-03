# Contributing to RL-UCO

Thank you for your interest in contributing! This guide covers setup, testing, and the contribution workflow.

## Development Setup

### Prerequisites

- **Python 3.11+** (3.12/3.13/3.14 also work)
- **Git**
- **LLVM 18+** (optional — only needed for real corpus extraction and pass execution; the test suite and demo work without it)

### Install

```bash
# Clone the repo
git clone https://github.com/FreakyAdy/RL-for-Universal-Compiler-Optimization.git
cd RL-for-Universal-Compiler-Optimization

# Create and activate a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[dev]"
```

### Verify Your Setup

```bash
# Run the test suite (no LLVM required)
pytest tests/ -v

# Run the end-to-end demo (no LLVM required)
python scripts/demo.py
```

Both should complete without errors. If they do, you're ready to contribute.

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=rl_uco --cov-report=term-missing

# Single test file
pytest tests/test_passes.py -v
```

The test suite is designed to run without LLVM installed. Tests that require external tools are skipped automatically.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```bash
# Format code
ruff format rl_uco/ tests/ scripts/

# Check for lint issues
ruff check rl_uco/ tests/ scripts/

# Auto-fix lint issues where possible
ruff check --fix rl_uco/ tests/ scripts/
```

CI will reject PRs that fail `ruff check`. Run it locally before pushing.

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `rl_uco/` | Core Python package |
| `rl_uco/corpus/` | Function extraction and normalization |
| `rl_uco/passes/` | LLVM pass registry, execution, validation |
| `rl_uco/env/` | Compile-run environment and reward computation |
| `rl_uco/hardware/` | Hardware profiling (x86 RAPL, ARM, CUDA) |
| `rl_uco/graph/` | IR → graph tensor conversion |
| `rl_uco/ir/` | LLVM/MLIR adapter interfaces |
| `rl_uco/data/` | Dataset schema, collection, Parquet export |
| `rl_uco/rl/` | Actor-critic models, IQL trainer, inference |
| `rl_uco/eval/` | Evaluation and reporting |
| `tests/` | Unit tests |
| `scripts/` | Demo and utility scripts |
| `infra/` | Docker, deployment drivers |
| `docs/` | Extended documentation |

## Contribution Workflow

1. **Open an issue first** — describe the bug or feature you're working on.
2. **Fork and branch** — create a branch from `main` named `feature/your-feature` or `fix/your-fix`.
3. **Write code** — follow the existing code style. Add docstrings (Google style) for new public APIs.
4. **Add tests** — new functionality should have tests in `tests/`. Aim for tests that don't require LLVM.
5. **Run checks locally:**
   ```bash
   ruff format rl_uco/ tests/
   ruff check rl_uco/ tests/
   pytest tests/ -v
   ```
6. **Submit a PR** — link to the issue, describe what changed and why, include test results.

## What We're Looking For

Good first contributions:
- Adding tests for uncovered modules
- Improving error messages and validation
- Documentation improvements
- New LLVM passes in `registry.yaml`

Larger contributions (open an issue to discuss first):
- New hardware profilers
- GNN encoder improvements
- Online RL training modes
- MLIR pass support expansion

## Commit Messages

Use clear, descriptive commit messages:

```
fix: handle empty pass sequences in validator

The validator returned an error with an empty reason string when given
a zero-length sequence. Now returns a clear message.
```

Prefixes: `fix:`, `feat:`, `docs:`, `test:`, `refactor:`, `ci:`.

## Questions?

Open an issue or start a discussion. We're happy to help.
