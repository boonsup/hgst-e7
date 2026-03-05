# Contributing to HGST-E7

Thank you for your interest in contributing. All skill levels are welcome —
physics, computation, biology, or documentation.

---

## Ways to contribute

| Area | Examples |
|------|----------|
| **Simulation** | SU(3) L=10 run, 3D extension, large-N scaling |
| **Frustrated systems** | Map MIXED fraction to Edwards-Anderson / chirality |
| **Biology** | Verify R on YEASTRACT, STRING, ARACNE GRN databases |
| **Theory** | Analytical bounds on R for SU(N), topological charge connection |
| **Code** | GPU port (CuPy/JAX), benchmarks, test coverage |
| **Docs** | Tutorials, worked examples, paper corrections |

---

## Quick start

```bash
git clone https://github.com/boonsup/hgst-e7.git
cd hgst-e7
pip install -r requirements.txt
python simulation/su2_l10_colab.py   # self-contained, runs in ~15 min
```

---

## Workflow

1. **Open an Issue** before starting significant work — describe what you plan
2. **Fork** the repo and work on a branch: `git checkout -b my-feature`
3. Keep commits focused; write a short commit message explaining *why*
4. **Open a Pull Request** against `main` — fill in the PR template
5. All PRs require at least one review (from @boonsup or a designated reviewer)

---

## Code style

- Python 3.10+, type hints encouraged
- `numpy` / `scipy` only — no additional runtime dependencies without discussion
- Physics functions should have a docstring stating the formula they implement
- New simulation runs must include: seed, N_meas, N_therm, output JSON

---

## Credit

Contributors are listed in:
- `CITATION.cff` (authors field, updated on each release)
- Paper acknowledgements section
- Co-authorship on new papers is negotiable for substantial physics contributions

---

## Questions

Open a [GitHub Discussion](https://github.com/boonsup/hgst-e7/discussions)
or email **boonsup@kku.ac.th**.
