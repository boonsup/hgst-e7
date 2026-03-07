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

## Adding a new empirical experiment

All empirical claims must be reproducible. Every new simulation run requires:

1. A script (or clear diff to an existing script) in `simulation/` with:
   - Fixed seed, N_therm, N_meas, L, β, κ documented at the top
   - No hard-coded absolute paths
2. Output JSON added to `data/` with keys: `R_mean`, `R_err`, `tau_int`, `seed`, `L`, `beta`, `N_meas`
3. FSS table updated in `paper/appendix/tables.tex` if a new L-point is added
4. Tag the Issue with **reproducibility-check** so a second contributor can independently rerun

---

## Adding a new theoretical proof

All theoretical claims follow the notation of `paper/Theoretical_Framework.tex`:
- Grade lattice Γ = ℕ², operations Op1–Op10, field **F***, grade-difference group **G∆**
- Use standard LaTeX environments: `\begin{theorem}`, `\begin{lemma}`, `\begin{proof}`
- No new mathematical symbols without a `\newcommand` in the preamble

**Steps:**
1. Open an Issue using the **Theory / Proof Contribution** template (label: `good-first-proof`)
2. Add the proof to `paper/appendix/proofs.tex`
3. Add any new BibTeX entries to `paper/Theoretical_Framework.bib`
4. Run `pdflatex -halt-on-error Theoretical_Framework` — CI will also check this
5. PR description must include the theorem statement in LaTeX

---

## Unit tests

There is currently no `tests/` directory — **this is a known gap and a `good-first-issue`**.
If you add unit tests:

```bash
# Suggested structure
tests/
  test_su2.py        # Haar measure mean, group closure, det=1
  test_lattice.py    # BFS covers all plaquettes, edge count = 2*L^2
  test_observables.py # R in [0,1], R(U1)=0 for uniform config
```

Tests should use only `numpy`, `scipy`, and `pytest`.  
Add `pytest` to `requirements.txt` and extend the CI `smoke-test` job.

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
