# HGST E7: Frustration Ordering in Non-Abelian Gauge Theory

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b)](https://arxiv.org)
[![DOI](https://zenodo.org/badge/1173352700.svg)](https://doi.org/10.5281/zenodo.18873889)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![Status](https://img.shields.io/badge/Paper-Accepted-green)](https://github.com/boonsup/hgst-e7)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![CI](https://github.com/boonsup/hgst-e7/actions/workflows/ci.yml/badge.svg)](https://github.com/boonsup/hgst-e7/actions/workflows/ci.yml)

**Author:** Boonsup Waikham  
**Affiliation:** College of Computing, Khon Kaen University, Khon Kaen, Thailand  
**ORCID:** [0009-0000-7693-7295](https://orcid.org/0009-0000-7693-7295)

---

## Overview

This repository contains the Python simulation code and LaTeX paper source for two companion papers (both **accepted**, 2026):

> **[Theory]** Hierarchical Graded Structure Theory: Axiomatic Foundations, the Grade-Difference Group, and Non-Abelian Frustration Ordering in Classical Lattice Gauge Extensions E1–E7  
> Boonsup Waikham, College of Computing, Khon Kaen University

> **[Empirical]** Frustration Universality in Hierarchical Graded Structure Theory: A Lattice Study of the E7 MIXED Class from U(1) to the SU(3)×SU(2)×U(1) Gauge Group  
> Boonsup Waikham, College of Computing, Khon Kaen University

We measure the HGST **MIXED frustration fraction** $R$ — the fraction of sign-inconsistent holonomy triads — across four gauge groups on classical 2D lattices, with finite-size scaling (FSS) to the thermodynamic limit.

### Principal results

| Gauge group | $R_\infty$ | Status |
|-------------|------------|--------|
| U(1) | 0 | Falsified |
| SU(2) | $0.3598 \pm 0.0041$ | Validated (5-pt FSS) |
| SU(3) | $0.3539 \pm 0.0195$ | Validated |
| SM C-scalar | $0.4981 \pm 0.0056$ | Validated |
| SM N-scalar | $0.4980 \pm 0.0076$ | Validated |

---

## Repository structure

```
HGST-E7/
├── simulation/          Python physics modules (26 files)
│   ├── su2.py           SU(2) algebra: Haar random, small perturbation
│   ├── su3.py           SU(3) algebra: expm-based
│   ├── u1.py            U(1) — Abelian baseline
│   ├── lattice.py       Lattice2D: sites, edges, plaquettes, BFS
│   ├── fields.py        Link and matter field initialization
│   ├── action.py        Wilson gauge + hopping action, ΔS functions
│   ├── observables.py   R observable via BFS holonomies
│   ├── updates.py       Metropolis sweep (links + matter)
│   ├── simulation.py    SimConfig dataclass, run_point(), FSS utilities
│   ├── su2_l10_colab.py ★ Self-contained Colab-ready SU(2) L=10 script
│   ├── su2_longrun_fss.py  Long-run FSS analysis (L=4–12, seeds 99–102)
│   ├── sm_*.py          Standard Model (SU(3)×SU(2)×U(1)) modules
│   └── ...
│
├── data/                JSON result files (all production runs)
│   ├── su2_l10_summary.json   SU(2) L=10 FSS result
│   ├── su2_L12_b8.json         SU(2) L=12 production run (R=0.3521±0.00089)
│   ├── p1_su3_fss_corrected.json
│   ├── p1_sm_fss_corrected.json
│   └── ...
│
├── arxiv_submit_theory/   LaTeX source — theory paper (Revision 4, ACCEPTED)
│   ├── main.tex           Axiomatic foundations, grade-difference group, E1–E7
│   ├── main.bbl           Pre-generated bibliography (required by arXiv)
│   ├── references.bib
│   └── figures/
│
├── arxiv_submit_empirical/ LaTeX source — empirical paper (Revision 4, ACCEPTED)
│   ├── main.tex           FSS lattice study, U(1)→SM gauge groups
│   ├── main.bbl
│   ├── references.bib
│   └── figures/
│
├── paper/               Working LaTeX drafts and reviewer response
│
├── notebooks/
│   └── vacation_analysis.ipynb
│
├── .env.example         Author/ORCID template → copy to .env (not committed)
├── .gitignore
├── requirements.txt
├── CITATION.cff         Machine-readable citation metadata
└── UPLOAD_PLAN.md       arXiv / Zenodo upload checklist
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt      # numpy + scipy only

# 2. Run SU(2) L=10 FSS (standalone, no local imports)
python simulation/su2_l10_colab.py
# → writes su2_l10_summary.json + su2_l10_runlog.json
# Runtime: ~15 min on a modern laptop (L=10, 500 measurements)
```

### Run in Google Colab

1. Upload `simulation/su2_l10_colab.py` to the Colab session
2. `!python su2_l10_colab.py`
3. Download `su2_l10_summary.json` for results

### Reproduce SU(3) FSS (4-point, L=4,6,8,10)

```bash
cd simulation
python run_simulation.py --gauge su3 --L 4 6 8 10 --beta 8.0 --kappa 0.3
```

### Reproduce SM scan

```bash
cd simulation
python sm_scan.py --L 4 6 8 --beta3 8.0 --beta2 4.0 --kappa 0.3
```

---

All JSON files in `data/` are the actual production outputs used in the paper.  
Simulation seeds are fixed (`SEED=203` for SU(2) L=10; see each script header).  
The paper PDFs can be recompiled separately:

```bash
cd paper
# For Theory:
pdflatex Theoretical_Framework && bibtex Theoretical_Framework && pdflatex Theoretical_Framework
# For Empirical:
pdflatex Empirical_Study && bibtex Empirical_Study && pdflatex Empirical_Study
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{waikham_hgst_e7_2026,
  author    = {Waikham, Boonsup},
  title     = {HGST E7: Simulation Code for Frustration Ordering
               in Non-Abelian Gauge Theory},
  year      = {2026},
  version   = {4.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18873889},
  url       = {https://github.com/boonsup/hgst-e7}
}
```

See also `CITATION.cff` for machine-readable metadata.

---

## License

MIT — see [LICENSE](LICENSE).
