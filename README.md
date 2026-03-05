# HGST E7: Frustration Ordering in Non-Abelian Gauge Theory

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b)](https://arxiv.org)
[![DOI](https://img.shields.io/badge/DOI-pending-blue)](https://zenodo.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)

**Author:** Boonsup Waikham  
**Affiliation:** College of Computing, Khon Kaen University, Khon Kaen, Thailand  
**ORCID:** [0009-0000-7693-7295](https://orcid.org/0009-0000-7693-7295)

---

## Overview

This repository contains the Python simulation code and LaTeX paper source for:

> **Frustration Ordering in Holographic Gauge-String Theory: U(1) to SU(3)×SU(2)×U(1)**  
> Boonsup Waikham, College of Computing, Khon Kaen University (2026)

We measure the HGST **MIXED frustration fraction** $R$ — the fraction of sign-inconsistent holonomy triads — across four gauge groups on classical 2D lattices, with finite-size scaling (FSS) to the thermodynamic limit.

### Principal results

| Gauge group | $R_\infty$ | Status |
|-------------|------------|--------|
| U(1) | 0 | Falsified |
| SU(2) | $0.3669 \pm 0.0036$ | Supported ✓ |
| SU(3) | $0.3539 \pm 0.0195$ | Supported ✓ |
| SM quarks | $\approx 0.493$ | Supported ✓ |
| *E. coli* RegulonDB | $0.349 \pm 0.018$ | (biological reference) |

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
│   ├── sm_*.py          Standard Model (SU(3)×SU(2)×U(1)) modules
│   └── ...
│
├── data/                JSON result files (all production runs)
│   ├── su2_l10_summary.json   SU(2) L=10 FSS result (R=0.364±0.001)
│   ├── p1_su3_fss_corrected.json
│   ├── p1_sm_fss_corrected.json
│   └── ...
│
├── paper/               LaTeX source
│   ├── main.tex         31-page preprint (Revision 2)
│   ├── references.bib
│   ├── main.bbl         Compiled bibliography (for arXiv)
│   └── response_to_reviewers.tex
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

## Reproducibility

All JSON files in `data/` are the actual production outputs used in the paper.  
Simulation seeds are fixed (`SEED=203` for SU(2) L=10; see each script header).  
The paper PDF can be recompiled from `paper/main.tex`:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
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
  version   = {2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/boonsup/hgst-e7}
}
```

See also `CITATION.cff` for machine-readable metadata.

---

## License

MIT — see [LICENSE](LICENSE).
