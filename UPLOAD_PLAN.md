# Upload & Release Plan — HGST E7 Preprint + Code
## arXiv · Zenodo · GitHub

Status: main.tex 31 pages, 0 errors (Revision 2, 2026-03-05)

---

## PHASE 1 — GitHub Repository (do first, Zenodo needs the URL)

### 1-A  Repo scaffold (files already created in this folder)
- [ ] Copy `.env.example` → `.env`, fill in your name/ORCID/institute
- [ ] Confirm `.gitignore` covers all binary/secret paths (see file)
- [ ] Confirm `requirements.txt` matches your venv versions

### 1-B  Proposed repo structure
```
HGST-E7/                          ← repo root
├── .env.example                   ← author metadata template (committed)
├── .gitignore
├── requirements.txt
├── README.md                      ← 1-paragraph overview + badge links
├── CITATION.cff                   ← citable metadata (GitHub auto-renders)
│
├── simulation/                    ← all Python physics modules
│   ├── su2.py
│   ├── su3.py
│   ├── u1.py
│   ├── lattice.py
│   ├── fields.py
│   ├── action.py
│   ├── observables.py
│   ├── positive_control.py
│   ├── updates.py
│   ├── simulation.py
│   ├── sm_action.py
│   ├── sm_fields.py
│   ├── sm_gauge.py
│   ├── sm_observables.py
│   ├── sm_updates.py
│   ├── kappa_scan.py
│   ├── sm_scan.py
│   ├── run_simulation.py
│   ├── su2_l10_run.py
│   └── su2_l10_colab.py          ← Colab-ready standalone
│
├── data/                          ← JSON result files (committed)
│   ├── su2_l10_summary.json
│   ├── su2_l10_runlog.json
│   ├── p1_su3_fss_corrected.json
│   ├── p1_sm_fss_corrected.json
│   ├── n_commutator_results.json
│   ├── h_regulondb_results.json
│   ├── p_null_distribution_results.json
│   ├── p1_fss_fits.json
│   ├── kappa_scan_results.json
│   ├── su3_L4_bscan.json
│   ├── su3_L4_kscan.json
│   ├── su3_L6_b8.json
│   ├── sm_beta3_scan_L4.json
│   ├── sm_fss_kappa0.2.json
│   ├── sm_kappa_scan_L4.json
│   ├── sm_matter_L4.json
│   ├── sm_pure_gauge_L4.json
│   └── sm_qvsl_L4.json
│
├── paper/                         ← LaTeX source
│   ├── main.tex
│   ├── references.bib
│   └── response_to_reviewers.tex
│
└── notebooks/
    └── vacation_analysis.ipynb
```

### 1-C  Files to NOT commit (covered by .gitignore)
- `.env`  (contains real name/ORCID)
- `__pycache__/`, `*.pyc`
- `*.pdf`, `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`, `*.toc`
- `*.tar.gz`, `hgst_data_*.tar.gz`
- `revision2/HGST_main_PhD_Mentor_Review_v2.docx` (confidential review)
- `main.tex.p1bak` (working backup)

### 1-D  GitHub steps
```bash
cd HGST-E7
git init
git add .
git commit -m "Initial release: HGST E7 simulation code + paper (Revision 2)"
# Create repo on github.com  →  hgst-e7  (public)
git remote add origin https://github.com/<USER>/hgst-e7.git
git branch -M main
git push -u origin main
# Create a release tag:
git tag -a v2.0.0 -m "Revision 2 — SU(2) L=10 FSS complete"
git push origin v2.0.0
```

---

## PHASE 2 — Zenodo Deposit (reproducible archive)

Zenodo auto-imports from GitHub releases with a DOI badge.

### 2-A  Link GitHub → Zenodo
1. Go to https://zenodo.org/account/settings/github/
2. Toggle ON the `hgst-e7` repository
3. Push the `v2.0.0` tag → Zenodo auto-creates a draft deposit

### 2-B  Zenodo metadata (fill in deposit form)
```
Title:       Simulation Code for "Frustration Ordering in
             Holographic Gauge-String Theory: U(1) to SU(3)×SU(2)×U(1)"
Version:     2.0.0
Upload type: Software
Authors:     (from .env: AUTHOR_NAME, AUTHOR_ORCID, AUTHOR_AFFIL)
Description: Python simulation code for HGST MIXED-fraction R measurement
             across U(1), SU(2), SU(3), and SM gauge groups on 2D lattices.
             Includes FSS analysis and Colab-ready standalone (su2_l10_colab.py).
License:     MIT  (or CC BY 4.0 — your choice)
Related identifiers:
  → arXiv:XXXX.XXXXX  (add once arXiv ID is known)
Keywords:    lattice gauge theory, finite-size scaling, non-Abelian gauge groups,
             MIXED frustration, HGST, SU(2) SU(3), gene regulatory networks
```

### 2-C  Add DOI badge to README.md
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](...)
```

---

## PHASE 3 — arXiv Submission

### 3-A  Prepare submission archive
```bash
mkdir arxiv_submit
cp paper/main.tex arxiv_submit/
cp paper/references.bib arxiv_submit/
# run bibtex + pdflatex twice locally to confirm clean compile
cd arxiv_submit && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### 3-B  arXiv metadata (https://arxiv.org/submit)
```
Title:     Frustration Ordering in Holographic Gauge-String Theory:
           U(1) to SU(3)×SU(2)×U(1)
Authors:   (from .env: AUTHOR_NAME)
Affil:     (from .env: AUTHOR_AFFIL)
Category:  hep-lat  (primary)
Cross:     cond-mat.dis-nn, q-bio.MN
Abstract:  (copy from main.tex \begin{abstract}...\end{abstract})
Comments:  31 pages. Code: https://github.com/<USER>/hgst-e7
           (Zenodo DOI: 10.5281/zenodo.XXXXXXX)
```

### 3-C  arXiv checklist
- [ ] No `\usepackage{hyperref}` version conflict
- [ ] All fonts embedded (check with `pdffonts main.pdf`)
- [ ] No absolute local file paths in .tex
- [ ] `.bbl` file included (arXiv needs it, not just .bib)
- [ ] No PDFs/figures referenced that aren't in the archive
- [ ] Compile passes with arXiv's TeX Live 2023 (or use `\pdfoutput=1`)

---

## PHASE 4 — CITATION.cff (auto-render on GitHub)

Create `CITATION.cff` in repo root (template in this folder).
GitHub will show a "Cite this repository" button automatically.

---

## PHASE 5 — README.md content outline

```markdown
# HGST E7: Frustration Ordering in Non-Abelian Gauge Theory

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](...)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](...)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quick start
pip install -r requirements.txt
python simulation/su2_l10_colab.py   # standalone FSS run

## Colab
Upload simulation/su2_l10_colab.py → runtime → run all

## Results summary
| Gauge group | R_inf         |
|-------------|---------------|
| U(1)        | 0 (falsified) |
| SU(2)       | 0.3669±0.0036 |
| SU(3)       | 0.3539±0.0195 |
| SM quarks   | ≈0.493        |

## Paper
paper/main.tex — arXiv:XXXX.XXXXX
```

---

## COMPLETION CHECKLIST

- [ ] P1 — `.env` filled, `.gitignore` ✓, `requirements.txt` ✓
- [ ] P1 — `CITATION.cff` written
- [ ] P1 — `README.md` written
- [ ] P1 — GitHub repo created + v2.0.0 tag pushed
- [ ] P2 — Zenodo linked, DOI obtained
- [ ] P2 — DOI badge added to README
- [ ] P3 — arXiv `.bbl` checked, archive built, submitted
- [ ] P3 — arXiv ID obtained → add to README + Zenodo related-id
- [ ] P3 — Update CITATION.cff with arXiv ID
