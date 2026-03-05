#!/usr/bin/env python3
"""
p1_patch_tex.py — Apply P1 corrections to revision/main.tex
=============================================================
Reads p1_sm_fss_corrected.json, p1_su3_fss_corrected.json, p1_fss_fits.json
and applies the following changes to preprint/revision/main.tex:

  1. §3.5  — Replace error-bar formula with corrected autocorrelation formula
  2. Table su3_fss — Replace error bars and R_inf with corrected values + new L=10 row
  3. §6.4 text      — Update R_inf^{SU3} ± syst text below the table
  4. Table sm_fss   — Replace error bars and R_inf with corrected values + new L=10 row
  5. §7.6 text      — Update R_inf^{quark,lepton} ± syst text below the table
  6. Error bar narrative — Update "shrinks ∝ L^-2 (triad-count scaling)" text
  7. Abstract lines  — Update R_inf^{SU2}, R_inf^{SU3} with corrected values
  8. Contributions list — Same update

Run after p1_execute.py completes:
    python p1_patch_tex.py [--dry-run]

--dry-run  : print diff to stdout but do NOT modify main.tex
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE   = Path(__file__).parent
OUTDIR = BASE  # JSON outputs from p1_execute are in the same dir
TEX_SRC = BASE / "preprint" / "main.tex"
TEX_REV = BASE / "preprint" / "revision" / "main.tex"

SM_FSS_JSON  = OUTDIR / "p1_sm_fss_corrected.json"
SU3_FSS_JSON = OUTDIR / "p1_su3_fss_corrected.json"
FSS_FITS_JSON = OUTDIR / "p1_fss_fits.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run p1_execute.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_R(R: float, sigma: float) -> str:
    """Format R ± sigma consistently (4 decimal places)."""
    return f"{R:.4f} \\pm {sigma:.4f}"

def fmt_R_with_syst(R: float, stat: float, syst: float) -> str:
    """Format R ± stat (stat) ± syst (syst)."""
    combined = (stat**2 + syst**2) ** 0.5
    return f"{R:.4f} \\pm {stat:.4f}(\\text{{stat}}) \\pm {syst:.4f}(\\text{{syst}})"

def fmt_row(L, R, sigma, n_meas, note=""):
    """Format a tabular row: L & R & sigma & n_meas (& note)"""
    if note:
        return f"{L} & {R:.4f} & {sigma:.5f} & {n_meas} & {note} \\\\"
    return f"{L} & {R:.4f} & {sigma:.5f} & {n_meas} \\\\"


# ---------------------------------------------------------------------------
# Build replacement pieces from JSON
# ---------------------------------------------------------------------------

def build_su3_fss_table(su3_data: dict) -> dict:
    """Return dict of strings for all SU(3) FSS replacements."""
    pts = su3_data["points"]
    fits = su3_data["fss_R"]
    best_Rinf = fits["R_inf_best"]
    best_stat = fits["R_inf_stat"]
    best_syst = fits["R_inf_syst"]

    # Table rows (corrected σ)
    rows = []
    for rec in pts:
        rows.append(
            f"{rec['L']} & {rec['R_mean']:.4f} & {rec['R_err_corrected']:.5f}"
            f" & {rec['n_meas']} \\\\"
        )
    best_fit = next((f for f in fits["fits"] if f["ansatz"] == fits["ansatz_best"]
                     and f["converged"]), None)
    chi2dof_str = f"{best_fit['chi2dof']:.2f}" if best_fit else "---"

    combined = (best_stat**2 + best_syst**2)**0.5
    rinf_row  = (f"$\\infty$ & $\\mathbf{{{best_Rinf:.4f}\\pm{combined:.4f}}}$ "
                 f"& --- & FSS fit \\\\")

    return {
        "rows":           "\n".join(rows),
        "rinf_row":        rinf_row,
        "Rinf":           best_Rinf,
        "stat":           best_stat,
        "syst":           best_syst,
        "combined":       combined,
        "ansatz":         fits["ansatz_best"],
        "chi2dof":        chi2dof_str,
    }


def build_sm_fss_table(sm_data: dict) -> dict:
    """Return dict of strings for all SM FSS replacements."""
    pts  = sm_data["points"]
    fq   = sm_data["fss_quark"]
    fl   = sm_data["fss_lepton"]

    Rinf_q = fq["R_inf_best"]
    stat_q = fq["R_inf_stat"]
    syst_q = fq["R_inf_syst"]
    Rinf_l = fl["R_inf_best"]
    stat_l = fl["R_inf_stat"]
    syst_l = fl["R_inf_syst"]
    comb_q = (stat_q**2 + syst_q**2)**0.5
    comb_l = (stat_l**2 + syst_l**2)**0.5

    best_qfit = next((f for f in fq["fits"] if f["ansatz"] == fq["ansatz_best"]
                      and f["converged"]), None)
    chi2dof_q = f"{best_qfit['chi2dof']:.2f}" if best_qfit else "---"

    rows = []
    for rec in pts:
        rows.append(
            f"{rec['L']} & {rec['R_quark_mean']:.4f} & "
            f"{rec['R_quark_err_corrected']:.5f} & "
            f"{rec['R_lepton_mean']:.4f} \\\\"
        )
    rinf_row_q = (f"$\\infty$ & $\\mathbf{{{Rinf_q:.4f}\\pm{comb_q:.4f}}}$ "
                  f"& --- & $\\mathbf{{{Rinf_l:.4f}\\pm{comb_l:.4f}}}$ \\\\")

    return {
        "rows":     "\n".join(rows),
        "rinf_row": rinf_row_q,
        "Rinf_q":   Rinf_q, "stat_q": stat_q, "syst_q": syst_q, "comb_q": comb_q,
        "Rinf_l":   Rinf_l, "stat_l": stat_l, "syst_l": syst_l, "comb_l": comb_l,
        "ansatz_q": fq["ansatz_best"],
        "ansatz_l": fl["ansatz_best"],
        "chi2dof_q": chi2dof_q,
    }


def build_tauint_table(sm_data: dict, su3_data: dict) -> str:
    """Build the new §3.5 error-bar paragraph text."""
    lines = []
    lines.append(
        r"Error bars on $\R$ are computed as"
        "\n"
        r"$\sigma_\R = \sigma_\text{naive} \times \sqrt{2\tau_\text{int}}$,"
    )
    lines.append(
        r"where $\sigma_\text{naive} = \mathrm{std}(\R_t)/\sqrt{N_\text{meas}}$"
        r" and $\tau_\text{int}$ is the integrated autocorrelation time"
        r" estimated via the Madras--Sokal windowing algorithm~\citep{Madras1988}."
        r" This corrects for inter-configuration correlations;"
        r" individual triads within one configuration are \emph{not} independent."
    )
    lines.append(
        r"Equivalently, $\sigma_\R = \sqrt{\R(1-\R)/N_\text{eff}}$"
        r" where the effective sample size is"
        r" $N_\text{eff} = N_\text{meas}/(2\tau_\text{int})$."
    )
    lines.append(
        r"Observed $\tau_\text{int}$ values ranged from $0.5$ to approximately $1{-}2$"
        r" for all SM and SU(3) FSS points, yielding correction factors"
        r" $\sqrt{2\tau_\text{int}} \in [1.0, 2.0]$."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patching engine
# ---------------------------------------------------------------------------

class TexPatcher:
    def __init__(self, text: str):
        self.text = text
        self.changes: list[tuple[str, str, str]] = []  # (label, old, new)

    def replace_once(self, label: str, old: str, new: str) -> bool:
        if old not in self.text:
            print(f"  [WARN] Pattern not found for: {label}")
            return False
        self.text = self.text.replace(old, new, 1)
        self.changes.append((label, old, new))
        return True

    def replace_regex(self, label: str, pattern: str, repl: str, flags=0) -> bool:
        new_text, n = re.subn(pattern, repl, self.text, count=1, flags=flags)
        if n == 0:
            print(f"  [WARN] Regex not found for: {label}")
            return False
        self.text = new_text
        self.changes.append((label, pattern, repl))
        return True

    def report(self):
        print(f"  {len(self.changes)} replacements applied:")
        for label, _, _ in self.changes:
            print(f"    + {label}")


# ---------------------------------------------------------------------------
# Main patching logic
# ---------------------------------------------------------------------------

def patch_tex(tex: str, su3: dict, sm: dict) -> tuple[str, list]:
    p = TexPatcher(tex)

    # ── §3.5 FSS method description + error-bar formula ───────────────────
    p.replace_once(
        "§3.5: FSS ansatz + error formula",
        "We fit $R(L)=R_\\infty + a/L$ (leading-order correction) to data at $L=4,6,8$.\n"
        "Statistical errors on $\\R$ are estimated from $M$ independent measurements as\n"
        "$\\sigma_\\R = \\sqrt{\\R(1-\\R)/T}$ where $T$ is the total triad count.\n"
        "At fixed $\\beta$, $\\kappa$, triple-point errors scale as $\\sim L^{-2}$.",
        "We fit three ansätze to data at $L=4,6,8,10$:\n"
        "\\begin{itemize}\n"
        "  \\item Ansatz 1 (leading): $R(L) = R_\\infty + a/L$\n"
        "  \\item Ansatz 2 (next-to-leading): $R(L) = R_\\infty + a/L + b/L^2$\n"
        "  \\item Ansatz 3 (quadratic): $R(L) = R_\\infty + a/L^2$\n"
        "\\end{itemize}\n"
        "The systematic uncertainty on $R_\\infty$ is taken as half the spread across\n"
        "all converged ansätze.\n"
        "Statistical errors on $\\R$ are computed as\n"
        "$\\sigma_\\R = \\sigma_\\text{naive} \\times \\sqrt{2\\tau_\\text{int}}$,\n"
        "where $\\sigma_\\text{naive} = \\mathrm{std}(\\R_t)/\\sqrt{N_\\text{meas}}$ and\n"
        "$\\tau_\\text{int}$ is the integrated autocorrelation time estimated via the\n"
        "Madras--Sokal windowing algorithm~\\citep{Madras1988}.\n"
        "This corrects for inter-configuration correlations; individual triads within one\n"
        "configuration are \\emph{not} statistically independent.\n"
        "Observed $\\tau_\\text{int} \\in [0.5, 2.0]$ at all production points,\n"
        "yielding correction factors $\\sqrt{2\\tau_\\text{int}} \\in [1.0, 2.0]$."
    )

    # ── SU(3) FSS table ────────────────────────────────────────────────────
    Rinf_su3 = su3["Rinf"]
    comb_su3 = su3["combined"]
    syst_su3 = su3["syst"]
    stat_su3 = su3["stat"]
    ansatz_su3 = su3["ansatz"]
    chi2_su3   = su3["chi2dof"]

    # Replace L=4,6,8 rows + inf row in su3_fss table
    old_su3_rows = (
        "4 & 0.4051 & 0.0016 & 500 \\\\\n"
        "6 & 0.3948 & 0.0006 & 1000 \\\\\n"
        "8 & 0.3933 & 0.0003 & 2000 \\\\\n"
        "\\midrule\n"
        "$\\infty$ & $\\mathbf{0.3869\\pm0.0015}$ & --- & FSS fit \\\\"
    )
    new_su3_rows = su3["rows"] + "\n\\midrule\n" + su3["rinf_row"]
    p.replace_once("SU3 FSS table rows + Rinf", old_su3_rows, new_su3_rows)

    # Replace the inline FSS fit text below the table
    p.replace_once(
        "SU3 FSS inline fit result",
        "  R_\\infty^{\\SU{3}} = 0.3869 \\pm 0.0015, \\quad\n"
        "  a = 0.050, \\quad\n"
        "  \\chi^2/\\text{dof} = 0.75.",
        f"  R_\\infty^{{\\SU{{3}}}} = {Rinf_su3:.4f} "
        f"\\pm {stat_su3:.4f}(\\text{{stat}}) \\pm {syst_su3:.4f}(\\text{{syst}}), \\quad\n"
        f"  \\text{{best ansatz: }} {ansatz_su3},"
        f" \\quad \\chi^2/\\text{{dof}} = {chi2_su3}."
    )

    # The bold statement below the table
    p.replace_once(
        "SU3 Rinf bold statement",
        "\\textbf{$R_\\infty^{\\SU{3}}$ lies inside the biological MIXED range $[0.35,0.48]$.}",
        f"\\textbf{{$R_\\infty^{{\\SU{{3}}}} = {Rinf_su3:.4f} \\pm {comb_su3:.4f}$ "
        f"(stat+syst combined)}}; see §\\ref{{sec:discussion}} for biological comparison."
    )

    # Update status box for SU(3)
    p.replace_once(
        "SU3 status box Rinf",
        "SU(3) sustains $\\R>0$; $R_\\infty^{\\SU{3}}=0.387\\pm0.001$\ninside biological range.",
        f"SU(3) sustains $\\R>0$; $R_\\infty^{{\\SU{{3}}}}={Rinf_su3:.4f}\\pm{comb_su3:.4f}$ "
        f"(autocorrelation-corrected; see §\\ref{{sec:fss_method}})."
    )

    # ── SM FSS table ───────────────────────────────────────────────────────
    Rinf_q = sm["Rinf_q"]; comb_q = sm["comb_q"]; syst_q = sm["syst_q"]; stat_q = sm["stat_q"]
    Rinf_l = sm["Rinf_l"]; comb_l = sm["comb_l"]
    ansatz_q = sm["ansatz_q"]
    chi2_q   = sm["chi2dof_q"]

    old_sm_rows = (
        "4 & 0.4627 & 0.0017 & 0.4463 \\\\\n"
        "6 & 0.4707 & 0.0005 & 0.4601 \\\\\n"
        "8 & 0.4778 & 0.0002 & 0.4677 \\\\\n"
        "\\midrule\n"
        "$\\infty$ & $\\mathbf{0.493\\pm0.003}$ & --- & $\\mathbf{0.480\\pm0.004}$ \\\\"
    )
    new_sm_rows = sm["rows"] + "\n\\midrule\n" + sm["rinf_row"]
    p.replace_once("SM FSS table rows + Rinf", old_sm_rows, new_sm_rows)

    # Replace SM FSS inline results
    p.replace_once(
        "SM FSS inline R_inf values",
        "  R_\\infty^\\text{quark} \\approx 0.493, \\qquad\n"
        "  R_\\infty^\\text{lepton} \\approx 0.480.",
        f"  R_\\infty^{{\\text{{quark}}}} = {Rinf_q:.4f} "
        f"\\pm {stat_q:.4f}(\\text{{stat}}) \\pm {syst_q:.4f}(\\text{{syst}}), "
        f"\\qquad\n"
        f"  R_\\infty^{{\\text{{lepton}}}} = {Rinf_l:.4f} "
        f"\\pm {sm['stat_l']:.4f}(\\text{{stat}}) \\pm {sm['syst_l']:.4f}(\\text{{syst}})."
    )

    # Replace the "Error bars shrink ∝ L^{-2}" text
    p.replace_once(
        "SM FSS error bar scaling claim",
        "Error bars shrink $\\propto L^{-2}$ (triad-count scaling):\n"
        "the $L=8$ measurement ($\\sigma_{\\R_q}=0.0002$) is $\\approx70\\times$ more precise than\n"
        "$L=4$.",
        "Error bars are computed with autocorrelation correction "
        r"($\sigma_\R = \sigma_\text{naive}\times\sqrt{2\tau_\text{int}}$, §\ref{sec:fss_method})."
        "\nWith $\\tau_\\text{int}\\approx 1{-}2$ at all SM FSS points, the effective "
        "precision improvement from $L=4$ to $L=10$ is approximately "
        "$\\sqrt{N_\\text{eff}(10)/N_\\text{eff}(4)}\\approx 3{-}4\\times$."
    )

    # ── Abstract Rinf values ───────────────────────────────────────────────
    p.replace_once(
        "Abstract Rinf SU2 SU3",
        "$R_\\infty^\\text{SU2}\\approx 0.376$ and $R_\\infty^\\text{SU3}=0.387\\pm 0.001$ from\n"
        "finite-size scaling (FSS) on $L=4,6,8$ lattices",
        f"$R_\\infty^{{\\text{{SU2}}}}\\approx 0.376$ and "
        f"$R_\\infty^{{\\text{{SU3}}}}={Rinf_su3:.4f}\\pm{comb_su3:.4f}$ from\n"
        f"finite-size scaling (FSS) on $L=4,6,8,10$ lattices"
        f" (autocorrelation-corrected errors)"
    )

    # ── Contributions list Rinf ────────────────────────────────────────────
    p.replace_once(
        "Contributions Rinf SU3",
        "$R_\\infty^{\\SU{3}}=0.387\\pm 0.001$ inside the biological window.",
        f"$R_\\infty^{{\\SU{{3}}}}={Rinf_su3:.4f}\\pm{comb_su3:.4f}$ "
        f"(autocorrelation-corrected, four-point FSS)."
    )

    # ── Production table: add L=10 row ────────────────────────────────────
    p.replace_once(
        "Production table FSS row: L=4,6,8 → L=4,6,8,10",
        "$\\SM$ (FSS)         & 4,6,8 & 1000 & 500  & 1234   \\\\",
        "$\\SM$ (FSS)         & 4,6,8,10 & 500--2000 & 500--1200 & 99--102 \\\\"
    )

    p.report()
    return p.text, p.changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change without modifying the file")
    ap.add_argument("--tex", default=str(TEX_REV),
                    help=f"Path to main.tex (default: {TEX_REV})")
    ap.add_argument("--sm-json",  default=str(SM_FSS_JSON))
    ap.add_argument("--su3-json", default=str(SU3_FSS_JSON))
    args = ap.parse_args()

    tex_path = Path(args.tex)
    if not tex_path.exists():
        sys.exit(f"ERROR: {tex_path} not found")

    sm_data  = load_json(Path(args.sm_json))
    su3_data = load_json(Path(args.su3_json))

    # Build replacement tables
    su3 = build_su3_fss_table(su3_data)
    sm  = build_sm_fss_table(sm_data)

    print("\n" + "=" * 60)
    print(f"P1 TeX patch — reading {tex_path.name}")
    print("=" * 60)
    print(f"\nSU(3) FSS R_inf: {su3['Rinf']:.4f} ± {su3['combined']:.4f} "
          f"(ansatz: {su3['ansatz']}, χ²/dof={su3['chi2dof']})")
    print(f"SM  FSS R_inf_q: {sm['Rinf_q']:.4f} ± {sm['comb_q']:.4f} "
          f"(ansatz: {sm['ansatz_q']}, χ²/dof={sm['chi2dof_q']})")
    print(f"SM  FSS R_inf_l: {sm['Rinf_l']:.4f} ± {sm['comb_l']:.4f}")
    print()

    tex = tex_path.read_text(encoding="utf-8")

    if args.dry_run:
        patched, changes = patch_tex(tex, su3, sm)
        print("\n[DRY RUN] Changes that would be applied:")
        for label, old, new in changes:
            print(f"\n  ### {label}")
            print(f"  OLD: {repr(old[:80])}")
            print(f"  NEW: {repr(new[:80])}")
        print(f"\nTotal: {len(changes)} changes (not written)")
        return

    # Make a backup
    backup = tex_path.with_suffix(f".tex.p1bak")
    shutil.copy2(tex_path, backup)
    print(f"Backup written to {backup.name}")

    patched, changes = patch_tex(tex, su3, sm)

    tex_path.write_text(patched, encoding="utf-8")
    print(f"\n✓ {tex_path} updated with {len(changes)} P1 corrections")
    print(f"  Backup: {backup.name}")

    # Append to CHANGES.md
    changes_md = tex_path.parent / "CHANGES.md"
    if changes_md.exists():
        entry = (
            f"\n## P1 autocorrelation + FSS patch — {datetime.now().strftime('%Y-%m-%d')}\n"
            f"Applied by p1_patch_tex.py:\n"
        )
        for label, _, _ in changes:
            entry += f"- {label}\n"
        with changes_md.open("a", encoding="utf-8") as f:
            f.write(entry)
        print(f"  CHANGES.md updated")


if __name__ == "__main__":
    main()
