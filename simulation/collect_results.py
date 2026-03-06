#!/usr/bin/env python3
"""
Collect and pretty-print all SU(3) scan results from su3_scans/data/.

Usage:
  python scripts/collect_results.py [--data-dir DATA_DIR]
"""

import json
import glob
import sys
import argparse
import numpy as np
from pathlib import Path


def _flatten(obj):
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(item if isinstance(item, list) else [item])
        return out
    return [obj]


def load_all(data_dir: str):
    records = []
    for fname in sorted(glob.glob(f"{data_dir}/*.json")):
        try:
            with open(fname) as f:
                records.extend(_flatten(json.load(f)))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: {fname}: {e}")
    return records


def summarize_beta_scan(records, L, kappa=0.3):
    pts = [r for r in records
           if int(r.get("L", 0)) == L
           and abs(r.get("kappa", -1) - kappa) < 0.01]
    if not pts:
        return
    pts.sort(key=lambda x: x["beta_g"])
    print(f"\n{'='*62}")
    print(f"SU(3) beta-scan: L={L}, kappa={kappa}")
    print(f"{'='*62}")
    print(f"  {'beta':>6}  {'plaq':>8}  {'R':>8}  {'R_err':>8}  "
          f"{'Omega_7':>8}  {'t(s)':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for r in pts:
        om7 = r.get("omega_7_mean", float("nan"))
        t   = r.get("t_meas_s", float("nan"))
        print(f"  {r['beta_g']:>6.2f}  "
              f"{r['plaq_mean']:>+8.4f}  "
              f"{r['R_mean']:>8.4f}  "
              f"{r['R_err']:>8.4f}  "
              f"{om7:>8.4f}  "
              f"{t:>6.1f}s")


def summarize_kappa_scan(records, L, beta):
    pts = [r for r in records
           if int(r.get("L", 0)) == L
           and abs(r.get("beta_g", -1) - beta) < 0.5]
    if not pts:
        return
    pts.sort(key=lambda x: x["kappa"])
    print(f"\n{'='*62}")
    print(f"SU(3) kappa-scan: L={L}, beta={beta:.1f}")
    print(f"{'='*62}")
    print(f"  {'kappa':>6}  {'plaq':>8}  {'R':>8}  {'R_err':>8}  "
          f"{'Omega_7':>8}  {'t(s)':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for r in pts:
        om7 = r.get("omega_7_mean", float("nan"))
        t   = r.get("t_meas_s", float("nan"))
        print(f"  {r['kappa']:>6.3f}  "
              f"{r['plaq_mean']:>+8.4f}  "
              f"{r['R_mean']:>8.4f}  "
              f"{r['R_err']:>8.4f}  "
              f"{om7:>8.4f}  "
              f"{t:>6.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    records = load_all(args.data_dir)
    print(f"\nLoaded {len(records)} result records from {args.data_dir}/")

    # Discover unique L values
    L_vals = sorted({int(r.get("L", 0)) for r in records if r.get("L")})

    # beta-scans at kappa=0.3
    for L in L_vals:
        summarize_beta_scan(records, L, kappa=0.3)

    # kappa-scans — detect by large kappa spread at fixed beta
    betas = sorted({round(r["beta_g"], 1) for r in records if "beta_g" in r})
    for beta in betas:
        for L in L_vals:
            pts = [r for r in records
                   if int(r.get("L", 0)) == L
                   and abs(r.get("beta_g", -1) - beta) < 0.5]
            kappas = [r.get("kappa", 0) for r in pts]
            if len(set(round(k, 2) for k in kappas)) >= 4:
                summarize_kappa_scan(records, L, beta)

    print()


if __name__ == "__main__":
    main()
