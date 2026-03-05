## Summary

<!-- One sentence: what does this PR add or fix? -->

## Type of change

- [ ] Bug fix (simulation produces wrong result or crashes)
- [ ] New simulation data (new JSON result file + FSS update)
- [ ] New feature (new gauge group, new observable, 3D extension, GPU port)
- [ ] Code quality (refactor, tests, docs)
- [ ] Paper / LaTeX update

## Physics checklist (for simulation changes)

- [ ] Run parameters (L, beta, kappa, seed, N_therm, N_meas) stated in JSON metadata
- [ ] Output JSON file added to `data/`
- [ ] Autocorrelation (τ_int via Madras-Sokal) reported
- [ ] FSS fit updated if a new L point was added
- [ ] No hard-coded absolute paths

## Code checklist

- [ ] `pip install -r requirements.txt && python simulation/su2_l10_colab.py` still runs
- [ ] New functions have a docstring stating the physics formula implemented
- [ ] No secrets or `.env` values committed

## Related issues

Closes #
