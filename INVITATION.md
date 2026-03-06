Subject: Invitation to Contribute — HGST E7: Frustration Ordering in Non-Abelian Gauge Theory

---

Dear [Colleague / Professor / Dr. Name],

I am writing to invite you to contribute to an open research project on
**frustration ordering in non-Abelian lattice gauge theory**, which I
believe aligns closely with your work on [lattice QCD / frustrated networks /
biological network topology / *your assessment here*].

---

**About the Project**

The HGST E7 project measures the *MIXED frustration fraction* R — the
fraction of sign-inconsistent holonomy triads — across a hierarchy of gauge
groups (U(1), SU(2), SU(3), and the full Standard Model product group) on
classical 2D lattices, using Metropolis Monte Carlo with finite-size scaling
(FSS).

Key results to date (Revision 3, March 2026):
  • U(1): R → 0 (Abelian, falsified)
  • SU(2): R_∞ = 0.3669 ± 0.0036  (verified)
  • SU(3): R_∞ = 0.3539 ± 0.0195  (consistent)
  • SM C-scalar: R_∞ = 0.4981 ± 0.0056 (verified)
  • SM N-scalar: R_∞ = 0.4980 ± 0.0076 (verified)
  • E. coli RegulonDB: R = 0.349 ± 0.018  (biological reference)

The simulation code is fully open, written in pure Python (numpy + scipy),
and includes a self-contained Colab-ready script for immediate reproducibility.

  GitHub:  https://github.com/boonsup/hgst-e7
  Zenodo:  https://doi.org/10.5281/zenodo.18873889
  Paper:   paper/Theoretical_Framework.tex (axiomatic)
           paper/Empirical_Study.tex (computational)

---

**Open Directions Where Contributions Would Be Valuable**

I am actively looking for collaborators in the following areas:

  1. LATTICE PHYSICS
     • SU(3) L=10 simulation (analogous to the completed SU(2) L=10 run)
       to reduce σ_syst on R_∞(SU3) from ±0.0195 to ±0.005 or better
     • 3D lattice extension (current code is 2D; physical relevance of 3D?)
     • Comparison with known SU(N) large-N scaling

  2. FRUSTRATED SYSTEMS / CONDENSED MATTER
     • Mapping the MIXED fraction to known frustration order parameters
       (e.g., Edwards-Anderson parameter, chirality)
     • Does R → 0.5 correspond to a known universality class?

  3. BIOLOGICAL NETWORKS / BIOPHYSICS
     • Independent verification of R(RegulonDB) using other GRN databases
       (YEASTRACT, STRING, ARACNE reconstructions)
     • Is the overlap R_∞(SU3) ≈ R(E. coli) numerically coincidental or structural?

  4. THEORY
     • Analytical bound on R for SU(N) in the weak/strong coupling limit
     • Relationship between R and the second Chern class / topological charge

  5. CODE QUALITY
     • GPU acceleration (CuPy / JAX port) of the Metropolis sweep
     • Unit-test coverage expansion
     • Benchmarking against known lattice observables (plaquette, Polyakov loop)

---

**What Collaboration Entails**

I propose a lightweight, open model:

  • All work is public on GitHub (MIT licence)
  • Contributors are credited in CITATION.cff and the paper acknowledgements
  • Co-authorship on any resulting new paper is negotiable based on contribution
  • No funding or institutional agreement required to start
  • Communication via GitHub Issues / Discussions, or email

---

**How to Get Started**

  1. Read the papers: paper/Theoretical_Framework.tex and paper/Empirical_Study.tex
  2. Run the standalone script in under 20 minutes:
       pip install numpy scipy
       python simulation/su2_l10_colab.py
  3. Browse open issues:  https://github.com/boonsup/hgst-e7/issues
  4. Reply to this letter, or open a GitHub Discussion

---

I would be very glad to discuss any aspect of the project, share additional
data, or schedule a video call at your convenience.

With kind regards,

Boonsup Waikham
College of Computing, Khon Kaen University
Khon Kaen, Thailand
ORCID: 0009-0000-7693-7295
Email: boonsup@kku.ac.th
GitHub: https://github.com/boonsup/hgst-e7
Zenodo: https://doi.org/10.5281/zenodo.18873889
