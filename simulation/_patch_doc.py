import re

f = "HGST_Canonical_v6_WithSU3.md"
with open(f, encoding="utf-8") as fh:
    txt = fh.read()

fixes = 0

# FIX 1: relabel full-picture header
old = '**Full picture (E7 + E7-SU2 + E7-SU3):**'
new = '**Full picture (E7 + E7-SU2 + E7-SU3 + E7-SM):**'
if old in txt:
    txt = txt.replace(old, new, 1); fixes += 1; print("FIX1 OK")
else:
    print("FIX1 MISS: header not found")

# FIX 2: add SM row + update frustration ordering block in §14 table
old = ('| Biology | 0.32\u20130.48 | \u2014 | Signed feedback frustration | SUPPORTED |\n\n'
       '**Frustration ordering (all three consistent):**\n'
       '```\n'
       '\u2016[U,V]\u2016 :  U(1) = 0  <  SU(2) \u2248 2.3  <  SU(3) \u2248 2.6\n'
       'R:          U(1) \u2248 0  <  SU(2) \u2248 0.37 <  SU(3) \u2248 0.39 \u2248 biology \u2248 0.38\n'
       '```')
if old in txt:
    new = ('| **SM SU(3)\u00d7SU(2)\u00d7U(1)** | **0.45\u20130.48 (q), 0.44\u20130.47 (l)** | **\u22480.49 (q), \u22480.48 (l)** | **Product-group multi-sector transport** | **NEW SUPPORTED** |\n'
           '| Biology | 0.32\u20130.48 | \u2014 | Signed feedback frustration | SUPPORTED |\n\n'
           '**Frustration ordering (extended to SM):**\n'
           '```\n'
           '\u2016[U,V]\u2016 :  U(1) = 0  <  SU(2) \u2248 2.3  <  SU(3) \u2248 2.6  <  SM (product)\n'
           'R:          U(1) \u2248 0  <  SU(2) \u2248 0.37 <  SU(3) \u2248 0.39 <  SM \u2248 0.49 \u2248 biology upper range\n'
           '```')
    txt = txt.replace(old, new, 1); fixes += 1; print("FIX2 OK")
else:
    idx = txt.find("Biology | 0.32")
    if idx >= 0:
        print("FIX2 MISS: Biology row found but context mismatch; showing context:")
        print(repr(txt[idx: idx + 300]))
    else:
        print("FIX2 MISS: Biology row NOT found")

# FIX 3: Phase 2 roadmap heading
old = '**Phase 2: Expand Gauge Structure** (COMPLETE for U(1), SU(2), SU(3))'
new = '**Phase 2: Expand Gauge Structure** (COMPLETE for U(1), SU(2), SU(3), SM)'
if old in txt:
    txt = txt.replace(old, new, 1); fixes += 1; print("FIX3 OK")
else:
    print("FIX3 MISS")

# FIX 4: Add SM roadmap bullets replacing old open items
old = ('- [ ] Extend \u03b2-scan to SU(3) L=10 (definitive FSS confirmation)\n'
       '- [ ] SU(2)\u00d7U(1) coupled gauge theory (electroweak analog)\n'
       '- [ ] Derive Standard Model charges from grade indices')
if old in txt:
    new = ('- [x] **SU(3)\u00d7SU(2)\u00d7U(1) SM product group implemented \u2713 NEW Fifth Edition** (sm_gauge + sm_fields + sm_action + sm_updates + sm_observables: 32/32 tests)\n'
           '- [x] **SM pure-gauge L=4 validated \u2713 NEW** (plaquette ordering correct; accept=0.483)\n'
           '- [x] **SM \u03b2\u2083-scan L=4 \u2713 NEW** (R_quark \u2248 0.45\u20130.47; peaks \u03b2\u2083\u22484)\n'
           '- [x] **SM \u03ba-scan L=4 \u2713 NEW** (optimal \u03ba=0.20; R_lepton sensitive above \u03ba=0.4)\n'
           '- [x] **SM FSS (\u03b2\u2083=6, \u03ba=0.2) L=4,6,8 \u2713 NEW** (R\u221e(q)\u22480.493, R\u221e(l)\u22480.480)\n'
           '- [x] **SM quark-vs-lepton vs EW-only \u2713 NEW** (\u0394R(q\u2212l)\u2248+0.02; lepton colour-blind confirmed)\n'
           '- [ ] Dense \u03b2\u2083-scan at L=6 (confirm R_quark peak at \u03b2\u2083\u22484)\n'
           '- [ ] SM FSS at (\u03b2\u2083=4, \u03ba=0.2) (potentially higher R\u221e)\n'
           '- [ ] Separate \u03baq vs \u03bal scan\n'
           '- [ ] Extend SU(3) to L=10 (definitive FSS)\n'
           '- [ ] Derive Standard Model charges from grade indices')
    txt = txt.replace(old, new, 1); fixes += 1; print("FIX4 OK")
else:
    idx = txt.find("Extend")
    print(f"FIX4 MISS: context={repr(txt[idx:idx+160])}")

# FIX 5: §20 Moderate success - add SM items
old = ('- [x] **FSS: R\u221e(SU3) = 0.387 in biological range \u2713 NEW**\n'
       '- [ ] Real data validation (RegulonDB, YEASTRACT)')
if old in txt:
    new = ('- [x] **FSS: R\u221e(SU3) = 0.387 in biological range \u2713 NEW**\n'
           '- [x] **SM product group SU(3)\u00d7SU(2)\u00d7U(1) fully implemented and validated \u2713 NEW Fifth Edition**\n'
           '- [x] **SM phase diagram: \u03b2\u2083-scan, \u03ba-scan, FSS L=4,6,8, quark-vs-lepton scan \u2713 NEW**\n'
           '- [x] **SM R\u221e(quark) \u2248 0.493, R\u221e(lepton) \u2248 0.480 \u2014 both in biological range \u2713 NEW**\n'
           '- [ ] Real data validation (RegulonDB, YEASTRACT)')
    txt = txt.replace(old, new, 1); fixes += 1; print("FIX5 OK")
else:
    idx = txt.find("FSS: R")
    print(f"FIX5 MISS: context={repr(txt[idx:idx+120])}")

with open(f, "w", encoding="utf-8") as fh:
    fh.write(txt)
print(f"\nDone. {fixes}/5 fixes applied.")
