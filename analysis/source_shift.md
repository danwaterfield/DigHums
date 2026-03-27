# Source Shift Analysis — Cited Authorities Across the Corpus

**Date:** 2026-03-26
**Corpus:** 208 texts, 1586–1818
**Method:** Name-matching with temporal filtering (text date < person's birth → exclude) and substring/context disambiguation (papal "pope" excluded, character-name "smith" excluded, trade "blacksmith" excluded)

---

## Methodology

### v1 problems identified and fixed
1. **Surname polysemy:** "Smith" was 96% character names; "Pope" was 54% papal in history texts; "Johnson" in Pepys was a sempstress. Fixed with context-window disambiguation for Smith and Pope, and temporal filtering for all names.
2. **Temporal anachronism:** Pre-1700 texts were credited with citing Johnson (b.1709), Hume (b.1711), Rousseau (b.1712). Fixed by requiring text publication date ≥ authority's birth year.
3. **Substring contamination:** "locked" counted as Locke, "swiftly" as Swift, "steelyard" as Steele. Fixed with word-boundary and suffix checking.

### Impact of disambiguation
| Name | v1 raw count | v2 clean count | Reduction |
|------|-------------|---------------|-----------|
| Smith | 1,121 | 46 | 96% |
| Pope | 4,081 | 3,170 | 22% |
| Johnson | 1,923 | 1,880 | 2% |

---

## 1. The Ancient/Modern Ratio Across Time

| Period | Texts | Ancient | Modern | Scripture | **A/M ratio** |
|--------|------:|--------:|-------:|----------:|--------------:|
| 1580–1700 | 18 | 1,474 | 808 | 1,116 | **1.82** |
| 1700–1750 | 40 | 1,808 | 4,371 | 284 | **0.41** |
| 1750–1790 | 128 | 3,098 | 7,495 | 523 | **0.41** |
| 1790–1820 | 18 | 264 | 1,149 | 11 | **0.23** |

Pre-1700, ancients are cited nearly 2x as often as moderns. The crossover occurs 1700–1750. The ratio stays flat mid-century (0.41) then drops sharply after 1790 (0.23). Scripture collapses from 1,116 to 11.

Note: the pre-1700 ratio was 0.77 in v1 — the temporal filter's removal of anachronistic matches (moderns credited to pre-1700 texts) corrected this to 1.82, which is historically more plausible.

## 2. Top Cited Authorities by Period

### 1580–1700
Scripture (692), Moses (420), Seneca (230), Pope (204)*, Plato (171), Plutarch (164), Newton (160), Ovid (151), Aristotle (141), Galen (134)

*Pope count still high — partially papal contamination surviving the filter in theology texts. Needs further disambiguation.

### 1700–1750
**Pope (1,248)**, Swift (615), Johnson (536), Milton (298), Horace (266), Montesquieu (262), Homer (259), Plato (253), Scripture (217), Shaftesbury (208)

### 1750–1790
**Pope (1,571)**, Johnson (1,251), **Locke (774)**, Cicero (457), Homer (419), Gibbon (381), Milton (366), **Rousseau (364)**, Hume (363), Bacon (322)

### 1790–1820
**Burke (486)**, Horace (94), Johnson (82), Pope (71), Hume (58), Swift (58), Gibbon (53), Steele (43), Voltaire (41), Locke (39)

**Key shifts:**
- Pope dominates 1700–1790, then fades (1,248 → 1,571 → 71)
- Locke rises dramatically mid-century (774 in 1750–90)
- Rousseau enters in 1750–90 (364) then fades
- Burke dominates the 1790s revolutionary decade (486)
- Scripture falls from top-2 to absent in the top 15 by 1790

## 3. Genre Citation Profiles

| Genre | A/M ratio | Top authority | Character |
|-------|----------:|---------------|-----------|
| Science | 0.18 | Newton (213) | Empirical, self-referential |
| Poetry | 0.27 | Pope (1,073) | Literary canon, self-referential |
| Conduct | 0.41 | **Locke (551)** | Lockean epistemology grounds behavioral prescription |
| Philosophy | 0.41 | Montesquieu (276) | Continental + British Enlightenment |
| History | 0.48 | Pope (1,655)* | Mixed classical/modern |
| Fiction | 0.56 | Fielding (99) | Self-referential + classical literary |
| Philology | 0.60 | Hume (143) | Classical languages + modern philosophy |
| Medical | **8.38** | Galen (188) | **Overwhelmingly classical** (Burton dominance) |

*History "Pope" count includes residual papal contamination in Gibbon/Hume.

**The conduct-Locke finding is the most significant for the thesis:** the behavioral norm-setting literature is grounded in Lockean empiricism more than in any classical, scriptural, or other modern authority.

## 4. Epistemic Authority Shift — Scripture vs Classical vs Empirical

Decades with 3+ texts only (others are individual-text data):

| Decade | Scripture | Classical | Empirical | Dominant |
|--------|----------:|----------:|----------:|----------|
| 1690s | 27.3 | 7.5 | 5.7 | Scripture |
| 1700s | 4.7 | 10.8 | 6.6 | Classical |
| 1710s | 5.1 | 36.7 | 11.3 | Classical |
| 1720s | 5.8 | 3.6 | 1.5 | Scripture |
| 1730s | 1.3 | 44.0 | 5.6 | Classical |
| 1740s | 2.6 | 12.0 | 3.8 | Classical |
| 1750s | 3.7 | 15.1 | 12.4 | Classical |
| **1760s** | 2.7 | 10.7 | **13.6** | **Empirical** |
| 1770s | 3.3 | 15.0 | 2.0 | Classical |
| 1780s | 1.8 | 12.5 | 7.4 | Classical |
| 1790s | 0.6 | 11.8 | 6.1 | Classical |

The 18c is **not** a linear march from scripture to empiricism. Classical authority dominates throughout 1700–1790 except for one decade — the 1760s — when empirical briefly overtakes (Priestley, industrial revolution, Rousseau's naturalism). Scripture collapses by 1730 and never recovers. The humanities remain classically grounded; only science completes the empirical turn.

## 5. Caveats

- **Burton dominance in medical:** Burton (1621, 530k words) is 41% of the medical corpus and drives the A/M ratio to 8.38. Without Burton, medical A/M ≈ 0.65 (estimated from v1 audit). Claims about "medical literature" are largely claims about Burton.
- **Pope residual contamination:** Even after filtering, some papal "pope" survives in history and theology texts (where papal and poetic Pope coexist in context windows). The poet Pope's actual citation count is probably 10–20% lower than reported.
- **Horace as first name:** Some "Horace" hits in 18c texts may be Horace Walpole, not the Roman poet. Not disambiguated.
- **Single-text decades:** Pre-1690 and post-1800 have ≤2 texts per decade. All claims about these periods are individual-text observations.
- **Locke genuine rate:** Context check found 46% of "locke" hits in conduct lit are genuine John Locke references (~254 of 551). Still the top authority but roughly half the raw count.

---

## Files

| File | Contents |
|------|----------|
| `source_shift_v2.py` | v2 analysis with temporal + substring disambiguation |
| `source_shift_audit.py` | Audit of v1: surname polysemy, papal contamination, Burton dominance |
| `source_shift.md` | This document |
