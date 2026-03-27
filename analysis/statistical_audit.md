# Statistical Audit of Corpus Analysis

**Date:** 2026-03-26
**Audits:** `sociability_state_neurodivergence.md` (v1) and `rebuilt_epoch_analysis.md` (v2)

---

## Summary

Several findings from v1 and v2 fail basic statistical scrutiny. This document records the problems found and the status of each claim.

---

## Fallacies and Problems Found

### 1. Single-text dominance in decade bins — SERIOUS

Seven of twenty-four decade bins contain only one text:

| Decade | Text | Words | Problem |
|--------|------|------:|---------|
| 1580s | Bright, *Melancholy* | 76,673 | All claims = one text |
| 1620s | Burton, *Anatomy of Melancholy* | 529,999 | All claims = one text |
| 1670s | Spinoza, *Ethics* | 89,096 | All claims = one text |
| 1680s | Halifax, *Advice to a Daughter* | 19,957 | All claims = one text |
| 1710s | Shaftesbury, *Characteristics* | 83,208 | All claims = one text |
| 1800s | Trotter, *Nervous Temperament* | 74,657 | All claims = one text |
| 1810s | Burney, *The Wanderer* | 332,032 | All claims = one text |

Additionally, the **1730s are 85% Cheyne** (89,260 of 104,946 words). The "1730s nervous-body crisis" is really just a claim about one book.

**Correction:** All decade-level claims for single-text decades must be restated as claims about individual texts, not period-level trends.

### 2. Valence analysis sample size — CRITICAL, RETRACT

The claim that "Burney uses sensibility in exclusively critical collocations" rests on:
- *Cecilia* (1782): 27 uses of "sensibility," **1 collocate hit** (critique)
- *Wanderer* (1814): 41 uses, **2 collocate hits** (both critique)

This is a 96% miss rate. The crude word-list method does not find real collocates. Moreover, the finding is **window-size dependent**:

| Window | *Cecilia* critique % | *Wanderer* critique % |
|--------|--------------------:|---------------------:|
| ±4 | 100% (1/1) | 100% (1/1) |
| ±6 | 100% (1/1) | 100% (2/2) |
| ±12 | **50%** (1/2) | 67% (2/3) |
| ±20 | **33%** (1/3) | 86% (6/7) |

At ±12 the *Cecilia* finding reverses. Bootstrap CIs are degenerate ([100%, 100%]) because the sample is too small to estimate anything.

Only Wollstonecraft's *Vindication* (69 uses, 11 collocates) has enough data for the method to produce non-trivial results, and even she is unstable (55% at ±6, 63% at ±20).

**Status: RETRACT.** Replace with proper collocate analysis (PMI, log-likelihood, or manual close reading) before re-asserting.

### 3. Term overlap between bins — METHODOLOGICAL FLAW

"Constitution" (623 corpus hits) and "temperament" (170 hits) appear in **both** the nervous_body and humoral_body bins. This creates mechanical correlation between the two scores.

After removing shared terms, the nervous/humoral crossover still occurs (ratio crosses 1.0 in the 1720s–40s), but the effect is less dramatic:

| Period | With shared terms | Without shared terms |
|--------|------------------:|---------------------:|
| 1720–1740 | 1.87 | 1.07 |
| 1740–1760 | 1.32 | 1.09 |
| 1760–1780 | 2.27 | 1.15 |

**Status:** The crossover is real but less dramatic than reported. Future analysis should use non-overlapping bins.

### 4. Selection bias — ACKNOWLEDGED

The corpus was selected to test a specific thesis:
- Medical texts selected *because* they discuss nervous disorders
- Conduct texts selected *because* they prescribe behavior
- Fiction = Burney (the thesis subject)

This is **not a random sample of 18c print culture.** Claims like "medical literature has higher conformity vocabulary than conduct literature" are claims about *these selected texts*, not about the categories in general. Missing: other novelists (Richardson, Fielding, Smollett, Austen), periodicals, parliamentary proceedings, legal texts, Dissenting/Methodist religious texts, commercial literature.

**Status:** The selection is defensible for the argument but must not be over-generalised.

### 5. Ecological fallacy — category averages mask variance

Conduct-literature sociability (1760–1820) has:
- Mean: 30.7/10k
- SD: 13.7
- CV: **44%**
- Range: 10.4 (Rousseau) to 72.0 (More's *Manners of the Great*)

The within-group variance is larger than the between-period differences being claimed as trends. The "category average" is a poor summary statistic for this data.

**Status:** Report individual-text scores alongside averages; avoid claims based on averages alone.

### 6. The sociability "rise" — BORDERLINE

Bootstrap 95% CIs for conduct-literature sociability:
- 1580–1700: 20.2 [16.8, 24.2] (5 texts)
- 1700–1760: 21.3 [19.3, 23.6] (8 texts)
- 1760–1820: 28.1 [26.7, 29.6] (22 texts)

The first two periods overlap — there is no significant rise from 1580 to 1760. The 1760–1820 rise is real (CIs don't overlap), but partly driven by a much larger sample (22 texts vs 5 and 8). The effect may be sample-composition rather than a genuine period trend.

**Status: HOLDS WEAKLY.** The late-century rise exists but is modest and could reflect sample composition.

### 7. Genre-time confound — UNRESOLVABLE

Medical texts cluster 1720–1800; theology clusters 1650–1710; politeness = 1711–1759; fiction = 1768–1814. We cannot separate "medical texts use more nervous vocabulary because they're medical" from "later texts use more nervous vocabulary because the concept evolved."

Test using conduct texts (which span the full period):
- 1680–1730: nervous 2.6, humoral 10.1, ratio 0.26
- 1730–1770: nervous 6.1, humoral 8.6, ratio 0.71
- 1770–1820: nervous 7.5, humoral 8.9, ratio 0.84

The nervous/humoral ratio rises in conduct literature but **never crosses 1.0** — the humoral model remains dominant in conduct literature throughout. The "paradigm shift" is a medical-literature phenomenon, not a general cultural one.

**Status:** The nervous-body vocabulary rises in conduct lit (0.26 → 0.84) but the humoral body retains primacy. The claim of a general "paradigm shift" needs qualification.

### 8. Multiple comparisons — NO CORRECTION

8 bins × 74 texts × 3 periods × 6 categories = thousands of implicit comparisons. No statistical tests applied (no p-values, no corrections for multiple testing). Any pattern could be noise unless the effect size is very large.

**Status:** Only claims with very large effect sizes (Cheyne nervous=54.7, Gregory nervous=48.1 vs conduct avg ~7) are safe from this problem.

### 9. Corpus size asymmetry

| Category | Texts | Words | Min text | Max text | Size ratio |
|----------|------:|------:|---------:|---------:|-----------:|
| Conduct | 35 | 2,822,225 | 3,790 | 292,404 | 77x |
| Medical | 10 | 1,279,362 | 20,012 | 529,999 | 26x |
| Fiction | 6 | 1,235,979 | 56,239 | 363,702 | 6x |
| Theology | 4 | 923,637 | 175,684 | 334,455 | 2x |
| Urban | 2 | 108,734 | 29,219 | 79,515 | 3x |

Conduct has 35 texts; urban has 2. Category-level comparisons are comparing very different sample sizes.

---

## Revised Status of Claims

### HOLDS — large effect, robust to audit

| Claim | Evidence |
|-------|----------|
| Cheyne's *English Malady* (1733) is a pivotal text for nervous-body vocabulary | Nervous score 54.7/10k, 10x the corpus average. Single-text, but the effect size is unambiguous. |
| Gregory's *Father's Legacy* (1774) fuses medical and conduct registers | Nervous score 48.1/10k, bootstrap CI [36.2, 56.9]. Next highest conduct text is Chapone at 38.4. Gregory is genuinely anomalous within conduct literature. |
| Gregory was himself a physician writing a conduct book | Biographical fact, not a statistical claim. |
| Nervous vocabulary rises in conduct literature (0.26 → 0.84 ratio) | Consistent direction across three periods using the only category that spans the full timeframe. |

### HOLDS WEAKLY — real but noisy

| Claim | Problem |
|-------|---------|
| Sociability vocabulary rises in late-18c conduct lit | CIs for 1760–1820 don't overlap earlier periods, but within-group CV is 44%. |
| The nervous/humoral crossover occurs in the 1720s–30s | Real even with shared terms removed (ratio 1.07), but less dramatic than reported (1.87 was inflated by collinear terms). |
| Attention vocabulary rises in late-18c conduct lit | Consistent direction (2.2 → 8.1) but no significance test. |

### RETRACT — insufficient evidence

| Claim | Problem |
|-------|---------|
| Burney critiques sensibility before Wollstonecraft | Rests on 1–2 collocates. Window-size dependent. Must be replaced with proper collocate analysis or close reading. |
| The "ratchet" — sociability/overwhelm gap narrows over time | v2 overwhelm numbers are 0–4/10k, too small and noisy for a trendline. Single-text decades distort the pattern. |
| Fiction overwhelm is higher than conduct overwhelm | 4.1 vs 2.8, tiny absolute difference, no significance test. |
| Medical texts have higher conformity vocabulary than conduct texts | Selection bias: we picked medical texts about pathology. Cannot generalise. |
| The 1730s are a "crisis point" | 85% driven by one text (Cheyne). |

---

## Recommendations

1. **Report individual texts, not decade averages,** for any decade with fewer than 4 texts.
2. **Remove shared terms** (constitution, temperament) from either the nervous or humoral bin — they cannot appear in both.
3. **Replace the valence analysis** with proper PMI-based collocate analysis, or with manual close reading of sensibility passages.
4. **Expand the fiction corpus** (Richardson, Fielding, Smollett, Austen) to test whether Burney's patterns are typical or exceptional.
5. **Add significance tests** (permutation tests, bootstrap CIs) to any claim that rests on between-group comparison.
6. **Acknowledge the selection frame** explicitly in any write-up: these findings describe the relationship between selected canonical texts, not 18c discourse in general.

---

## Files

| File | Contents |
|------|----------|
| `statistical_audit.py` | Audit script: single-text dominance, bootstrap CIs, term overlap, valence sensitivity |
| `statistical_audit.md` | This document |
