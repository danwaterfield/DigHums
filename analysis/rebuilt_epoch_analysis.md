# Rebuilt Epoch Analysis — Validated Lexicons

**Date:** 2026-03-26
**Supersedes:** `sociability_state_neurodivergence.md` (which used unvalidated lexicons)

---

## Methodology Changes

### What was wrong with v1
1. **Anachronisms:** "normal" (1828), "self-control" (1838), "social" (modern sense c.1830s), "standard" (behavioral sense 19c) — all projected 19c concepts onto 18c texts
2. **Polysemy inflation:** "civil" (political), "government" (state), "command" (military), "patient" (medical noun), "violent" (physical), "common" (shared/vulgar), "proper" (one's own), "absent" (physical) — all inflated bins with non-target meanings, accounting for 40–60% of hits in several bins
3. **Missing period vocabulary:** Hyphenated compounds ("self-command," "good-breeding," "fellow-feeling," "animal spirits") lost by tokenizer; period-specific terms ("fibres," "temperament," "constitution," "spirits," "breeding," "deportment," "reverie") absent from lexicons

### What v2 does differently
1. **Anachronisms removed:** "normal," "self-control," "standard" (behavioral), "exhaustion," "eccentric" dropped
2. **Polysemous terms removed:** "civil," "government," "command," "patient," "violent/violence," "common," "proper," "ordinary," "regular," "correct," "observe," "regard," "absent/absence," "notice," "feeling/feelings," "impression/impressions," "sensible," "disorder" (standalone)
3. **Phrase tokenizer:** Pre-processes text to join "self command" → SELF_COMMAND, "good breeding" → GOOD_BREEDING, "animal spirits" → ANIMAL_SPIRITS, "fellow feeling" → FELLOW_FEELING, etc.
4. **Period vocabulary added:** "fibres," "temperament," "constitution," "breeding," "manners," "deportment," "genteel," "address," "benevolence," "sympathy," "reverie," "hypochondriac/al"
5. **New humoral_body bin:** Tracks the older body model ("humour/s," "phlegmatic," "sanguine," "choleric," "bilious," "blood," "bile," "spirits," "complexion," "habit," "temper") to measure the paradigm shift
6. **Valence check:** Crude collocate analysis for "sensibility" — does it appear near approval or critique words?

### Bins used

| Bin | Term count | Description |
|-----|----------:|-------------|
| sociability | 28 | Social performance demand (polite, breeding, manners, benevolence, sympathy) |
| conformity_propriety | 12 | Behavioral norms, via media (propriety, decorum, moderation, temperance) |
| deviance_marked | 24 | Pathologised difference (madness, enthusiasm, melancholy, spleen, vapours) |
| nervous_body | 16 | The nervous body (nerves, sensibility, fibres, irritability, delicacy) |
| self_command | 12 | Self-governance (self-command, composure, forbearance, fortitude, patience) |
| overwhelm_sensory | 15 | Environmental sensory overload (tumult, noise, crowd, bustle, stench, agitation) |
| attention_faculty | 13 | Attention as cognitive faculty (attention, heed, watchful; inattention, heedless, abstracted, reverie) |
| humoral_body | 14 | The older body model (humours, sanguine, choleric, spirits, constitution, temperament) |

---

## Key Findings (revised)

### 1. The nervous/humoral paradigm shift

The ratio of nervous-body to humoral-body vocabulary crosses 1.0 in the **1730s**:

| Decade | Nervous/10k | Humoral/10k | **N/H ratio** |
|--------|------------:|------------:|--------------:|
| 1580s | 2.2 | 35.7 | 0.06 |
| 1620s | 2.2 | 17.6 | 0.13 |
| 1700s | 0.8 | 6.8 | 0.11 |
| 1720s | 8.2 | 12.9 | 0.64 |
| **1730s** | **48.0** | **25.7** | **1.87** |
| 1750s | 13.1 | 9.9 | 1.32 |
| 1760s | 27.0 | 11.9 | 2.27 |
| 1790s | 11.4 | 8.8 | 1.29 |

The humoral body is *typological* (choleric, sanguine, phlegmatic, bilious — discrete categories). The nervous body is *scalar* (more or less sensitive, more or less irritable — continuous variation). This shift from typology to scale is the conceptual precondition for measuring deviation. It precedes statistical normalisation by a century.

Cheyne's *English Malady* (1733) is the tipping point: nervous body 54.7/10k, humoral body 29.8/10k, ratio 1.84. Before Cheyne, humoral dominates everywhere. After, nervous dominates medical literature (44.3/10k in 1760–1820) and enters conduct literature (rising from 3.9 to 7.1).

### 2. The sociability ratchet (revised)

With polysemy stripped, the ratchet is subtler but real:

| Period | Conduct sociability | Fiction sociability | Fiction overwhelm |
|--------|-------------------:|-------------------:|------------------:|
| 1580–1700 | 20.4 | — | — |
| 1700–1760 | 22.4 | — | — |
| 1760–1820 | 29.7 | 18.7 | 4.1 |

Sociability demand in conduct literature rises 46% across the period. Fiction runs at 18.7 — lower than the prescription, higher than the medical texts (4.8). Overwhelm is real but quieter than v1 suggested (4.1 vs the inflated 10.8).

The gap between what's demanded (sociability 29.7 in conduct) and what the novels register as overwhelming (4.1) is still the space where behavioral failure is constructed. But it's a gap between two discourses, not a simple environmental variable.

### 3. Gregory remains the hinge text

| Metric | Gregory | Conduct avg (1760–1820) | Medical avg (1760–1820) |
|--------|--------:|------------------------:|------------------------:|
| Sociability | 54.2 | 29.7 | 4.8 |
| Nervous body | **48.1** | 7.1 | 44.3 |
| Attention | **13.0** | 8.1 | 5.2 |
| Conformity | 7.6 | 4.4 | 4.0 |

Gregory's nervous-body score (48.1) is comparable to the medical literature average (44.3) while his sociability score (54.2) nearly doubles the conduct average (29.7). He is the text that fuses medical and conduct registers.

### 4. The valence of "sensibility" shifts from approval to critique

| Author | Date | Uses | Approval | Critique |
|--------|------|-----:|---------:|---------:|
| Smith, *Moral Sentiments* | 1759 | 25 | 100% | 0% |
| Rousseau, *Emile* | 1762 | 17 | 100% | 0% |
| Gregory, *Father's Legacy* | 1774 | 8 | 67% | 33% |
| **Burney, *Cecilia*** | **1782** | **27** | **0%** | **100%** |
| Wollstonecraft, *Vindication* | 1792 | 69 | 45% | 55% |
| Edgeworth, *Practical Education* | 1795 | 36 | 60% | 40% |
| **Burney, *Wanderer*** | **1814** | **41** | **0%** | **100%** |

Burney uses "sensibility" in exclusively critical collocations from *Cecilia* onward — **before** Wollstonecraft's *Vindication* (1792). She is critiquing sensibility as a behavioral standard before the explicit feminist critique arrives. Wollstonecraft is split (45/55); Burney is unambiguous.

### 5. Attention vocabulary rises in conduct lit but not in fiction

| Period | Conduct attention | Fiction attention | Medical attention |
|--------|------------------:|------------------:|------------------:|
| 1580–1700 | 2.6 | — | 1.0 |
| 1700–1760 | 2.2 | — | 1.5 |
| 1760–1820 | **8.1** | 4.3 | **5.2** |

Conduct literature nearly quadruples its attention vocabulary in the late 18c (2.2 → 8.1). Medical literature also rises (1.5 → 5.2). Fiction sits between (4.3). The conduct books are increasingly prescribing attention as a behavioral requirement; the medical texts are increasingly theorising it as a faculty; the fiction dramatises the difficulty of meeting the prescription.

Inattention ratios (negative/positive) are highest in:
- Hume *Enquiry Morals* (0.62) — philosophically interrogating attention limits
- More *Coelebs* (0.49) — anxiously policing inattention
- Locke *Essay* (0.45) — defining attention as a cognitive operation with failure modes
- Burney *Cecilia* (0.30) — dramatising social inattention

### 6. Phrase-level findings

"Self-command" (40 corpus hits) is concentrated in Smith (17) and Burney's *Wanderer* (11) — the moral philosopher and the late novelist. "Good breeding" (162 hits) is overwhelmingly Chesterfield (99). "Fellow feeling" (51 hits) is almost entirely Smith (37). "Animal spirits" (56 hits) is distributed across medical, philosophical, and conduct texts.

These phrase-level signatures show that the self-governance vocabulary is not evenly distributed — it's authored by specific writers (Smith, Burney) rather than being a period-wide idiom. This matters: Burney's *Wanderer* (1814) has more "self-command" instances than any text except Smith, suggesting she is engaging directly with the Smithian moral framework.

---

## Caveats (updated)

- **Polysemy persists:** Even the cleaned bins have some polysemous terms ("society" can be specific societies; "breeding" can be animal breeding; "spirits" can be alcoholic). Further reduction would lose signal.
- **Valence analysis is crude:** 6-word window with hand-picked approval/critique lists. A proper collocate analysis with PMI scores would be more reliable.
- **OCR noise:** Internet Archive texts contain OCR artifacts that reduce accuracy. The long-s normalisation catches "ſ" but not garbled words.
- **Corpus balance:** Medical texts cluster in 1730s/1750s/1790s; conduct texts are denser 1790s–1800s. Decade averages can be driven by a single large text.
- **Missing from the argument:** Methodism (Wesley), Dissenting academies, parish records, Poor Law documents, actual asylum records. The institutional enforcement layer is still represented only by Howard and Woodward.

---

## Files

| File | Contents |
|------|----------|
| `rebuilt_epoch_analysis.py` | Analysis script with phrase tokenizer and validated lexicons |
| `lexicon_audit.py` | Audit of v1 lexicons: polysemy, anachronism, corpus frequencies, collocate analysis |
| `sociability_state_neurodivergence.md` | v1 findings (unvalidated lexicons — retained for comparison) |
