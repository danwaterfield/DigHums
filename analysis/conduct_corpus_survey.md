# Conduct Book Corpus — Survey Against Burney & Young Fiction

**Date:** 2026-03-25
**Corpus:** 35 conduct/courtesy/polemic texts (22 authors, ~15 MB), 1688–1809
**Compared against:** Burney's 4 novels (Evelina, Cecilia, Camilla, The Wanderer); Young's 2 novels (Lucy Watson, Julia Benson)

---

## 1. Corpus Holdings

### Already held (5)
Gregory, Fordyce, Chapone (Letters on Improvement), More (Strictures), Chesterfield

### Newly acquired (30)

| Author | Title | Date | Source |
|--------|-------|------|--------|
| Mary Wollstonecraft | *Vindication of the Rights of Woman* | 1792 | Gutenberg |
| Mary Wollstonecraft | *Thoughts on the Education of Daughters* | 1787 | Gutenberg |
| Maria Edgeworth | *Practical Education* (2 vols) | 1798 | Gutenberg |
| Maria Edgeworth | *Letters for Literary Ladies* | 1795 | UPenn |
| Hannah More | *Coelebs in Search of a Wife* | 1809 | Gutenberg |
| Hannah More | *Thoughts on the Manners of the Great* | 1788 | Internet Archive |
| Mary Astell | *A Serious Proposal to the Ladies* | 1694 | Gutenberg |
| Mary Astell | *Some Reflections upon Marriage* | 1700 | Gutenberg |
| Daniel Defoe | *An Essay upon Projects* | 1697 | Gutenberg |
| Lady Mary Wortley Montagu | *Turkish Embassy Letters* | 1763 | Gutenberg |
| Thomas Day | *Sandford and Merton* | 1783 | Gutenberg |
| Jean-Jacques Rousseau | *Emile* (Foxley trans.) | 1762 | Gutenberg |
| James Nelson | *Essay on the Government of Children* | 1753 | Gutenberg |
| Jonathan Swift | *Letter to a Very Young Lady* | 1723 | Wikisource |
| Catharine Macaulay | *Letters on Education* | 1790 | Internet Archive |
| Thomas Gisborne | *Duties of the Female Sex* | 1797 | Internet Archive |
| Sarah Pennington | *Unfortunate Mother's Advice* | 1761 | Internet Archive |
| Hester Chapone | *Letter to a New Married Lady* | 1777 | Internet Archive |
| John Bennett | *Letters to a Young Lady* | 1789 | Internet Archive |
| Priscilla Wakefield | *Reflections on the Female Sex* | 1798 | Internet Archive |
| Mary Hays | *Appeal to the Men of Great Britain* | 1798 | Internet Archive |
| John Locke | *Some Thoughts Concerning Education* | 1693 | Internet Archive |
| Vicesimus Knox | *Liberal Education* | 1781 | Internet Archive |
| Samuel Richardson | *Familiar Letters* | 1741 | Internet Archive |
| William Darrell | *The Gentleman Instructed* | 1704 | Internet Archive |
| Adam Petrie | *Rules of Good Deportment* | 1720 | Internet Archive |
| Eliza Haywood | *Female Spectator* (Vol 1) | 1744 | Internet Archive |
| "Sophia" | *Woman Not Inferior to Man* | 1739 | UPenn |
| Marquess of Halifax | *Advice to a Daughter* | 1688 | Internet Archive |

### Not yet acquired (PDF-only, would need OCR)
Allestree (*Whole Duty of Man*, *Ladies Calling*), Steele (*Ladies Library*), Kenrick, Trimmer (2 texts), Mary Ann Radcliffe (*Female Advocate*), William Alexander (*History of Women*), Cadogan, Broadhurst, Clara Reeve (*Plans of Education*)

---

## 2. Key Findings

### 2.1 John Bennett is Burney's nearest conduct-book analogue

Bennett's *Letters to a Young Lady* (1789) is the **closest conduct book to every Burney novel** by cosine similarity on six thematic dimensions (female education, modesty/delicacy, marriage/duty, independence, sensibility, reputation):

| Burney novel | Bennett similarity | Next closest |
|--------------|-------------------|--------------|
| Evelina | 0.993 | Fordyce (0.981) |
| Cecilia | 0.991 | Fordyce (0.967) |
| Camilla | 0.974 | Rousseau (0.955) |
| The Wanderer | 0.950 | Rousseau (0.918) |

Bennett is largely forgotten. This consistent proximity suggests his vocabulary of female education, sensibility, propriety, and marriage occupies almost exactly the register Burney writes in. Published 1789, between *Cecilia* (1782) and *Camilla* (1796) — the influence direction is open: Bennett may be absorbing Burney's language as much as she absorbs his.

### 2.2 Gregory's *Father's Legacy* is a distilled concentrate

Gregory dominates the modesty/delicacy axis at 51.9/10k — nearly 3x the next highest text (Wollstonecraft at 18.6, who is arguing *against* these values). His sensibility score (64.9/10k) also towers above everything else. Despite being a very short text (13,096 words), it is essentially a purified extract of the values Burney's heroines navigate.

### 2.3 Wollstonecraft shares Burney's lexicon but inverts it

The *Vindication* appears in the top 5 matches for Evelina, Cecilia, and Camilla. This is not because Burney agrees with Wollstonecraft, but because Wollstonecraft writes *about* the same vocabulary cluster — reputation (50.5/10k, highest of any text), modesty, education, marriage — while attacking the prescriptive framework. They share the lexicon but invert the valence. This makes the *Vindication* a critical comparison text for understanding the ideological field Burney's fiction inhabits.

### 2.4 The post-1770 sensibility explosion

The conduct literature shows a dramatic chronological shift between pre-1770 and post-1770 texts:

| Theme | Pre-1770 avg | Post-1770 avg | Shift |
|-------|-------------|--------------|-------|
| Sensibility | 11.5/10k | 27.8/10k | +141% |
| Modesty/delicacy | 4.4/10k | 10.3/10k | +138% |
| Female education | 46.9/10k | 61.8/10k | +32% |
| Independence | 7.5/10k | 9.5/10k | +27% |

Burney's career (1778–1814) sits right in the middle of this amplification. The conduct books aren't just background — the vocabulary they amplified after 1770 is the vocabulary her novels are made of.

### 2.5 "Conduct" and "delicacy" are Burney keywords

Direct references to conduct-book concerns in Burney's fiction:

| Term | Evelina | Cecilia | Camilla | Wanderer |
|------|--------:|--------:|--------:|---------:|
| conduct | 25 | 94 | 98 | 61 |
| delicacy | 9 | 55 | 58 | 41 |
| propriety | 7 | 28 | 27 | 24 |
| education | 13 | 21 | 30 | 20 |
| accomplishments | 1 | 10 | 10 | 26 |

"Conduct" is doing heavy narrative work in the mature novels, not just appearing in passing. The rise in "accomplishments" in *The Wanderer* (26 occurrences) reflects that novel's sustained engagement with the question of what women can *do*.

### 2.6 More's *Coelebs* has the highest vocabulary overlap with Burney

Ranked by shared rare words (6+ characters, frequency 2–50) with Burney's combined fiction:

| Text | Shared rare words |
|------|------------------:|
| More: *Coelebs in Search of a Wife* | 2,573 |
| Rousseau: *Emile* | 2,500 |
| Chesterfield: *Letters to His Son* | 2,490 |
| Day: *Sandford and Merton* | 2,158 |
| Macaulay: *Letters on Education* | 1,913 |

*Coelebs* is a conduct book disguised as a novel (published 1809). Its vocabulary overlaps with Burney's fiction more than any other text in the corpus — suggesting either shared readership, mutual influence, or a common discursive register for conduct-novel hybrids.

### 2.7 Young's nearest conduct matches differ from Burney's

| Young: Lucy Watson | Young: Julia Benson |
|-------------------|---------------------|
| Montagu (0.980) | Bennett (0.988) |
| Haywood (0.980) | Wollstonecraft (0.988) |
| Richardson (0.969) | Hays (0.983) |
| Petrie (0.969) | Montagu (0.976) |
| Bennett (0.946) | Fordyce (0.969) |

Young's *Lucy Watson* clusters with the earlier, more practical, more social-observation end of the tradition (Montagu, Haywood, Richardson's letter manual, Petrie's deportment guide). Burney's matches cluster around the post-1770 sensibility/education debate. This is another way of measuring the transformation: Young writes in the older courtesy-book register, Burney in the newer sensibility register.

### 2.8 Shared distinctive bigrams

The most characteristic phrases shared between the conduct corpus and Burney's fiction:

| Bigram | Conduct corpus | Burney fiction |
|--------|---------------:|---------------:|
| young lady | 264 | 248 |
| young ladies | 110 | 122 |
| young woman | 85 | 111 |
| good nature | 150 | 37 |
| good humour | 84 | 58 |
| good breeding | 187 | 15 |
| good sense | 204 | 19 |
| good opinion | 46 | 54 |
| human nature | 174 | 27 |
| fellow creatures | 133 | 14 |
| young people | 300 | 13 |

"Young lady/ladies/woman" is equally dense in both corpora — the subject position is shared. "Good breeding," "good sense," and "good nature" are conduct-book staples that Burney uses but at lower density — she dramatises these qualities rather than prescribing them. "Fellow creatures" is notably high in the conduct corpus (133) and present in Burney (14) — a Wollstonecraftian/evangelical register.

---

## 3. Burney's named references to conduct authors

| Author | Evelina | Cecilia | Camilla | Wanderer |
|--------|:-------:|:-------:|:-------:|:--------:|
| Rousseau | 2 | — | — | — |
| Richardson | 2 | — | — | — |
| Locke | — | — | 1 | — |

Burney names very few conduct-book authors directly. Her engagement is structural and lexical rather than citational.

---

## 4. Directions for further work

- **Bennett deep-dive:** The consistent Bennett–Burney proximity warrants close reading. What is Bennett's argument? Does he cite Burney? Does his vocabulary reflect hers or an independent source?
- **Wollstonecraft valence analysis:** Shared lexicon, inverted values. A sentiment-aware comparison (are "delicacy" and "modesty" used approvingly or critically?) would distinguish Burney's conservative deployment from Wollstonecraft's critical one.
- **More's Coelebs as conduct-novel hybrid:** The highest vocabulary overlap with Burney. A structural comparison (epistolary vs narrative, plot patterns, heroine types) would test whether More is consciously imitating Burney's mode.
- **Chronological modelling:** The pre/post-1770 shift suggests a period effect. A finer-grained analysis (by decade) would show whether Burney leads, follows, or runs parallel to the conduct-book amplification of sensibility vocabulary.
- **Remaining acquisitions:** The PDF-only texts (Allestree, Steele, Trimmer, Radcliffe) would fill gaps at the early (1658–1714) and late (1787–1799) ends of the chronology.
