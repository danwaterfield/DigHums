# 18th-Century Novel Corpus Catalog

## Purpose
This catalog documents all texts in the corpus, with special attention to:
- **Attribution status** (named, anonymous, "By a Lady", disputed)
- **Publication dates**
- **Suitability for testing BERT attribution model**

---

## Corpus Summary

**Total Authors:** 13
**Total Texts:** 28+ texts
**Date Range:** 1719-1814

---

## Authors & Texts

### 1. Jane Austen (1 text)
- ✅ **Pride and Prejudice** (1813) - Named publication
- Status: Training corpus

### 2. Frances Burney (4 novels, 10+ texts)
- ✅ **Evelina** (1778) - Published **ANONYMOUSLY** as "By a Lady"
  - Later editions revealed as Burney after success
  - **Perfect test case for attribution!**
- ✅ **Cecilia** (1782, 3 volumes) - Published under Burney's name
- ✅ **Camilla** (1796) - Named publication
- ✅ **The Wanderer** (1814, 5 volumes) - Named publication
- Status: Primary subject, training corpus

### 3. Ann Radcliffe (6+ texts)
- ✅ **The Castles of Athlin and Dunbayne** (1789) - Published **ANONYMOUSLY**
  - Her FIRST novel
  - No author attribution on original title page
  - **Excellent test case!**
- ✅ **A Sicilian Romance** (1790) - Published as "By the Authoress of The Castles of Athlin and Dunbayne"
  - Still anonymous, but referenced first work
  - **Good test case for cross-work attribution**
- ✅ **The Romance of the Forest** (1791) - Name revealed in 2nd edition
- ✅ **The Mysteries of Udolpho** (1794) - Named publication (famous by this point)
- ✅ **The Italian** (1797, 2 volumes downloaded) - Named publication
- Status: Training corpus + test cases (early anonymous works)

### 4. Charlotte Smith (1 text)
- ✅ **Emmeline, the Orphan of the Castle** (1788) - Named publication
  - Gothic precursor, influenced Radcliffe and Austen
- Status: Corpus expansion (adds female Gothic voice)

### 5. Eliza Haywood (1 text)
- ✅ **Love in Excess** (1719-20) - Published **ANONYMOUSLY initially**
  - One of the bestselling novels of 18th century
  - Later attributed to Haywood
  - **Potential test case**
- Status: Corpus expansion (early period, female)

### 6. Clara Reeve (1 text)
- ✅ **The Old English Baron** (1777/1778) - Named publication
  - Originally published as "The Champion of Virtue" (1777) anonymously
  - Revised as "The Old English Baron" (1778) with name
  - **Potential test if we can find 1777 edition**
- Status: Corpus expansion (female Gothic)

### 7. Maria Edgeworth (1 text)
- ✅ **Castle Rackrent** (1800) - Published **ANONYMOUSLY**
  - First historical novel in English
  - **Excellent test case!**
- Status: Training corpus + potential test case

### 8. Samuel Richardson (1 text)
- ✅ **Pamela** (1740) - Published anonymously initially, then named
- Status: Training corpus

### 9. Henry Fielding (1 text)
- ✅ **Tom Jones** (1749) - Named publication
- Status: Training corpus

### 10. Tobias Smollett (2 texts)
- ✅ **Ferdinand Count Fathom** (1753) - Named
- ✅ **Humphry Clinker** (1771) - Named
- Status: Training corpus

### 11. Horace Walpole (1 text)
- ✅ **The Castle of Otranto** (1764) - First edition **ANONYMOUS** ("By Onuphrio Muralto")
  - Second edition (1765) revealed as Walpole
  - Founding text of Gothic novel
  - **Excellent test case!**
- Status: Corpus expansion (male Gothic) + test case

### 12. M.G. Lewis (1 text)
- ✅ **The Monk** (1796) - Initially "By M.G.L."
  - Later editions fully named
- Status: Corpus expansion (male Gothic, controversial)

### 13. William Beckford (1 text)
- ✅ **Vathek** (1786) - Originally published **ANONYMOUSLY** in English
  - French edition credited to Beckford
  - Oriental Gothic
  - **Potential test case**
- Status: Corpus expansion (male Gothic, unusual setting)

---

## Test Cases for Attribution Study

### Tier 1: High-Confidence Test Cases (Known Anonymous Publications)

1. **Frances Burney - Evelina (1778)**
   - Published as "By a Lady"
   - Later revealed as Burney after success
   - Model should identify as Burney based on style

2. **Ann Radcliffe - Castles of Athlin and Dunbayne (1789)**
   - Completely anonymous first publication
   - Model should identify as Radcliffe

3. **Ann Radcliffe - A Sicilian Romance (1790)**
   - "By the Authoress of The Castles..."
   - Tests cross-work attribution

4. **Maria Edgeworth - Castle Rackrent (1800)**
   - Published anonymously
   - Model should identify as Edgeworth

5. **Horace Walpole - Castle of Otranto (1764, 1st ed)**
   - Published as "By Onuphrio Muralto"
   - Model should identify as Walpole

### Tier 2: Interesting Attribution Questions

6. **Eliza Haywood - Love in Excess (1719-20)**
   - Initially anonymous bestseller
   - Tests early-period attribution

7. **William Beckford - Vathek (1786)**
   - Anonymous English edition
   - Unusual Oriental Gothic style

8. **Clara Reeve - Champion of Virtue (1777)**
   - Need to source 1777 anonymous edition
   - Compare to 1778 named edition

### Tier 3: Disputed/Unknown Cases (Future Research)

- Search for genuinely disputed 18th-century novels
- Anonymous "By a Lady" works with unknown authors
- Works initially attributed to wrong authors

---

## Corpus Statistics

### By Gender
- **Female authors:** 6 (Austen, Burney, Radcliffe, Smith, Haywood, Reeve, Edgeworth)
- **Male authors:** 7 (Richardson, Fielding, Smollett, Walpole, Lewis, Beckford)
- **Anonymous initially:** ~8 texts (many by women)

### By Genre
- **Gothic:** Radcliffe (6), Walpole (1), Lewis (1), Reeve (1), Beckford (1) = 10 texts
- **Domestic/Social:** Burney (4), Austen (1), Smith (1), Edgeworth (1) = 7 texts
- **Epistolary:** Richardson (1), Burney (1), Smollett (1) = 3 texts
- **Picaresque:** Fielding (1), Smollett (2) = 3 texts
- **Amatory:** Haywood (1)

### By Period
- **Early (1719-1749):** Haywood, Richardson, Fielding = 3 texts
- **Mid (1750-1779):** Smollett (1), Walpole, Reeve, Burney (1) = 4 texts
- **Late (1780-1799):** Burney (2), Radcliffe (5), Smith, Beckford, Lewis, Camilla = 9 texts
- **Turn of century (1800-1814):** Edgeworth, Austen, Burney (1) = 3 texts

---

## Next Steps

1. **Preprocess new texts** - Run through same pipeline as original corpus
2. **Update metadata.csv** - Include all new texts
3. **Retrain model** - Expand to 13 authors instead of 7
4. **Test on anonymous editions** - Create separate test set from known-anonymous works
5. **Document findings** - Which anonymous works are correctly attributed?

---

## Sources
- All texts from Project Gutenberg (public domain)
- Internet Archive for specific editions
- University of Pennsylvania Digital Library

Last updated: 2025-02-10
