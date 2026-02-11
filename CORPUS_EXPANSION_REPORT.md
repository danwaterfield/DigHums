# Corpus Expansion Report
**Date:** 2025-02-10

## Mission Accomplished ✅

Successfully expanded the 18th-century English novel corpus from **7 authors** to **13 authors**, adding **10 new texts** with a focus on anonymous publications suitable for testing BERT attribution.

---

## New Texts Downloaded

### Women Authors (3 new authors, 4 texts)

1. **Charlotte Smith** - *Emmeline, the Orphan of the Castle* (1788)
   - 1.2 MB | Gothic precursor that influenced Radcliffe and Austen
   - Published under her name

2. **Eliza Haywood** - *Love in Excess* (1719-20)
   - 511 KB | 18th-century bestseller
   - **Originally published ANONYMOUSLY** ✨

3. **Clara Reeve** - *The Old English Baron* (1777/1778)
   - 328 KB | Gothic novel
   - First edition (1777) was anonymous as "The Champion of Virtue"

### Male Authors (3 new authors, 6 texts)

4. **Horace Walpole** - *The Castle of Otranto* (1764)
   - 234 KB | **Founding Gothic novel**
   - **First edition published anonymously as "By Onuphrio Muralto"** ✨

5. **M.G. Lewis** - *The Monk* (1796)
   - 809 KB | Controversial Gothic masterpiece
   - Initially "By M.G.L."

6. **William Beckford** - *Vathek* (1786)
   - 237 KB | Oriental Gothic
   - **English edition published anonymously** ✨

### Ann Radcliffe Expansion (4 new texts)

7. **The Castles of Athlin and Dunbayne** (1789)
   - Her **FIRST novel, published completely anonymously** ✨
   - **Perfect test case!**

8. **The Romance of the Forest** (1791) - 774 KB

9. **The Italian, Vol 1** (1797) - 336 KB

10. **The Italian, Vol 2** (1797) - 334 KB

---

## Corpus Statistics

### Before → After
- **Authors:** 7 → **13** (+86%)
- **Texts:** 18 → **28+** (+55%)
- **Total size:** ~12 MB → **~19 MB** (+58%)
- **Female authors:** 4 → **7** (54% of corpus)
- **Anonymous publications:** 2-3 → **8** texts

### Genre Distribution
- **Gothic:** 10 texts (Radcliffe, Walpole, Lewis, Reeve, Beckford)
- **Domestic/Social:** 7 texts (Burney, Austen, Smith, Edgeworth)
- **Epistolary:** 3 texts (Richardson, Burney, Smollett)
- **Picaresque:** 3 texts (Fielding, Smollett)
- **Amatory:** 1 text (Haywood)

---

## Test Cases for Anonymous Attribution

### Tier 1: Known Anonymous Publications (High Confidence)

These texts were published anonymously but we know the true author:

1. **Frances Burney - Evelina (1778)** - "By a Lady"
2. **Ann Radcliffe - Castles of Athlin and Dunbayne (1789)** - Completely anonymous
3. **Ann Radcliffe - A Sicilian Romance (1790)** - "By the Authoress of..."
4. **Maria Edgeworth - Castle Rackrent (1800)** - Anonymous
5. **Horace Walpole - Castle of Otranto (1764)** - "By Onuphrio Muralto"

### Tier 2: Initially Anonymous

6. **Eliza Haywood - Love in Excess (1719-20)**
7. **William Beckford - Vathek (1786)**
8. **Samuel Richardson - Pamela (1740)** - First edition anonymous

---

## Key Insights for Attribution Study

### Anonymous Publication Patterns

**Women authors** predominantly used anonymity:
- Burney: "By a Lady" (social propriety)
- Radcliffe: Completely anonymous first two novels
- Edgeworth: Anonymous publication for first novel
- Haywood: Anonymous during amatory fiction scandal period

**Male authors** used pseudonyms or initials:
- Walpole: Fake foreign author "Onuphrio Muralto" (Gothic authenticity)
- Lewis: "M.G.L." (scandalous content)
- Beckford: Anonymous (Oriental subject matter)

### Research Questions

1. **Can BERT identify authors from anonymous works?**
   - Train on named works → Test on anonymous editions
   - Expected: Should identify correctly based on style

2. **Does anonymity correlate with stylistic differences?**
   - Compare Burney's Evelina (anon) vs. Camilla (named)
   - Did authors "disguise" their style when anonymous?

3. **Cross-work attribution**
   - Can model identify Radcliffe's 2nd novel from "by the Authoress of..." attribution?
   - Tests whether model learns author signature vs. work signature

4. **Viral launch hook**
   - "AI Solves 200-Year-Old Literary Mystery"
   - Could test genuinely disputed attributions (need to source)

---

## Next Steps

### Phase 3A: Data Preparation
- [ ] Run preprocessing pipeline on all 10 new texts
- [ ] Update metadata.csv with attribution status field
- [ ] Create separate test set for anonymous works

### Phase 3B: Model Training
- [ ] Retrain BERT on expanded 13-author corpus
- [ ] Expected: More robust model with better generalization
- [ ] Training time: ~45 min on Colab (13 authors vs. 7)

### Phase 3C: Anonymous Attribution Testing
- [ ] Create "anonymous test set" from known-anonymous works
- [ ] Run attribution predictions
- [ ] Measure accuracy on anonymous vs. named works
- [ ] Document confidence scores

### Phase 3D: Publication/Demo
- [ ] Write Substack post: "Can AI Identify Anonymous 18th-Century Authors?"
- [ ] Build demo showing attribution process
- [ ] Create Twitter thread with visualizations
- [ ] Prepare findings for DH conference

---

## Commercial Implications

### Expanded Use Cases

1. **Academic Integrity** - Multi-author detection in essays
2. **Legal/Forensic** - Anonymous document attribution
3. **Historical Research** - Disputed authorship resolution
4. **Publishing** - Ghostwriting detection

### Market Validation

- Larger corpus (13 authors) = more impressive demo
- Anonymous attribution = clear value prop
- Gothic genre = popular (Frankenstein, Dracula adjacent)
- Gender analysis angle = DH research appeal

---

## File Locations

```
/Users/danielwaterfield/Documents/DigHums/
├── CharlotteSmith/Emmeline.txt
├── ElizaHaywood/LoveInExcess.txt
├── ClaraReeve/TheOldEnglishBaron.txt
├── HoraceWalpole/CastleOfOtranto.txt
├── MGLewis/TheMonk.txt
├── WilliamBeckford/Vathek.txt
└── AnnRadcliffe/
    ├── CastlesOfAthlinAndDunbayne.txt  ⭐ Anonymous
    ├── TheRomanceOfTheForest.txt
    ├── TheItalianVol1.txt
    └── TheItalianVol2.txt
```

See `CORPUS_CATALOG.md` for complete author/text listing.

---

## Sources Consulted

- [Project Gutenberg Gothic Fiction Bookshelf](https://www.gutenberg.org/ebooks/bookshelf/39)
- [Charlotte Smith on Project Gutenberg](https://www.gutenberg.org/ebooks/author/41281)
- [Ann Radcliffe Works](https://www.gutenberg.org/ebooks/author/1147)
- [A Sicilian Romance - Wikipedia](https://en.wikipedia.org/wiki/A_Sicilian_Romance)
- [The Old English Baron on Project Gutenberg](https://www.gutenberg.org/ebooks/5182)
- [University of Pennsylvania Digital Library](https://digital.library.upenn.edu/women/radcliffe/athlin/athlin.html)

---

**Status:** ✅ Corpus expansion complete | Ready for preprocessing
