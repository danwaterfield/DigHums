# Project Roadmap

## Completed Phases

### ✅ Phase 1: Data Acquisition & Preparation
- Assembled 18-text corpus (7 authors, 1740-1814)
- Preprocessing pipeline for Project Gutenberg texts
- Metadata generation and EDA
- **Status:** Complete

### ✅ Phase 2: Deep Learning Implementation
- Burrows' Delta baseline (80% accuracy)
- Stratified data splitting (all 7 authors in train/val/test)
- BERT fine-tuning for authorship attribution
- **Result:** 99.9% test accuracy
- **Status:** Complete

---

## Phase 3: Interpretability & Feature Analysis

**Goal:** Understand WHAT linguistic features BERT learns for authorship attribution

**Motivation:**
- Transform black-box model into transparent, interpretable system
- Bridge computational methods with literary scholarship
- Identify concrete stylistic markers for each author
- Validate that model learns style, not content

### 3.1 Attention Analysis
**Tasks:**
- [ ] Extract attention weights from trained BERT model
- [ ] Visualize attention patterns for sample passages
- [ ] Identify which tokens/positions receive highest attention per author
- [ ] Compare attention patterns across authors

**Deliverables:**
- `scripts/analyze_attention.py` - Attention extraction & visualization
- `results/attention_analysis.md` - Findings on attention patterns
- `outputs/figures/attention_*.png` - Visualizations

**Expected insights:** Do attention heads focus on function words (like Delta)?
Or narrative markers (she said, he replied)? Or content words?

### 3.2 Feature Importance Analysis
**Tasks:**
- [ ] Extract BERT embeddings for each author's test samples
- [ ] Perform dimensionality reduction (PCA, t-SNE) on embeddings
- [ ] Visualize author clusters in embedding space
- [ ] Identify most discriminative features using gradient-based methods

**Deliverables:**
- `scripts/feature_analysis.py` - Embedding extraction & clustering
- `outputs/figures/embedding_space.png` - t-SNE visualization
- `results/discriminative_features.txt` - Top features per author

**Expected insights:** Are authors linearly separable? Which linguistic
dimensions separate them?

### 3.3 Token-Level Attribution
**Tasks:**
- [ ] Implement token-level saliency scoring (integrated gradients or LIME)
- [ ] For each author, extract the 100 most discriminative tokens
- [ ] Compare to traditional stylometric features (function words, punctuation)
- [ ] Generate example passages with token-level attribution scores

**Deliverables:**
- `scripts/token_attribution.py` - Saliency scoring implementation
- `results/author_signatures.md` - Literary analysis of findings
- `outputs/figures/saliency_examples.png` - Annotated passages

**Expected insights:** Does BERT learn known stylistic markers (e.g., Austen's
use of "must" and "might", Burney's "Dear Sir")?

### 3.4 Comparative Analysis: BERT vs. Delta
**Tasks:**
- [ ] Re-implement Burrows' Delta with same train/test split
- [ ] Extract Delta's most discriminative function words
- [ ] Compare BERT's learned features with Delta's features
- [ ] Analyze where each method succeeds/fails

**Deliverables:**
- `scripts/delta_baseline.py` - Updated Delta implementation
- `results/bert_vs_delta.md` - Comparative analysis
- `outputs/figures/method_comparison.png` - Feature overlap visualization

**Expected insights:** Does BERT learn traditional features + semantic features?
Or completely different patterns?

**Phase 3 Timeline:** 1-2 weeks
**Phase 3 Output:** Publication-ready interpretability study

---

## Phase 4: Gender & Authorship Analysis

**Goal:** Analyze gendered patterns in 18th-century authorial style

**Motivation:**
- 3 female authors (Austen, Burney, Edgeworth) vs. 4 male
- Investigate whether BERT learns gendered linguistic patterns
- Connect to literary scholarship on women's writing
- High-impact DH research question

### 4.1 Binary Gender Classification
**Tasks:**
- [ ] Train separate BERT model for male vs. female classification
- [ ] Evaluate accuracy of gender prediction
- [ ] Analyze what features distinguish male/female authors

**Deliverables:**
- `scripts/train_gender_model.py`
- `results/gender_classification.txt`

### 4.2 Gendered Feature Analysis
**Tasks:**
- [ ] Extract most discriminative tokens for female authors
- [ ] Compare female author features to known patterns from lit scholarship
  - Domestic spaces, epistolary conventions, emotional language?
- [ ] Test whether male authors have distinct patterns
  - Adventure, external action, mock-epic devices?

**Deliverables:**
- `results/gendered_features.md` - Literary analysis
- `outputs/figures/gender_features.png`

### 4.3 Within-Gender Variation
**Tasks:**
- [ ] Cluster female authors (Austen, Burney, Edgeworth)
- [ ] Cluster male authors (Fielding, Richardson, Smollett)
- [ ] Measure within-gender vs. between-gender variation

**Expected findings:** Are gender differences larger than individual differences?

**Phase 4 Timeline:** 1 week
**Phase 4 Output:** DH conference paper draft

---

## Phase 5: Burney Evolution Analysis

**Goal:** Track Frances Burney's stylistic development across 36 years

**Motivation:**
- Only author with 4+ works (1778-1814)
- Spans major life events (marriage, exile, War of 1812)
- Test whether style changes over career
- Rigorous leave-one-work-out validation

### 5.1 Leave-One-Work-Out Cross-Validation
**Tasks:**
- [ ] Train 4 models, each leaving out one Burney novel
- [ ] Test if excluded novel is still identified as Burney
- [ ] Measure confidence scores for each work

**Deliverables:**
- `scripts/burney_loocv.py`
- `results/burney_validation.txt`

### 5.2 Temporal Style Analysis
**Tasks:**
- [ ] Plot embedding similarity across 4 novels
- [ ] Identify features that change over time
- [ ] Correlate with biographical/historical context

**Works:**
- Evelina (1778) - Age 26, debut
- Cecilia (1782) - Age 30, fame
- Camilla (1796) - Age 44, after marriage/exile
- The Wanderer (1814) - Age 62, post-Napoleonic

**Expected findings:** Does she become more conservative? More experimental?

**Phase 5 Timeline:** 1 week
**Phase 5 Output:** Burney-focused publication

---

## Phase 6: Interactive Demonstration

**Goal:** Create public-facing tool for exploring authorship attribution

### 6.1 Web Interface
**Tasks:**
- [ ] Build Gradio/Streamlit app
- [ ] Allow users to paste 18th-century passages
- [ ] Display author prediction + confidence scores
- [ ] Show attention visualization and key features

**Deliverables:**
- `app.py` - Web interface
- Deployed to Hugging Face Spaces (free hosting)

### 6.2 Educational Materials
**Tasks:**
- [ ] Create tutorial notebook explaining the method
- [ ] Write blog post for DH audience
- [ ] Prepare conference presentation slides

**Phase 6 Timeline:** 1 week
**Phase 6 Output:** Public demo + outreach materials

---

## Phase 7: Publication Preparation

**Goal:** Prepare findings for academic publication

### 7.1 Manuscript Drafting
**Target venues:**
- *Digital Scholarship in the Humanities* (Oxford)
- *Digital Humanities Quarterly*
- DH conference (ACH, ADHO)

**Sections:**
1. Introduction: 18th-century authorship + neural methods
2. Related Work: Stylometry + transformers
3. Methodology: Data, BERT fine-tuning, stratified splitting
4. Results: 99.9% accuracy
5. Interpretability: What BERT learns (Phase 3)
6. Gender Analysis: Gendered patterns (Phase 4)
7. Case Study: Burney evolution (Phase 5)
8. Discussion: Implications for DH

### 7.2 Code & Data Release
**Tasks:**
- [ ] Add comprehensive documentation
- [ ] Create requirements.txt with exact versions
- [ ] Write reproducibility guide
- [ ] Archive on Zenodo with DOI

**Phase 7 Timeline:** 2 weeks
**Phase 7 Output:** Submitted manuscript

---

## Optional Extensions (Phase 8+)

### Cross-Period Transfer
- Test on 19th-century authors
- Measure domain shift

### Character Voice Attribution
- Distinguish character voices within novels
- Especially interesting for epistolary works

### Disputed Attribution
- Test on anonymous 18th-century texts
- Real scholarly contribution if successful

### Multilingual Extension
- French 18th-century novels
- Compare cross-linguistic patterns

---

## Key Decisions to Make

### For Phase 3:
1. **Which interpretability method?**
   - Attention visualization (built-in)
   - Integrated gradients (most rigorous)
   - LIME (most interpretable for non-experts)
   - **Recommendation:** Start with attention, add LIME if needed

2. **Scope of analysis?**
   - All authors or focus on contrasting pairs (e.g., Austen vs. Burney)?
   - **Recommendation:** All 7 for completeness, but highlight 2-3 interesting contrasts

3. **Literary scholar collaboration?**
   - Would significantly strengthen Phase 3-5
   - **Recommendation:** Reach out to 18th-century lit specialist at your institution

### For Phase 4:
1. **Risk of overgeneralization?**
   - Only 3 female authors (n=3 is small)
   - May reflect individual style more than gender
   - **Mitigation:** Frame as exploratory, emphasize limitations

2. **Theoretical framework?**
   - Need to engage with gender & language scholarship
   - **Recommendation:** Read Nancy Armstrong, Mary Poovey on women's writing

---

## Success Metrics

**Phase 3:**
- ✅ Identified 10+ interpretable features per author
- ✅ Published attention visualizations
- ✅ Comparative analysis with traditional methods

**Phase 4:**
- ✅ Gender classification >70% accuracy (if patterns exist)
- ✅ Identified 3+ gendered linguistic patterns
- ✅ Connected findings to lit scholarship

**Phase 5:**
- ✅ Leave-one-out validation >90% for Burney
- ✅ Documented style evolution with examples

**Overall:**
- ✅ 1-2 DH publications
- ✅ Public demo with >100 users
- ✅ Reproducible, well-documented codebase

---

## Timeline Summary

- **Phase 3:** 1-2 weeks (Interpretability)
- **Phase 4:** 1 week (Gender)
- **Phase 5:** 1 week (Burney evolution)
- **Phase 6:** 1 week (Demo)
- **Phase 7:** 2 weeks (Publication)

**Total:** 6-8 weeks to completion

---

**Questions before starting Phase 3?**
- Do you have access to literature scholars for collaboration?
- Any specific interpretability methods you prefer?
- Planning to publish or just for a course?
