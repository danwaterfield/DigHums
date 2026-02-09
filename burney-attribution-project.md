# Burney Authorship Attribution Project

**Goal**: Build a deep learning system for identifying Frances Burney's stylistic fingerprint in 18th-century texts using ECCO-BERT and modern NLP techniques.

**Timeline**: 8-12 weeks part-time

---

## Phase 1: Data Acquisition & Preparation
**Duration**: Weeks 1-2  
**Status**: Not Started

### Objectives
- Assemble clean, labeled corpus of Burney and contemporary authors
- Create preprocessing pipeline for both plain text and TEI-XML sources
- Establish metadata management system

### Tasks

#### 1.1 Burney Corpus Collection
- [ ] Download Burney's major novels:
  - [ ] *Evelina* (1778)
  - [ ] *Cecilia* (1782) 
  - [ ] *Camilla* (1796)
  - [ ] *The Wanderer* (1814)
- [ ] Locate and download letters/diaries if available digitized
- [ ] Sources to check:
  - Project Gutenberg
  - Oxford Text Archive (TEI-XML)
  - ECCO (if accessible)
  - Internet Archive

#### 1.2 Comparison Authors Collection
**Target**: 5-10 contemporary authors (mix male/female, genres)

Suggested authors:
- [ ] Samuel Richardson (*Pamela*, *Clarissa*)
- [ ] Henry Fielding (*Tom Jones*, *Joseph Andrews*)
- [ ] Tobias Smollett (*Humphry Clinker*)
- [ ] Eliza Haywood (*Love in Excess*)
- [ ] Charlotte Lennox (*The Female Quixote*)
- [ ] Ann Radcliffe (*The Mysteries of Udolpho*)
- [ ] Maria Edgeworth (*Belinda*, *Castle Rackrent*)
- [ ] Sarah Fielding (*David Simple*)

#### 1.3 Test Cases (Anonymous/Disputed Works)
- [ ] Identify 2-3 anonymous/pseudonymous works from the period
- [ ] Research existing scholarly debates about attribution
- [ ] Document hypothesized authors

#### 1.4 Preprocessing Pipeline
- [ ] Write TEI-XML to text converter
  ```python
  # Handle:
  # - Extract from <body> only (skip front matter)
  # - Resolve line-break hyphens
  # - Optionally preserve dialogue tags
  # - Clean up OCR errors
  ```
- [ ] Write plain text cleaner
  ```python
  # Handle:
  # - Strip Project Gutenberg headers/footers
  # - Normalize quotation marks
  # - Handle chapter breaks
  # - Preserve sentence/paragraph boundaries
  ```
- [ ] Create volume concatenation script (for multi-volume novels)

#### 1.5 Directory Structure
```
burney-attribution/
├── data/
│   ├── raw/
│   │   ├── burney/
│   │   ├── richardson/
│   │   ├── fielding/
│   │   └── ...
│   ├── processed/
│   │   └── (cleaned text files)
│   └── metadata.csv
├── notebooks/
├── scripts/
├── models/
└── outputs/
```

#### 1.6 Metadata Management
Create `metadata.csv` with columns:
- `author` (standardized name)
- `title` 
- `year` (publication year)
- `genre` (novel, letters, diary)
- `volume` (if multi-volume)
- `word_count`
- `source` (Gutenberg, OTA, etc.)
- `file_path`
- `notes`

#### 1.7 EDA Notebook
- [ ] Basic corpus statistics (word counts, vocabulary size)
- [ ] Distribution of authors, genres, years
- [ ] Sample text inspection
- [ ] Identify any obvious quality issues

### Deliverables
- ✅ Clean text files for all works
- ✅ `metadata.csv` with complete corpus information
- ✅ Preprocessing scripts (documented)
- ✅ EDA notebook with visualizations
- ✅ Data quality report

### Success Criteria
- At least 4 Burney works collected
- At least 5 comparison authors with 2+ works each
- All texts cleaned and validated
- No obvious OCR errors in samples
- Metadata complete and accurate

---

## Phase 2: Baseline Implementation
**Duration**: Weeks 3-4  
**Status**: Not Started

### Objectives
- Set up ML environment and tooling
- Implement traditional stylometry baseline
- Fine-tune ECCO-BERT for authorship classification
- Establish evaluation framework

### Tasks

#### 2.1 Environment Setup
- [ ] Create Python virtual environment
  ```bash
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  ```
- [ ] Install dependencies:
  ```bash
  pip install transformers datasets torch scikit-learn pandas numpy
  pip install matplotlib seaborn jupyter
  pip install wandb  # for experiment tracking
  ```
- [ ] Set up Weights & Biases account (optional but recommended)
- [ ] Configure GPU access (local or cloud)

#### 2.2 Traditional Stylometry Baseline
Implement Burrows' Delta or similar:
- [ ] Extract features:
  - Most frequent words (MFW)
  - Function word frequencies
  - Punctuation patterns
  - Average sentence length
- [ ] Calculate Delta distances between texts
- [ ] Classify using k-NN or similar
- [ ] Create baseline results notebook

#### 2.3 Data Preparation for BERT
- [ ] Create train/validation/test split
  - Strategy: Split by **work**, not by chunk (avoid data leakage)
  - Proportions: 70/15/15
  - Stratify by author
- [ ] Tokenize texts with ECCO-BERT tokenizer
- [ ] Create chunks (512 tokens each)
- [ ] Save as HuggingFace datasets

#### 2.4 ECCO-BERT Fine-tuning
- [ ] Load pre-trained ECCO-BERT from HuggingFace
  ```python
  from transformers import AutoModelForSequenceClassification, AutoTokenizer
  
  model_name = "Brendan/ecco_bert"  # Verify exact name on HF
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(
      model_name, 
      num_labels=len(author_list)
  )
  ```
- [ ] Set up training with HuggingFace Trainer:
  - Learning rate: 2e-5
  - Batch size: 8-16 (depending on GPU)
  - Epochs: 3-5
  - Warmup steps: 500
- [ ] Implement early stopping
- [ ] Track metrics: accuracy, F1 per author, confusion matrix

#### 2.5 Evaluation
- [ ] Whole-work accuracy (aggregate chunk predictions)
- [ ] Per-author F1 scores
- [ ] Confusion matrix analysis
- [ ] Comparison with baseline

#### 2.6 Initial Experiments
Document in notebook:
- [ ] Effect of chunk size (256 vs 512 vs 1024 tokens)
- [ ] Effect of overlap (no overlap vs 50% vs 75%)
- [ ] Learning rate sensitivity

### Deliverables
- ✅ Working training script
- ✅ Saved model checkpoint (best performing)
- ✅ Evaluation notebook with results
- ✅ Baseline comparison (ECCO-BERT vs traditional methods)
- ✅ Training logs and metrics (WandB dashboard or local)

### Success Criteria
- ECCO-BERT achieves >70% accuracy on test set
- Outperforms traditional stylometry baseline
- Clear per-author performance metrics
- Reproducible training pipeline

---

## Phase 3: Passage-Level Attribution
**Duration**: Weeks 5-6  
**Status**: Not Started

### Objectives
- Move from whole-work classification to passage-level detection
- Implement sliding window inference
- Develop aggregation strategies for uncertain predictions
- Create visualization tools

### Tasks

#### 3.1 Sliding Window Implementation
- [ ] Implement sliding window over any text:
  ```python
  def sliding_window(text, window_size=512, stride=256):
      """
      Args:
          text: Input text
          window_size: tokens per window
          stride: step size between windows
      Returns:
          List of (chunk_text, start_pos, end_pos) tuples
      """
  ```
- [ ] Handle edge cases (text shorter than window, final partial window)

#### 3.2 Passage Classification
- [ ] Run model inference on each window
- [ ] Collect per-window predictions and confidence scores
- [ ] Store results with position information

#### 3.3 Aggregation Strategies
Experiment with multiple approaches:
- [ ] **Majority vote**: Most common predicted author
- [ ] **Average probability**: Mean confidence across windows
- [ ] **Weighted vote**: Weight by confidence scores
- [ ] **Threshold-based**: Only count predictions above confidence threshold (e.g., 0.7)
- [ ] Compare strategies on validation set

#### 3.4 Visualization Tools
- [ ] Create heatmap showing "Burney-ness" across a novel:
  ```python
  # X-axis: position in text
  # Y-axis: confidence that passage is by Burney
  # Color: predicted author
  ```
- [ ] Implement text highlighting (color-code passages by attribution)
- [ ] Create summary statistics dashboard

#### 3.5 Validation Experiments
- [ ] **Known Burney test**: Run on held-out Burney novel
  - What % of passages correctly identified?
  - Where does model fail?
- [ ] **Known non-Burney test**: Run on held-out contemporary novel
  - False positive rate?
  - Which authors get confused with Burney?
- [ ] **Mixed-voice test**: Epistolary novels (multiple character voices)
  - Does model attribute different character voices differently?

#### 3.6 Confidence Calibration
- [ ] Plot confidence vs accuracy
- [ ] Determine reliable confidence thresholds
- [ ] Create ROC curves for different threshold values

### Deliverables
- ✅ Passage-level inference pipeline
- ✅ Visualization notebook with heatmaps
- ✅ Aggregation strategy comparison
- ✅ Validation results on known texts
- ✅ Recommended confidence thresholds

### Success Criteria
- Can accurately identify Burney passages in held-out work (>80% precision)
- False positive rate on non-Burney texts <20%
- Clear visualizations that show where model is confident/uncertain
- Documented aggregation strategy with justification

---

## Phase 4: Interpretability & Analysis
**Duration**: Weeks 7-8  
**Status**: Not Started

### Objectives
- Understand what the model has learned
- Extract and analyze attention patterns
- Compare with traditional literary-critical understanding of Burney's style
- Identify systematic errors

### Tasks

#### 4.1 Attention Analysis
- [ ] Extract attention weights for sample passages:
  ```python
  # Use model attention outputs
  # Visualize which tokens the model focuses on
  # For Burney vs non-Burney passages
  ```
- [ ] Implement attention rollout or integrated gradients
- [ ] Identify most discriminative tokens/patterns

#### 4.2 Feature Importance
- [ ] Use SHAP or LIME for token-level importance:
  ```python
  # Which words push prediction toward "Burney"?
  # Which words push toward other authors?
  ```
- [ ] Aggregate across multiple examples
- [ ] Create ranked lists of discriminative features

#### 4.3 Qualitative Analysis
Compare model findings with known Burney scholarship:
- [ ] Research existing literary analysis of Burney's style
  - Vocabulary choices
  - Sentence structure
  - Use of indirect discourse
  - Characteristic phrases
- [ ] Check if model patterns align with scholarly observations
- [ ] Document novel patterns the model discovered

#### 4.4 Error Analysis
- [ ] Examine false positives (non-Burney classified as Burney):
  - Which authors are most confused?
  - What do misclassified passages have in common?
- [ ] Examine false negatives (Burney classified as non-Burney):
  - Are they from specific works/periods?
  - Specific narrative situations (e.g., male character dialogue)?
- [ ] Create error taxonomy

#### 4.5 Cross-Genre Robustness
- [ ] If you have Burney's letters/diaries:
  - Train on novels only
  - Test on letters
  - Does model generalize across genres?
- [ ] Document genre-specific patterns

#### 4.6 Temporal Analysis
- [ ] Does model performance vary by Burney's career stage?
  - *Evelina* (1778) vs *The Wanderer* (1814)
  - Did her style evolve?
- [ ] Compare early vs late works

### Deliverables
- ✅ Attention visualization notebook
- ✅ Feature importance analysis
- ✅ Comparison with literary scholarship (brief report)
- ✅ Error analysis taxonomy
- ✅ Cross-genre/temporal analysis results

### Success Criteria
- Clear explanation of what model uses to identify Burney
- At least some alignment with known stylistic features
- Comprehensive error analysis
- Documentation of where model succeeds/fails

---

## Phase 5: Applied Case Study
**Duration**: Weeks 9-10  
**Status**: Not Started

### Objectives
- Apply model to real attribution question
- Compare results with existing scholarship
- Assess practical utility of the approach

### Tasks

#### 5.1 Select Test Case(s)
Options:
- [ ] Anonymous reviews Burney might have written
- [ ] Disputed collaborative works
- [ ] Sections of novels with questioned authorship
- [ ] Other 18th-century anonymous works where Burney is a candidate

#### 5.2 Run Full Pipeline
For each test case:
- [ ] Preprocess text
- [ ] Run passage-level classification
- [ ] Generate visualizations
- [ ] Calculate aggregate attribution scores
- [ ] Document confidence levels

#### 5.3 Scholarly Comparison
- [ ] Research existing attribution arguments for test case
- [ ] Document evidence scholars have used:
  - Internal evidence (style, vocabulary, themes)
  - External evidence (letters, publication history)
- [ ] Compare model findings with scholarly consensus/debate

#### 5.4 Critical Assessment
- [ ] Does model add new evidence?
- [ ] Does it confirm or contradict existing theories?
- [ ] What are the limitations?
- [ ] How confident should we be in results?

#### 5.5 Write-up
Create case study report:
- [ ] Background on the attribution question
- [ ] Methodology (brief)
- [ ] Results (with visualizations)
- [ ] Comparison with existing scholarship
- [ ] Interpretation and caveats
- [ ] Conclusion

### Deliverables
- ✅ Complete analysis of 1-2 test cases
- ✅ Visualizations showing model's attribution
- ✅ Comparison with scholarly arguments
- ✅ Case study report (3-5 pages)

### Success Criteria
- Rigorous application of method to real question
- Fair comparison with traditional scholarship
- Honest assessment of strengths/limitations
- Results that either:
  - Support existing theory with new evidence
  - Challenge existing theory with compelling alternative
  - Add nuance to ongoing debate

---

## Phase 6: Polish & Release
**Duration**: Weeks 11-12  
**Status**: Not Started

### Objectives
- Clean up code for public release
- Create documentation
- Package model for reuse
- Write public-facing explanation

### Tasks

#### 6.1 Code Cleanup
- [ ] Refactor scripts into modular functions
- [ ] Add docstrings to all functions
- [ ] Create `requirements.txt`:
  ```
  transformers>=4.30.0
  torch>=2.0.0
  datasets>=2.12.0
  scikit-learn>=1.2.0
  pandas>=2.0.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
  ```
- [ ] Add configuration file for hyperparameters
- [ ] Create command-line interface (optional):
  ```bash
  python classify.py --input text.txt --model path/to/model
  ```

#### 6.2 Documentation
- [ ] Write comprehensive README.md:
  - Project overview
  - Installation instructions
  - Quick start guide
  - Usage examples
  - Citation information
- [ ] Create Jupyter notebook tutorial:
  - Walk through entire pipeline
  - Explain key decisions
  - Show example outputs
- [ ] Add inline code comments

#### 6.3 Model Packaging
- [ ] Upload fine-tuned model to HuggingFace Hub:
  ```python
  model.push_to_hub("your-username/burney-attribution-ecco-bert")
  tokenizer.push_to_hub("your-username/burney-attribution-ecco-bert")
  ```
- [ ] Create model card with:
  - Description
  - Training data
  - Performance metrics
  - Intended use & limitations
  - Citation

#### 6.4 Demo Interface (Optional)
- [ ] Create simple Gradio demo:
  ```python
  import gradio as gr
  
  def classify_passage(text):
      # Run model inference
      # Return predicted author + confidence
      pass
  
  demo = gr.Interface(fn=classify_passage, 
                     inputs="text", 
                     outputs="label")
  demo.launch()
  ```
- [ ] Host on HuggingFace Spaces (free)

#### 6.5 Public Write-up
Create blog post or preprint:
- [ ] Introduction (the problem)
- [ ] Background (18th-century attribution, existing methods)
- [ ] Approach (ECCO-BERT, fine-tuning strategy)
- [ ] Results (key findings, visualizations)
- [ ] Case study (if compelling)
- [ ] Discussion (what worked, what didn't, future directions)
- [ ] Conclusion
- [ ] Publish on:
  - Personal blog
  - Medium
  - ArXiv (if substantial enough)

#### 6.6 GitHub Repository
- [ ] Create public repo with MIT or similar license
- [ ] Organize structure:
  ```
  burney-attribution/
  ├── README.md
  ├── requirements.txt
  ├── LICENSE
  ├── data/
  │   └── README.md  (data sources, not data itself)
  ├── notebooks/
  │   └── tutorial.ipynb
  ├── scripts/
  │   ├── preprocess.py
  │   ├── train.py
  │   ├── evaluate.py
  │   └── classify.py
  ├── models/
  │   └── README.md  (link to HuggingFace)
  └── outputs/
      └── figures/
  ```
- [ ] Add badge links (HuggingFace model, license, etc.)

### Deliverables
- ✅ Clean, documented codebase on GitHub
- ✅ Published model on HuggingFace Hub
- ✅ Tutorial notebook
- ✅ Blog post or preprint
- ✅ Optional: Demo interface

### Success Criteria
- Code is reproducible (someone else can run it)
- Documentation is clear
- Model is accessible for others to use
- Public explanation is understandable to non-specialists
- Proper attribution and licensing

---

## Publication Strategy

### Potential Venues

**DH Conferences/Journals:**
- Digital Humanities (DH) conference
- Computational Humanities Research (CHR)
- Digital Scholarship in the Humanities (journal)
- Literary and Linguistic Computing

**NLP Venues (if methodology is novel):**
- NLP for Historical Texts workshop (at ACL/EMNLP)
- LChange workshop
- Cultural Analytics journal

**18th-Century Studies:**
- The Eighteenth Century: Theory and Interpretation
- Age of Johnson
- (Though these may be less receptive to computational work)

### Paper Outline (If Pursuing Publication)

1. **Introduction**
   - Attribution as scholarly problem
   - Limitations of traditional stylometry
   - Potential of modern NLP

2. **Related Work**
   - Stylometry history (Burrows, Mosteller & Wallace)
   - Deep learning for authorship
   - Prior work on 18th-century texts

3. **Methodology**
   - ECCO-BERT background
   - Fine-tuning approach
   - Evaluation framework

4. **Experiments**
   - Corpus description
   - Baseline comparisons
   - Main results
   - Ablation studies

5. **Interpretability**
   - What the model learned
   - Comparison with literary scholarship

6. **Case Study**
   - Applied attribution example

7. **Discussion**
   - Where method succeeds/fails
   - Limitations
   - Future work

8. **Conclusion**

---

## Risk Register

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ECCO-BERT doesn't work well for this task | Medium | High | Have traditional stylometry as fallback; try RoBERTa or GPT-2 fine-tuned on 18C texts |
| Insufficient training data | Medium | High | Use data augmentation; consider few-shot learning approaches; synthesize additional data from known works |
| Model learns topic/genre, not author | Medium | High | Careful train/test splits; cross-genre validation; attention analysis to verify |
| OCR errors in ECCO degrade performance | Medium | Medium | Manual correction of key texts; OCR post-processing; test on clean Gutenberg texts first |
| GPU access issues | Low | Medium | Use Google Colab free tier; consider AWS/GCP credits; start with smaller models |

### Scholarly Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Results don't align with known scholarship | Medium | Medium | Don't hide disagreements; investigate why; could be novel finding |
| Test case too ambiguous for useful results | Medium | Medium | Choose cases with some existing evidence; be honest about uncertainty |
| DH community skeptical of deep learning | Medium | Low | Emphasize interpretability; compare with traditional methods; acknowledge limitations |
| Can't access necessary texts | Low | High | Use multiple sources (Gutenberg, OTA, Internet Archive, ECCO if available) |

---

## Resources & References

### Tools
- **ECCO-BERT**: https://huggingface.co/Brendan/ecco_bert (verify name)
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Weights & Biases**: https://wandb.ai

### Datasets
- **Project Gutenberg**: https://www.gutenberg.org
- **Oxford Text Archive**: https://ota.bodleian.ox.ac.uk
- **ECCO**: (institutional access required)
- **Internet Archive**: https://archive.org

### Key Papers
- Burrows, John. "'Delta': a Measure of Stylistic Difference and a Guide to Likely Authorship." *Literary and Linguistic Computing* 17.3 (2002): 267-287.
- Stamatatos, Efstathios. "A survey of modern authorship attribution methods." *JASIST* 60.3 (2009): 538-556.
- Kestemont, Mike. "Function words in authorship attribution." *Language Resources and Evaluation* 48.2 (2014): 345-375.
- Recent work on transformers for authorship (search: "BERT authorship attribution")

### Burney Scholarship
- (Add specific works on Burney's style as you encounter them)
- Doody, Margaret Anne. *Frances Burney: The Life in the Works*
- Epstein, Julia. *The Iron Pen: Frances Burney and the Politics of Women's Writing*

---

## Progress Tracking

### Weekly Check-ins
Every Sunday, review:
- [ ] What got done this week
- [ ] What's blocking progress
- [ ] What's the plan for next week
- [ ] Any pivots needed

### Milestones
- [ ] **Week 2**: Data collected and cleaned
- [ ] **Week 4**: Baseline model trained
- [ ] **Week 6**: Passage attribution working
- [ ] **Week 8**: Interpretability analysis complete
- [ ] **Week 10**: Case study finished
- [ ] **Week 12**: Code/model/write-up published

---

## Next Actions

### This Weekend (If Starting Now)
1. Create project directory structure
2. Download 3-4 Burney novels from Gutenberg
3. Download 3-4 contemporary author novels
4. Write basic text cleaning script
5. Get first accuracy number from quick ECCO-BERT fine-tune
   - Even on small data, this validates the approach

### This Week
1. Complete Phase 1.1-1.3 (data collection)
2. Set up development environment
3. Write preprocessing scripts
4. Create metadata.csv

---

## Notes & Ideas

*(Space for freeform notes as project progresses)*

### Open Questions
- Should I train separate models for novels vs letters vs diaries?
- How to handle anonymous reviews (very short texts)?
- Is there value in author clustering before classification?
- Could I use this to study Burney's stylistic evolution over time?

### Future Extensions
- Extend to other 18th-century authors
- Build multi-task model (author + gender + genre + time period)
- Analyze influence (who influenced whom stylistically?)
- Create "Burney-ness" score for any 18th-century text
- Apply to anonymous periodical essays (Spectator, Rambler, etc.)

---

**Last Updated**: [Current Date]  
**Project Status**: Planning Phase
