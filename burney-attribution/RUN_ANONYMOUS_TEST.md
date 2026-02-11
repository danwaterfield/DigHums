# Running the Anonymous Attribution Test

## What This Tests

**Question:** Can BERT identify authors from their anonymous works?

We test 3 works that were published anonymously:
1. **Burney's Evelina (1778)** - "By a Lady"
2. **Radcliffe's A Sicilian Romance (1790)** - "By the Authoress of..."
3. **Edgeworth's Castle Rackrent (1800)** - Anonymous

These authors are ALREADY in the 7-author training set, so we're testing:
**Does the model learn authorial signature that transcends publication attribution?**

## Current Status

✅ Test script created: `scripts/test_anonymous_attribution.py`
❌ Model needed: `models/bert_authorship/final/`

## Options to Get Model

### Option A: Retrain on Google Colab (RECOMMENDED - 30 min)
```bash
# 1. Check if Colab usage limit reset
# 2. Upload burney_colab_data.zip to Colab
# 3. Run: python scripts/train_bert.py
# 4. Download model from Colab to local
```

**Pros:** Fast (30 min), free GPU, already worked once
**Cons:** Depends on Colab availability

### Option B: Train Locally (5+ hours)
```bash
cd /Users/danielwaterfield/Documents/DigHums/burney-attribution
python scripts/train_bert.py
```

**Pros:** No dependencies, can run overnight
**Cons:** Very slow on M1 (we killed this before after 5 hours with no progress)

### Option C: Use Smaller Model for Quick Test
Could use a pre-trained BERT and fine-tune for 1 epoch locally just to get SOME results quickly.

## Once Model Is Ready

Simply run:
```bash
python scripts/test_anonymous_attribution.py
```

This will:
- Load the 3 anonymous works
- Chunk them into 512-token segments
- Run BERT predictions on each chunk
- Calculate accuracy, confidence scores
- Show whether model identifies correct authors
- Save results to `results/anonymous_attribution_test.json`

## Expected Results

**If it works well (>80% accuracy):**
- Strong evidence that BERT learns authorial signature
- Great blog post: "AI Successfully Identifies Anonymous 18th-Century Authors"
- Validates the approach for expanded corpus

**If it's mediocre (50-80% accuracy):**
- Still interesting - shows challenges of anonymous attribution
- Can analyze what the model learned instead
- Blog post: "What Happens When AI Tries to Unmask Anonymous Authors"

**If it fails (<50% accuracy):**
- Important negative result
- Might indicate model memorized content not style
- Would inform how we expand/improve

## Next Steps After Test

Regardless of results:
1. Write up findings in Substack post
2. Share on Twitter/LinkedIn with visualizations
3. Then expand to 13-author corpus properly
4. Retrain and do rigorous version for paper

---

**Current blocker:** Need trained model
**Recommendation:** Try Colab again (quickest path to results)
