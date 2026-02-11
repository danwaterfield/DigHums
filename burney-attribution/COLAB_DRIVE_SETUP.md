# Google Drive + Colab Training Setup

**Problem:** Uploading/downloading to Colab is slow (5-10 min per run)

**Solution:** Keep everything in Google Drive permanently, train directly from/to Drive

---

## One-Time Setup (10 minutes)

### 1. Upload Data to Google Drive

```bash
# On your Mac, zip the data
cd /Users/danielwaterfield/Documents/DigHums/burney-attribution
zip -r burney_colab_data.zip data/bert_data/

# Upload burney_colab_data.zip to your Google Drive root
# (via web interface or Google Drive app)
```

### 2. Open Notebook in Colab

1. Upload `colab_drive_training.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Runtime → Change runtime type → GPU (T4)

### 3. Run the Notebook

The notebook will:
- Mount your Google Drive
- Extract data to Drive (first time only - persists!)
- Train model
- Save model directly to Drive
- **No upload/download needed!**

---

## Every Training Run After Setup

```python
# Just run the notebook cells in order:
1. Mount Drive (instant)
2. Skip extraction (data already there)
3. Load data from Drive (fast)
4. Train (30 min)
5. Model saves to Drive (instant)
```

**Total time:** ~30 min (vs 45+ min with upload/download)

---

## Accessing Your Model

### Option A: From Another Colab Notebook (FASTEST)

```python
from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    '/content/drive/MyDrive/burney_models/bert_authorship/final'
)
```

### Option B: Download to Mac

1. Go to Google Drive web interface
2. Navigate to `burney_models/bert_authorship/final/`
3. Right-click folder → Download
4. Much faster than Colab direct download!

---

## File Locations in Drive

```
Google Drive/
├── burney_colab_data.zip           # Original data zip (keep for reference)
├── burney_data/                    # Extracted data (persists)
│   └── data/bert_data/
│       ├── chunked_datasets/
│       └── label_mapping.json
└── burney_models/                  # Models directory
    └── bert_authorship/
        ├── checkpoint-XXX/         # Training checkpoints
        ├── checkpoint-YYY/
        └── final/                  # ← Final model (download this)
            ├── model.safetensors
            ├── config.json
            └── tokenizer files
```

---

## Benefits

✅ **No upload** - Data stays in Drive permanently
✅ **No download** - Access model from Drive on any device
✅ **Faster iterations** - Save 10-15 min per training run
✅ **Persistent storage** - Data survives Colab session resets
✅ **Easy sharing** - Just share Drive folder link

---

## Troubleshooting

**"Data not found in Drive"**
- Make sure you uploaded `burney_colab_data.zip` to Drive root
- Run the extraction cell (cell 2)

**"No GPU detected"**
- Runtime → Change runtime type → Select "GPU"
- Check GPU is available: `!nvidia-smi`

**"Drive I/O slow"**
- First time: Extracting from zip is one-time cost
- After that: Loading from Drive is fast (cached)

**"Out of Drive storage"**
- Model + data = ~500 MB total
- Free Drive tier = 15 GB (plenty of space)
- Can delete checkpoints to save space (keep only `final/`)

---

## Next Steps

After training completes:

1. **Test on anonymous works** (optional, in Colab):
   - Upload `test_anonymous_attribution.py` to Colab
   - Point it to Drive model path
   - Run tests directly in Colab

2. **Download to local** (if needed for demos):
   - Download `final/` folder from Drive
   - Place in `burney-attribution/models/bert_authorship/final/`
   - Run local tests: `python scripts/test_anonymous_attribution.py`

3. **Train 13-author model**:
   - Prepare new data with 13 authors
   - Upload to Drive
   - Retrain using same notebook
   - Compare results!

---

**Time savings per run:** ~10-15 minutes
**Setup effort:** 10 minutes one-time
**Payoff after:** 1 training run ✅
