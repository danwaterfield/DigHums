# Google Colab Training Instructions

## Quick Start (5 minutes setup, 1-2 hours training)

### Option 1: Upload Data to Google Drive (Recommended)

1. **Package the data locally:**
   ```bash
   cd burney-attribution
   chmod +x scripts/package_for_colab.sh
   ./scripts/package_for_colab.sh
   ```

2. **Upload to Google Drive:**
   - Upload `burney_colab_data.zip` to your Google Drive
   - Note the folder path (e.g., `MyDrive/burney-data/`)

3. **Open the notebook in Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Upload notebook
   - Upload `notebooks/train_bert_colab.ipynb`

4. **Set GPU runtime:**
   - Runtime → Change runtime type → **GPU (T4 or better)**
   - Save

5. **Run the notebook:**
   - In the "Mount Google Drive" cell, update the path to your zip file
   - Run all cells (Runtime → Run all)
   - Training will take 1-2 hours

6. **Download results:**
   - The final cell will download `bert_model.zip`
   - Extract locally to `burney-attribution/models/`

### Option 2: Direct Upload (Simpler but slower)

1. **Open Colab notebook:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Upload notebook
   - Upload `notebooks/train_bert_colab.ipynb`

2. **Set GPU runtime:**
   - Runtime → Change runtime type → GPU
   - Save

3. **Upload data directly:**
   - Run the notebook until the "Upload Data Directly" cell
   - Click "Choose Files" and upload `burney_colab_data.zip`
   - Continue running cells

## What Gets Trained

- **Model**: bert-base-uncased (110M parameters)
- **Data**: 7,787 training chunks, 2,860 validation chunks
- **Authors**: 7 (Burney, Austen, Fielding, Richardson, Smollett, Radcliffe, Edgeworth)
- **Epochs**: 3 (with early stopping)
- **Time**: ~1-2 hours on T4 GPU

## Expected Results

- **Validation Accuracy**: 85-95% (estimated)
- **Test Accuracy**: Goal is to beat 80% baseline
- **Output**: Trained model + metrics in JSON

## Cost

- **Free tier**: Works fine (T4 GPU, may disconnect after 12 hours)
- **Colab Pro**: $12/month for better GPUs and longer runtime
- **Colab Pro+**: $50/month for A100 access (overkill for this)

## Troubleshooting

**"Runtime disconnected"**: Re-run from the start. The training will resume from the last saved checkpoint if you set up checkpointing.

**"Out of memory"**: Reduce `per_device_train_batch_size` from 16 to 8 in the training args cell.

**"No GPU available"**: Make sure you selected GPU in Runtime settings.

## After Training

Once complete, you'll have:
- `models/bert_authorship/final/` - Trained model
- `models/bert_authorship/test_results.json` - Test metrics
- `models/bert_authorship/train_metrics.json` - Training history

Extract the downloaded zip to your local `burney-attribution/models/` directory.
