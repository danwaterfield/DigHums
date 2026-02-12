# Azure ML Compute Instance Setup

Quick guide to training your BERT model on Azure ML with persistent storage and experiment tracking.

---

## Why Azure ML Compute Instance?

- ✅ **Persistent storage** - files don't disappear between sessions
- ✅ **No limits** - train as much as you want
- ✅ **Cost-effective** - ~$0.25 per training run (only pay for compute time)
- ✅ **Experiment tracking** - automatic logging of metrics
- ✅ **Familiar tools** - Jupyter or VS Code built-in
- ✅ **Professional** - proper ML workflow for portfolio

---

## Setup (10 minutes)

### 1. Create Azure ML Workspace (if you don't have one)

```bash
# Option A: Azure Portal
# Go to portal.azure.com
# Create Resource → Machine Learning → Create

# Option B: Azure CLI
az ml workspace create \
  --name burney-ml-workspace \
  --resource-group your-rg \
  --location eastus
```

### 2. Create Compute Instance

**Via Portal (Easiest):**
1. Go to https://ml.azure.com
2. Select your workspace
3. Navigate to **Compute** → **Compute instances**
4. Click **+ New**
5. Configure:
   - **Compute name:** `burney-gpu-instance`
   - **Virtual machine type:** GPU
   - **Virtual machine size:** `STANDARD_NC4AS_T4_V3` (T4 GPU, ~$0.51/hour)
     - Or `STANDARD_NC6S_V3` (V100, faster but ~$3.06/hour)
   - **Enable SSH:** Yes (optional, for direct access)
6. Click **Create**
7. Wait 3-5 minutes for provisioning

**Via Azure CLI:**
```bash
az ml compute create \
  --name burney-gpu-instance \
  --type ComputeInstance \
  --size STANDARD_NC4AS_T4_V3 \
  --workspace-name burney-ml-workspace \
  --resource-group your-rg
```

### 3. Open Jupyter or VS Code

Once instance is **Running**:
1. Click on the compute instance name
2. Choose: **Jupyter** or **VS Code** (both work)
3. Opens in browser with GPU access!

---

## Quick Start: Train Your Model

### Option A: Using Notebook (Recommended for first run)

1. **Upload the notebook:**
   - In Jupyter, click Upload
   - Select `azure_ml_training.ipynb` (created below)
   - Run cells in order

2. **Or clone from GitHub:**
   ```bash
   # In Jupyter terminal or VS Code terminal:
   git clone https://github.com/danwaterfield/DigHums.git
   cd DigHums/burney-attribution
   ```

3. **Run training:**
   - Open `azure_ml_training.ipynb`
   - Run all cells
   - Watch experiment tracking in real-time!

### Option B: Using Script (For repeated runs)

```bash
# In terminal (Jupyter or VS Code terminal):
cd ~/cloudfiles/code/DigHums/burney-attribution

# Upload your data (first time only)
# Via Jupyter upload interface or:
# azcopy copy 'local/data/bert_data' 'https://yourstore.blob.core.windows.net/data'

# Run training with Azure ML logging
python scripts/train_bert_azureml.py
```

---

## File Storage Locations

Azure ML Compute Instance has several storage options:

### 1. Local VM Disk (Simple, Fast)
```
/home/azureuser/
├── cloudfiles/
│   ├── code/          # Your code (git repos)
│   └── data/          # Your data
```
**Use for:** Quick iterations, small datasets

### 2. Azure Blob Storage (Persistent, Shareable)
```python
from azure.storage.blob import BlobServiceClient

# Upload data once, use everywhere
blob_service = BlobServiceClient.from_connection_string(conn_str)
blob_client = blob_service.get_blob_client("data", "burney_data.zip")
blob_client.upload_blob(data)
```
**Use for:** Large datasets, team sharing

### 3. Azure Files (Shared Drive)
```bash
# Mount as network drive
sudo mount -t cifs //yourstore.file.core.windows.net/share /mnt/azurefiles
```
**Use for:** Shared models, persistent storage

**Recommendation:** Start with local disk, move to Blob if you need persistence.

---

## Training Workflow

### First Run (Upload Data)

```bash
# In compute instance terminal:
cd ~/cloudfiles/code

# Clone repo
git clone https://github.com/danwaterfield/DigHums.git
cd DigHums/burney-attribution

# Create data directory
mkdir -p data/bert_data

# Upload your prepared data
# Either: Use Jupyter upload interface
# Or: Copy from local via Azure Storage Explorer
# Or: Generate data on the compute instance:
python scripts/prepare_bert_data.py
```

### Every Run After

```bash
cd ~/cloudfiles/code/DigHums/burney-attribution

# Pull latest code
git pull

# Run training (with Azure ML tracking!)
python scripts/train_bert_azureml.py

# Or use notebook
jupyter notebook azure_ml_training.ipynb
```

### Monitoring

- **Real-time:** Check terminal output
- **Metrics:** Go to ml.azure.com → Experiments → View run
- **TensorBoard:** Available in Azure ML Studio

---

## Cost Management

### Pricing
- **T4 GPU (STANDARD_NC4AS_T4_V3):** ~$0.51/hour
- **Training time:** ~30 minutes = **~$0.25 per run**
- **Storage:** ~$0.01/month for < 1 GB

### Cost Optimization

**Stop instance when not using:**
```bash
# Via portal: Click "Stop" on compute instance
# Via CLI:
az ml compute stop --name burney-gpu-instance --workspace-name burney-ml-workspace
```

**Automatic shutdown:**
1. In compute instance settings
2. Enable **Idle shutdown**
3. Set to 15-30 minutes of inactivity
4. Saves money automatically!

**Delete when done (optional):**
```bash
az ml compute delete --name burney-gpu-instance --workspace-name burney-ml-workspace
```
Can recreate anytime - your data in blob storage persists.

---

## Experiment Tracking (Built-in!)

Azure ML automatically tracks:
- ✅ Training metrics (loss, accuracy)
- ✅ Model parameters
- ✅ Training duration
- ✅ GPU utilization
- ✅ Code version (git commit)

**View experiments:**
1. Go to https://ml.azure.com
2. Click **Experiments**
3. Select your run
4. See all metrics, logs, outputs!

**Compare runs:**
- Select multiple experiments
- Click **Compare**
- See side-by-side metrics

---

## Bonus: Deploy Model as API (Optional)

Once trained, deploy as Azure endpoint:

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment

ml_client = MLClient.from_config()

# Register model
model = Model(
    path="./models/bert_authorship/final",
    name="burney-attribution",
    description="BERT authorship attribution - 99.9% accuracy"
)
ml_client.models.create_or_update(model)

# Deploy endpoint (optional - for demo)
endpoint = ManagedOnlineEndpoint(name="burney-attribution-api")
ml_client.online_endpoints.begin_create_or_update(endpoint)
```

**Result:** REST API for your model!
```bash
curl -X POST https://burney-attribution-api.azureml.net/score \
  -H "Authorization: Bearer $KEY" \
  -d '{"text": "It was on the first of April..."}'
```

**Perfect for portfolio demos!**

---

## Troubleshooting

**"No GPU detected"**
```bash
# Check GPU
nvidia-smi

# If not showing, restart compute instance
```

**"Out of memory"**
```python
# Reduce batch size in training script
per_device_train_batch_size=8  # Instead of 16
```

**"Can't access data"**
```bash
# Check file permissions
ls -la ~/cloudfiles/code/DigHums/burney-attribution/data/
```

**"Compute instance won't start"**
- Check quota: Portal → Subscriptions → Usage + quotas
- May need to request GPU quota increase

---

## Next Steps

1. **Set up compute instance** (10 min)
2. **Clone repo & upload data** (5 min)
3. **Run first training** (30 min)
4. **Compare to Colab results** (validate)
5. **Train 13-author model** (30 min)
6. **Deploy as API** (optional, 15 min)

---

## Files Created for Azure ML

- `azure_ml_training.ipynb` - Jupyter notebook with Azure ML logging
- `train_bert_azureml.py` - Training script with experiment tracking
- `deploy_model.py` - Deployment script for API endpoint

All include proper Azure ML integration for professional workflow.

---

**Questions?** Check Azure ML docs: https://learn.microsoft.com/azure/machine-learning/
