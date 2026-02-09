#!/bin/bash
# Package data for Google Colab upload

echo "Packaging data for Colab..."

# Create a zip with just the essentials
zip -r burney_colab_data.zip \
    data/bert_data/label_mapping.json \
    data/bert_data/chunked_datasets/ \
    -x "*.pyc" "__pycache__/*"

echo ""
echo "✓ Created burney_colab_data.zip ($(du -h burney_colab_data.zip | cut -f1))"
echo ""
echo "Next steps:"
echo "1. Upload burney_colab_data.zip to your Google Drive"
echo "2. Open notebooks/train_bert_colab.ipynb in Colab"
echo "3. Mount Drive and unzip the data"
echo "4. Run all cells!"
