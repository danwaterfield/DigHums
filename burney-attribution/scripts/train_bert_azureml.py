#!/usr/bin/env python3
"""
BERT Authorship Attribution Training - Azure ML Version

This script trains the BERT model with Azure ML experiment tracking.
Logs metrics, parameters, and artifacts to Azure ML workspace.

Usage:
    python train_bert_azureml.py [--data-dir PATH] [--output-dir PATH]
"""

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

# Azure ML imports
try:
    import mlflow
    import mlflow.pytorch
    AZURE_ML_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: mlflow not installed. Running without Azure ML tracking.")
    print("   Install with: pip install mlflow azureml-mlflow")
    AZURE_ML_AVAILABLE = False

# ML imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import torch


class AzureMLCallback(TrainerCallback):
    """Custom callback to log metrics to Azure ML during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics to Azure ML."""
        if not AZURE_ML_AVAILABLE or logs is None:
            return

        # Log metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=state.global_step)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    accuracy = (predictions == labels).mean()

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }


def main():
    parser = argparse.ArgumentParser(description='Train BERT for authorship attribution')
    parser.add_argument('--data-dir', type=str, default='data/bert_data',
                        help='Directory containing prepared datasets')
    parser.add_argument('--output-dir', type=str, default='models/bert_authorship',
                        help='Directory to save trained model')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='Pretrained model name or path')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--experiment-name', type=str, default='burney-attribution',
                        help='Azure ML experiment name')

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / args.data_dir
    output_dir = project_dir / args.output_dir

    print("=" * 70)
    print("BERT AUTHORSHIP ATTRIBUTION TRAINING")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print()

    # Start Azure ML run
    if AZURE_ML_AVAILABLE:
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run()
        print("✅ Azure ML experiment tracking enabled")
        print(f"   Experiment: {args.experiment_name}")
        print()

        # Log parameters
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
    else:
        print("⚠️  Running without Azure ML tracking")
        print()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load datasets
    print("📖 Loading datasets...")
    train_dataset = load_from_disk(str(data_dir / 'chunked_datasets' / 'train'))
    val_dataset = load_from_disk(str(data_dir / 'chunked_datasets' / 'validation'))
    test_dataset = load_from_disk(str(data_dir / 'chunked_datasets' / 'test'))

    # Load label mapping
    with open(data_dir / 'label_mapping.json', 'r') as f:
        label_info = json.load(f)

    num_labels = len(label_info['author_to_id'])
    id_to_author = {int(k): v for k, v in label_info['id_to_author'].items()}

    print(f"✅ Datasets loaded:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Validation: {len(val_dataset):,} samples")
    print(f"   Test: {len(test_dataset):,} samples")
    print(f"\n📚 Authors ({num_labels}): {', '.join(label_info['author_to_id'].keys())}")
    print()

    if AZURE_ML_AVAILABLE:
        mlflow.log_param("num_authors", num_labels)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("val_size", len(val_dataset))
        mlflow.log_param("test_size", len(test_dataset))

    # Load model and tokenizer
    print(f"🤖 Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    print("✅ Model loaded")
    print()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1_weighted',
        save_total_limit=2,
        report_to='none',  # We're using mlflow directly
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
    )

    # Create trainer with Azure ML callback
    callbacks = [AzureMLCallback()] if AZURE_ML_AVAILABLE else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Train
    print("🏃 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print()

    train_result = trainer.train()

    print("\n✅ Training complete!")
    print(f"   Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"   Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    print()

    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"F1 (weighted): {test_results['eval_f1_weighted']:.4f}")
    print(f"F1 (macro): {test_results['eval_f1_macro']:.4f}")

    # Detailed per-author results
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    target_names = [id_to_author[i] for i in range(num_labels)]

    print("\n" + "=" * 70)
    print("PER-AUTHOR PERFORMANCE")
    print("=" * 70)
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=4,
        output_dict=True
    )
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=4
    ))
    print("=" * 70)

    # Log test results to Azure ML
    if AZURE_ML_AVAILABLE:
        mlflow.log_metric("test_accuracy", test_results['eval_accuracy'])
        mlflow.log_metric("test_f1_weighted", test_results['eval_f1_weighted'])
        mlflow.log_metric("test_f1_macro", test_results['eval_f1_macro'])

        # Log per-author metrics
        for author in target_names:
            mlflow.log_metric(f"test_{author}_precision", report[author]['precision'])
            mlflow.log_metric(f"test_{author}_recall", report[author]['recall'])
            mlflow.log_metric(f"test_{author}_f1", report[author]['f1-score'])

        # Log confusion matrix as artifact
        cm = confusion_matrix(true_labels, pred_labels)
        cm_file = output_dir / 'confusion_matrix.npy'
        np.save(cm_file, cm)
        mlflow.log_artifact(str(cm_file))

    # Save final model
    final_model_dir = output_dir / 'final'
    final_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving final model to {final_model_dir}...")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Save label mapping with model
    with open(final_model_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_info, f, indent=2)

    # Save test results
    results_file = project_dir / 'results' / 'test_results.json'
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'test_accuracy': test_results['eval_accuracy'],
            'test_f1_weighted': test_results['eval_f1_weighted'],
            'test_f1_macro': test_results['eval_f1_macro'],
            'per_author': {
                author: {
                    'precision': report[author]['precision'],
                    'recall': report[author]['recall'],
                    'f1': report[author]['f1-score'],
                    'support': report[author]['support']
                }
                for author in target_names
            }
        }, f, indent=2)

    print("✅ Model saved")

    # Log model to Azure ML
    if AZURE_ML_AVAILABLE:
        print("\n📦 Logging model to Azure ML...")
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="burney-authorship-attribution"
        )
        mlflow.log_artifact(str(results_file))
        print("✅ Model logged to Azure ML Model Registry")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"✅ Model saved to: {final_model_dir}")
    print(f"✅ Results saved to: {results_file}")
    if AZURE_ML_AVAILABLE:
        print(f"✅ Experiment tracked in Azure ML")
        print(f"   View at: https://ml.azure.com")
    print("=" * 70)

    # End Azure ML run
    if AZURE_ML_AVAILABLE:
        mlflow.end_run()


if __name__ == "__main__":
    main()
