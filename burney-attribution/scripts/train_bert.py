#!/usr/bin/env python3
"""
Fine-tune BERT for authorship attribution.
"""

import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize texts."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


def main():
    """Train BERT model."""
    print("="*60)
    print("FINE-TUNING BERT FOR AUTHORSHIP ATTRIBUTION")
    print("="*60)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data' / 'bert_data'
    output_dir = project_dir / 'models' / 'bert_authorship'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    print("\nLoading label mapping...")
    with open(data_dir / 'label_mapping.json', 'r') as f:
        label_info = json.load(f)

    author_to_id = label_info['author_to_id']
    id_to_author = {int(k): v for k, v in label_info['id_to_author'].items()}
    num_labels = len(author_to_id)

    print(f"Training for {num_labels} authors: {sorted(author_to_id.keys())}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_from_disk(str(data_dir / 'chunked_datasets'))

    print(f"Train: {len(datasets['train'])} chunks")
    print(f"Val: {len(datasets['validation'])} chunks")
    print(f"Test: {len(datasets['test'])} chunks")

    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    model_name = "bert-base-uncased"  # Fallback since ECCO-BERT unavailable

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id_to_author,
            label2id=author_to_id
        )
        print(f"✓ Loaded {model_name}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text', 'author', 'title', 'year']
    )

    # Set format for PyTorch
    tokenized_datasets.set_format('torch')

    # Training arguments (optimized for overnight local training)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",  # Save after each epoch (safety for overnight run)
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Reduced from 8 for memory
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # Effective batch size = 4*2 = 8
        num_train_epochs=3,  # Reduced from 5
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        load_best_model_at_end=True,  # Load best checkpoint
        metric_for_best_model='f1_weighted',
        save_total_limit=2,  # Keep best + latest
        report_to='none',  # Disable wandb for now
        fp16=False,  # Disabled for M1/MPS stability
        dataloader_pin_memory=False  # Disable for MPS compatibility
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Increased patience
    )

    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    train_result = trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    val_metrics = trainer.evaluate()
    print(f"\nValidation Accuracy: {val_metrics['eval_accuracy']:.2%}")
    print(f"Validation F1 (macro): {val_metrics['eval_f1_macro']:.3f}")
    print(f"Validation F1 (weighted): {val_metrics['eval_f1_weighted']:.3f}")

    # Detailed evaluation on validation set
    val_predictions = trainer.predict(tokenized_datasets['validation'])
    val_preds = np.argmax(val_predictions.predictions, axis=1)
    val_labels = val_predictions.label_ids

    print("\nValidation Classification Report:")
    print(classification_report(
        val_labels,
        val_preds,
        target_names=sorted(author_to_id.keys()),
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\nValidation Confusion Matrix:")
    authors = sorted(author_to_id.keys())
    print(f"{'':12} " + " ".join(f"{a[:8]:>8}" for a in authors))
    for i, author in enumerate(authors):
        print(f"{author[:12]:12} " + " ".join(f"{cm[i,j]:8}" for j in range(len(authors))))

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    test_predictions = trainer.predict(tokenized_datasets['test'])
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    test_labels = test_predictions.label_ids

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')

    print(f"\nTest Accuracy: {test_accuracy:.2%}")
    print(f"Test F1 (macro): {test_f1_macro:.3f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.3f}")

    print("\nTest Classification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=sorted(author_to_id.keys()),
        zero_division=0
    ))

    # Save test results
    test_results = {
        'accuracy': float(test_accuracy),
        'f1_macro': float(test_f1_macro),
        'f1_weighted': float(test_f1_weighted),
        'predictions': test_preds.tolist(),
        'labels': test_labels.tolist()
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"Model saved to: {output_dir / 'final'}")
    print(f"Test results saved to: {output_dir / 'test_results.json'}")

    # Compare with baseline
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    print(f"Baseline (Burrows' Delta): 80.0% accuracy")
    print(f"BERT:                      {test_accuracy:.1%} accuracy")

    improvement = test_accuracy - 0.80
    if improvement > 0:
        print(f"\n✓ BERT improves on baseline by {improvement:.1%}")
    elif improvement < 0:
        print(f"\n⚠ BERT underperforms baseline by {-improvement:.1%}")
    else:
        print(f"\nBERT matches baseline performance")


if __name__ == "__main__":
    main()
