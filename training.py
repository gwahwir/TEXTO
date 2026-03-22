"""
Continual learning module for fine-tuning sentence transformers
based on user ranking feedback.
"""
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import losses
from datasets import Dataset

FEEDBACK_FILE = "user_feedback.jsonl"
TRAINED_MODELS_DIR = "trained_models"


def load_feedback_data():
    """Load and parse all user feedback."""
    path = Path(FEEDBACK_FILE)
    if not path.exists():
        return []

    feedback = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    feedback.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return feedback


def convert_to_training_format(feedback_list):
    """
    Convert user rankings to MultipleNegativesRankingLoss format.

    For each query:
    - Top-ranked issue = positive example
    - Lower-ranked issues = hard negatives (weighted by rank distance)
    - Issues below threshold = negatives
    """
    training_samples = []

    for entry in feedback_list:
        query = entry.get("query", "").strip()
        ranked = entry.get("ranked_issues", [])

        if not query or len(ranked) < 2:
            continue  # Need at least positive + negative

        # Sort by user rank (0 = most relevant)
        ranked_sorted = sorted(ranked, key=lambda x: x.get("user_rank", 999))

        # Top item is positive
        positive = ranked_sorted[0].get("text", "")
        if not positive:
            continue

        # Lower ranked items are negatives
        negatives = [item.get("text", "") for item in ranked_sorted[1:] if item.get("text")]

        if not negatives:
            continue

        # Create training sample
        sample = {
            "sentence1": query,      # Query/anchor
            "sentence2": positive,   # Best match
        }

        # Add up to 2 hard negatives for MultipleNegativesRankingLoss
        if len(negatives) > 0:
            sample["sentence3"] = negatives[0]
        if len(negatives) > 1:
            sample["sentence4"] = negatives[1]

        training_samples.append(sample)

    return training_samples


def train_model(base_model_name, output_dir, learning_rate=2e-5, epochs=1):
    """
    Fine-tune the base model on user feedback.

    Args:
        base_model_name: Name or path of the base sentence-transformer model
        output_dir: Directory to save the fine-tuned model
        learning_rate: Learning rate for training (default: 2e-5)
        epochs: Number of training epochs (default: 1)

    Returns:
        output_dir: Path to the saved model

    Raises:
        ValueError: If not enough feedback data
    """
    # Load feedback
    feedback = load_feedback_data()
    if len(feedback) < 10:
        raise ValueError(f"Not enough feedback data: {len(feedback)} < 10")

    # Convert to training format
    samples = convert_to_training_format(feedback)
    if len(samples) < 5:
        raise ValueError(f"Not enough valid training samples: {len(samples)} < 5")

    train_dataset = Dataset.from_list(samples)

    # Load base model
    print(f"[training] Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=learning_rate,
        warmup_steps=min(100, len(samples) // 2),  # Adjust warmup based on data size
        fp16=False,  # Disable mixed precision for CPU compatibility
        save_strategy="epoch",
        logging_steps=max(1, len(samples) // 10),
        save_total_limit=1,  # Only keep the final model
        report_to="tensorboard",  # Log locally, no account needed
    )

    # Loss function for ranking
    loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=20.0,  # Temperature for softmax
    )

    # Create trainer
    print(f"[training] Creating trainer with {len(samples)} training samples")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    # Train
    print(f"[training] Starting training...")
    trainer.train()

    # Save final model
    print(f"[training] Saving model to {output_dir}")
    model.save(output_dir)

    # Save metadata
    metadata = {
        "base_model": base_model_name,
        "training_samples": len(samples),
        "feedback_entries": len(feedback),
        "timestamp": time.time(),
        "learning_rate": learning_rate,
        "epochs": epochs,
    }

    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"[training] Training complete!")
    return output_dir


def list_trained_models():
    """List all available fine-tuned models."""
    models_dir = Path(TRAINED_MODELS_DIR)
    if not models_dir.exists():
        return []

    models = []
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    models.append({
                        "name": model_path.name,
                        "path": str(model_path),
                        "metadata": metadata
                    })
                except (json.JSONDecodeError, IOError):
                    continue

    return sorted(models, key=lambda m: m["metadata"].get("timestamp", 0), reverse=True)
