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
    Convert user rankings to CoSENTLoss format.

    CoSENTLoss preserves full ranking order by using rank as label.
    For each query-document pair:
    - texts: [query, document_text]
    - label: user_rank (0 = most relevant, higher = less relevant)

    This uses 100% of ranking information, not just binary relevance.
    """
    training_samples = []

    for entry in feedback_list:
        query = entry.get("query", "").strip()
        ranked = entry.get("ranked_issues", [])

        if not query or len(ranked) < 2:
            continue  # Need at least 2 items for ranking

        # Create one training sample per ranked item
        for item in ranked:
            text = item.get("text", "").strip()
            user_rank = item.get("user_rank")

            if not text or user_rank is None:
                continue

            # CoSENTLoss format: (query, document, rank_label)
            training_samples.append({
                "query": query,
                "document": text,
                "label": float(user_rank)  # Lower rank = more relevant
            })

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

    total_ranked_items = sum(len(e.get("ranked_issues", [])) for e in feedback)
    print(f"[training] Using CoSENTLoss - preserving full ranking order from {total_ranked_items} ranked items")

    # CoSENTLoss expects InputExample format with texts and label
    from sentence_transformers import InputExample
    train_examples = [
        InputExample(texts=[s["query"], s["document"]], label=s["label"])
        for s in samples
    ]
    train_dataset = Dataset.from_dict({
        "sentence1": [ex.texts[0] for ex in train_examples],
        "sentence2": [ex.texts[1] for ex in train_examples],
        "label": [ex.label for ex in train_examples]
    })

    # Load base model
    print(f"[training] Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,  # Increased for pairwise ranking
        learning_rate=learning_rate,
        warmup_steps=min(100, len(samples) // 2),  # Adjust warmup based on data size
        fp16=False,  # Disable mixed precision for CPU compatibility
        save_strategy="epoch",
        logging_steps=max(1, len(samples) // 10),
        save_total_limit=1,  # Only keep the final model
        report_to="tensorboard",  # Log locally, no account needed
    )

    # Loss function for ranking - CoSENTLoss preserves full ranking order!
    # It learns: similarity(query, rank_0) > similarity(query, rank_1) > ...
    loss = losses.CoSENTLoss(model=model)

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
    total_ranked_items = sum(len(e.get("ranked_issues", [])) for e in feedback)
    metadata = {
        "base_model": base_model_name,
        "training_samples": len(samples),
        "feedback_entries": len(feedback),
        "total_ranked_items": total_ranked_items,
        "loss_function": "CoSENTLoss",
        "ranking_aware": True,
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
