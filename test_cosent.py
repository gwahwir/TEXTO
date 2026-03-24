"""
Quick test script to verify CoSENTLoss implementation.
"""
from training import load_feedback_data, convert_to_training_format

# Load and convert data
feedback = load_feedback_data()
print(f"Loaded {len(feedback)} feedback entries")

samples = convert_to_training_format(feedback)
print(f"Generated {len(samples)} training samples")

# Show sample data structure
if samples:
    print("\nExample training sample:")
    print(f"  Query: {samples[0]['query']}")
    print(f"  Document: {samples[0]['document']}")
    print(f"  Label (rank): {samples[0]['label']}")

    # Count total ranked items
    total_items = sum(len(e.get("ranked_issues", [])) for e in feedback)
    print(f"\nData utilization: {len(samples)}/{total_items} items ({len(samples)/total_items*100:.1f}%)")

    # Show rank distribution
    from collections import Counter
    rank_counts = Counter(s['label'] for s in samples)
    print(f"\nRank distribution (top 10):")
    for rank, count in sorted(rank_counts.items())[:10]:
        print(f"  Rank {int(rank)}: {count} samples")
