#!/usr/bin/env python3
"""
train.py — Train the RLPredictor neural network.

Generates self-supervised training data from cached Ballchasing replays
and trains the MLP to predict Over/Under probabilities.

Usage:
    python3 train.py                        # Full training
    python3 train.py --epochs 5 --sample 100  # Quick smoke test
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from features import build_training_data
from model import train_model, save_model, HAS_TORCH


def main():
    parser = argparse.ArgumentParser(description="Train the RLPredictor neural network.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--sample", type=int, default=0, help="If set, randomly sample N rows for a quick test")
    parser.add_argument("--cache", type=str, default=".bc_cache.json", help="Path to Ballchasing cache file")
    parser.add_argument("--min-lookback", type=int, default=5, help="Minimum games of history before generating a row")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch is not installed. Run: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("RLPredictor — Neural Network Training")
    print("=" * 60)

    # Step 1: Generate training data from cache
    print("\nStep 1: Generating training data from cached replays...\n")
    features, labels, meta = build_training_data(
        cache_path=args.cache,
        min_lookback=args.min_lookback,
    )

    # NEW: Load verified live predictions if they exist
    log_path = "data/prediction_log.csv"
    if os.path.exists(log_path):
        try:
            df_live = pd.read_csv(log_path)
            # Only take rows that have been verified (have a label)
            df_live = df_live.dropna(subset=["label"])
            if not df_live.empty:
                print(f"📈 Found {len(df_live)} verified live predictions. Merging into training set...")
                
                live_feats = []
                live_labels = []
                for _, row in df_live.iterrows():
                    # Parse semicolon-separated features
                    f_vec = np.array([float(x) for x in str(row["features"]).split(";")], dtype=np.float32)
                    live_feats.append(f_vec)
                    live_labels.append(float(row["label"]))
                
                # Stack and append
                features = np.vstack([features, np.array(live_feats)])
                labels = np.concatenate([labels, np.array(live_labels)])
                print(f"✅ Total samples now: {len(features)} ({len(df_live)} from live log)")
        except Exception as e:
            print(f"⚠️  Could not load live log: {e}")

    if len(features) == 0:
        print("No training data could be generated. Make sure .bc_cache.json has replay data.")
        print("Run some queries first: python3 main.py <url> --mode chat")
        sys.exit(1)

    # Optional: subsample for a quick test
    if args.sample and args.sample < len(features):
        idx = np.random.choice(len(features), args.sample, replace=False)
        features = features[idx]
        labels = labels[idx]
        meta = [meta[i] for i in idx]
        print(f"Subsampled to {args.sample} rows for quick test\n")

    # Step 2: Train
    print(f"\nStep 2: Training model ({args.epochs} epochs, lr={args.lr})...\n")
    model, history = train_model(
        features, labels,
        epochs=args.epochs,
        lr=args.lr,
        verbose=True,
    )

    # Step 3: Save
    print("\nStep 3: Saving model...\n")
    save_model(model)

    # Step 4: Quick sanity check
    print("\nStep 4: Sanity check on a few samples...\n")
    from model import predict
    test_indices = np.random.choice(len(features), min(5, len(features)), replace=False)
    for i in test_indices:
        prob = predict(model, features[i])
        actual = "OVER" if labels[i] == 1 else "UNDER"
        m = meta[i]
        pred_str = "OVER" if prob > 0.5 else "UNDER"
        correct = ":)" if (prob > 0.5) == (labels[i] == 1) else ":("
        print(f"{m['player']:>15s} | {m['stat']} > {m['threshold']} | "
              f"Predicted: {prob:.1%} ({pred_str}) | Actual: {actual} {correct}")

    print(f"\n{'=' * 60}")
    print(f"Training complete! Model saved to data/model.pt")
    print(f"{len(features)} samples, best val acc: {max(history['val_acc']):.1%}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
