"""
model.py — RLPredictor Neural Network model definition.

PyTorch MLP for Over/Under prediction:
  13 input features → 64 → 32 → 16 → 1 (sigmoid probability)
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.pt")
INPUT_DIM = 13


class RLPredictorNet(nn.Module):
    """
    MLP for predicting P(over) from a 13-dimensional feature vector.

    Architecture: 13 → 64 → 32 → 16 → 1
    Each hidden layer: Linear → BatchNorm → ReLU → Dropout(0.3)
    Output: Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 13 → 64
            nn.Linear(INPUT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 2: 64 → 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 3: 32 → 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output: 16 → 1
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(features, labels, epochs=200, lr=0.001, val_split=0.2, verbose=True):
    """
    Train the RLPredictorNet on the given features and labels.

    Args:
        features: np.ndarray of shape (N, 13)
        labels:   np.ndarray of shape (N,) — binary 0/1
        epochs:   number of training epochs
        lr:       learning rate
        val_split: fraction to hold out for validation
        verbose:  print progress

    Returns: trained model, training history dict
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    # Train/val split
    n = len(X)
    perm = torch.randperm(n)
    val_n = int(n * val_split)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if verbose:
        print(f"📊 Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

    # Model, loss, optimizer
    model = RLPredictorNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 30

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        optimizer.zero_grad()

        # Mini-batch training for better generalisation
        batch_size = min(256, len(X_train))
        perm_t = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm_t[i : i + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:>4d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.1%}"
            )

        if patience_counter >= patience_limit:
            if verbose:
                print(f"  ⏹️  Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}, Final val acc: {history['val_acc'][-1]:.1%}")

    return model, history


def predict(model, features):
    """
    Run inference on a single feature vector or batch.

    Args:
        model: trained RLPredictorNet
        features: np.ndarray of shape (13,) or (N, 13)

    Returns: float probability or np.ndarray of probabilities
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    model.eval()
    if features.ndim == 1:
        features = features.reshape(1, -1)

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        prob = model(x)

    result = prob.numpy()
    return float(result[0]) if len(result) == 1 else result


def save_model(model, path=MODEL_PATH):
    """Save trained model weights to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")


def load_model(path=MODEL_PATH):
    """Load a trained model from disk. Returns None if no model found."""
    if not HAS_TORCH:
        return None
    if not os.path.exists(path):
        return None

    model = RLPredictorNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
