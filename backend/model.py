"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: PyTorch MLP Neural Network model definition, training loop, and load/save utilities.
- Usage: Imported by train.py and chat.py to train models and execute inference.
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
    Description:
        Multi-Layer Perceptron (MLP) for predicting P(over) from a 13-dimensional feature vector.
        Architecture: 13 -> 64 -> 32 -> 16 -> 1.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Description:
            Executes a forward pass through the neural network.
        Arguments:
            x: Input feature tensor.
        Returns:
            Output prediction tensor.
        """
        return self.net(x).squeeze(-1)


def train_model(features, labels, epochs=200, lr=0.001, val_split=0.2, verbose=True):
    """
    Description:
        Trains the RLPredictorNet MLP model on the provided features and labels.
    Arguments:
        features: Numpy array of shape (N, 13) containing feature vectors.
        labels: Numpy array of shape (N,) containing target binary labels (0/1).
        epochs: Count of training iterations.
        lr: Learning rate optimizer speed coefficient.
        val_split: Fraction to hold out for evaluation.
        verbose: print progress metrics.
    Returns:
        Tuple: (trained RLPredictorNet model, dictionary detailing validation history).
    """
    if not HAS_TORCH: raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    n = len(X)
    perm = torch.randperm(n)
    val_n = int(n * val_split)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if verbose: print(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

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
        model.train()
        optimizer.zero_grad()

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

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

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
            if verbose: print(f"  Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
            break

    if best_state: model.load_state_dict(best_state)

    if verbose: print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}, Final val acc: {history['val_acc'][-1]:.1%}")

    return model, history


def predict(model, features):
    """
    Description:
        Runs neural network prediction inference on single or batched features.
    Arguments:
        model: Trained RLPredictorNet instance.
        features: Numpy array of shape (13,) or (N, 13) containing feature vectors.
    Returns:
        Float or Numpy array representing probabilities.
    """
    if not HAS_TORCH: raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    model.eval()
    if features.ndim == 1: features = features.reshape(1, -1)

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        prob = model(x)

    result = prob.numpy()
    return float(result[0]) if len(result) == 1 else result


def save_model(model, path=MODEL_PATH):
    """
    Description:
        Saves trained model state dict weights to disk.
    Arguments:
        model: Trained RLPredictorNet instance.
        path: target filepath string.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    """
    Description:
        Loads a saved model from disk.
    Arguments:
        path: Path string to torch save file.
    Returns:
        RLPredictorNet instance if successful, else None.
    """
    if not HAS_TORCH: return None
    if not os.path.exists(path): return None

    model = RLPredictorNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
