# train.py

import torch
import torch.nn.functional as F

from models import *
from data import get_dataloaders
import notebook_tools as nbtools
import json

# =========================================================
# Main training loop
# =========================================================
def main():
    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    in_channels = train_loader.dataset[0].x.shape[1]
    # --- Model ---
    model = HybridGNN_QNN_basic(in_channels=in_channels)

    model, history = nbtools.train_model(
            model,
            train_loader,
            val_loader=val_loader,   # or split a validation set
            epochs=50,
            patience=5,
            save_every=1,
            save_history_every_epoch=True,          # ✅ NEW
            history_dir="history_logs_QGNN_basic"              # ✅ NEW
    )
    acc, auc = nbtools.evaluate_model(model, test_loader)

    with open("training_history_QGNNBasic.json", "w") as f:
        json.dump(history, f, indent=4)

    print("📁 Saved training history to QGNNBasic.json")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()