# train.py

import torch
import torch.nn.functional as F

from models import *
from data import get_dataloaders
import notebook_tools as nbtools

# =========================================================
# Main training loop
# =========================================================
def main():
    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    in_channels = train_loader.dataset[0].x.shape[1]
    # --- Model ---
    model = JetGNN(in_channels=in_channels)

    model = nbtools.train_model(
            model,
            train_loader,
            val_loader=val_loader,   # or split a validation set
            epochs=10,
            patience=5,
            save_every=1
    )
    acc, auc = nbtools.evaluate_model(model, test_loader)


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()