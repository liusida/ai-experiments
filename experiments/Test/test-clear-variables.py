# %% Load small model (use toolbar Load or run this cell)
"""Minimal script to exercise clear-variables vs loaded model bindings."""

from __future__ import annotations

import stonesoup

MODEL_ID = "Qwen/Qwen3.5-0.8B"

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
print("loaded:", MODEL_ID, flush=True)

test_variable = 1

# Now run the cell and then clear variables, the model should still be loaded.
