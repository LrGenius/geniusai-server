"""
Prepares the CLIP model for bundling with the application.
This script creates a bundled model directory with all necessary files
including the config and weights.
"""
import json
import os
import shutil
from pathlib import Path

# Model configuration
MODEL_ID = "ViT-SO400M-16-SigLIP2-384"
HF_CACHE_MODEL_PATH = "models/models--timm--ViT-SO400M-16-SigLIP2-384/snapshots/fec784dabb3081a5f101fc74eefaf9d1ed08237b"
OUTPUT_DIR = "models/bundle/clip"

# Read the model config from open_clip
config_file = Path("open_clip/model_configs") / f"{MODEL_ID}.json"
with open(config_file, "r") as f:
    model_cfg = json.load(f)

# Create the complete config with preprocessing defaults
complete_config = {
    "model_cfg": model_cfg,
    "preprocess_cfg": {
        "size": [384, 384],
        "mode": "RGB",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "interpolation": "bicubic",
        "resize_mode": "shortest",
        "fill_color": 0
    }
}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Write the config file
config_output = os.path.join(OUTPUT_DIR, "open_clip_config.json")
with open(config_output, "w") as f:
    json.dump(complete_config, f, indent=2)
print(f"Created config: {config_output}")

# Copy the model weights and tokenizer files
source_dir = Path(HF_CACHE_MODEL_PATH)
if source_dir.exists():
    for file in ["open_clip_model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        src = source_dir / file
        dst = Path(OUTPUT_DIR) / file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {file}")
        else:
            print(f"Warning: {file} not found at {src}")
else:
    print(f"Warning: Source directory not found: {source_dir}")
    print("Please ensure the model has been downloaded from HuggingFace first.")

print(f"\nBundled model directory ready at: {OUTPUT_DIR}")
