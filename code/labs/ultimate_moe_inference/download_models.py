#!/usr/bin/env python3
"""
download_models.py - Download gpt-oss models from HuggingFace

Downloads:
- openai/gpt-oss-20b (single GPU, ~40GB)
- openai/gpt-oss-120b (multi-GPU, ~240GB)

Usage:
    python download_models.py              # Download 20b only
    python download_models.py --all        # Download both models
    python download_models.py --model 120b # Download specific model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Check for huggingface_hub
try:
    from huggingface_hub import snapshot_download, login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import snapshot_download, login

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
LOCAL_MODELS_DIR = Path(__file__).parent / "models"


MODELS = {
    "20b": {
        "repo_id": "openai/gpt-oss-20b",
        "description": "21B params, 3.6B active (MoE), single B200 GPU",
        "size_gb": 40,
    },
    "120b": {
        "repo_id": "openai/gpt-oss-120b",
        "description": "117B params, 5.1B active (MoE), 8x B200 GPUs",
        "size_gb": 240,
    },
}


def download_model(
    model_key: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    token: str | None = None,
) -> Path:
    """Download a model from HuggingFace Hub.
    
    Args:
        model_key: "20b" or "120b"
        cache_dir: Where to cache the model
        token: HuggingFace token (optional, for gated models)
        
    Returns:
        Path to the downloaded model
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")
    
    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]
    
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"Description: {model_info['description']}")
    print(f"Size: ~{model_info['size_gb']}GB")
    print(f"Cache: {cache_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Download the model
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            token=token,
            resume_download=True,  # Resume if interrupted
            local_files_only=False,
        )
        
        print(f"\n✅ Downloaded to: {local_path}")
        return Path(local_path)
        
    except Exception as e:
        print(f"\n❌ Failed to download {repo_id}: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. If model is gated, run: huggingface-cli login")
        print("  3. Ensure you have enough disk space")
        raise


def check_existing_models() -> dict[str, Path | None]:
    """Check which models are already downloaded."""
    results = {}
    
    for key, info in MODELS.items():
        repo_id = info["repo_id"]
        # Check in default cache
        cache_path = DEFAULT_CACHE_DIR / f"models--{repo_id.replace('/', '--')}"
        
        if cache_path.exists():
            # Find the actual snapshot
            snapshots = list((cache_path / "snapshots").glob("*")) if (cache_path / "snapshots").exists() else []
            if snapshots:
                results[key] = snapshots[0]
            else:
                results[key] = None
        else:
            results[key] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Download gpt-oss models")
    parser.add_argument("--model", choices=["20b", "120b"], default="20b",
                       help="Which model to download (default: 20b)")
    parser.add_argument("--all", action="store_true",
                       help="Download all models")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                       help="Cache directory for models")
    parser.add_argument("--check", action="store_true",
                       help="Check which models are already downloaded")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace token (for gated models)")
    args = parser.parse_args()
    
    # Check existing models
    existing = check_existing_models()
    
    print("\n" + "="*60)
    print("GPT-OSS MODEL STATUS")
    print("="*60)
    
    for key, path in existing.items():
        info = MODELS[key]
        status = "✅ Downloaded" if path else "❌ Not downloaded"
        print(f"\n  {info['repo_id']}")
        print(f"    Status: {status}")
        if path:
            print(f"    Path: {path}")
        print(f"    Size: ~{info['size_gb']}GB")
    
    print("="*60 + "\n")
    
    if args.check:
        return
    
    # Download requested models
    models_to_download = []
    if args.all:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]
    
    for model_key in models_to_download:
        if existing.get(model_key):
            print(f"✅ {MODELS[model_key]['repo_id']} already downloaded")
            print(f"   Path: {existing[model_key]}")
        else:
            download_model(model_key, args.cache_dir, args.token)
    
    # Print usage instructions
    print("\n" + "="*60)
    print("USAGE")
    print("="*60)
    print("\nTo use with the ultimate lab:")
    print("  cd labs/ultimate_moe_inference")
    print("  python run_full_analysis.py --benchmark-only")
    print("\nThe models will be loaded from the HuggingFace cache automatically.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


