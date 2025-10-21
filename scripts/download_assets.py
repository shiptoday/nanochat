#!/usr/bin/env python3
"""
Utility to download tokenizer assets and model checkpoints from Hugging Face
into the local nanochat cache directory.

Defaults:
  - Tokenizer files from karpathy/nanochat-d32
  - Base checkpoint from shiptoday101/nanochat-d20-base (diego d20 model)

You can override the base directory via --base-dir or the NANOCHAT_BASE_DIR
environment variable. Otherwise ~/.cache/nanochat is used.

Example:
    python -m scripts.download_assets
    python -m scripts.download_assets --base-dir /workspace/cache/nanochat
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

DEFAULT_ASSETS = {
    "tokenizer": dict(
        repo_id="karpathy/nanochat-d32",
        repo_type="model",
        subdir="tokenizer",
        allow=("tokenizer.pkl", "token_bytes.pt"),
    ),
    "diego-d20-base": dict(
        repo_id="shiptoday101/nanochat-d20-base",
        repo_type="model",
        subdir="base_checkpoints/diego-d20-base",
        allow=("model_021400.pt", "meta_021400.json", "optim_021400.pt"),
    ),
}


@dataclass
class DownloadItem:
    name: str
    repo_id: str
    repo_type: str
    subdir: Path
    allow_patterns: tuple[str, ...]


def resolve_base_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("NANOCHAT_BASE_DIR"):
        return Path(os.environ["NANOCHAT_BASE_DIR"]).expanduser().resolve()
    return Path.home() / ".cache" / "nanochat"


def parse_extra_specs(specs: Iterable[str]) -> list[DownloadItem]:
    """
    Parse extra specs of the form:
        repo_id::subdir::pattern1,pattern2
    """
    items: list[DownloadItem] = []
    for spec in specs:
        parts = spec.split("::")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid extra spec '{spec}'. Expected format "
                "repo_id::subdir::pattern1,pattern2"
            )
        repo_id, subdir, pattern_str = parts
        patterns = tuple(p.strip() for p in pattern_str.split(",") if p.strip())
        if not patterns:
            raise ValueError(f"No patterns provided in spec '{spec}'")
        items.append(
            DownloadItem(
                name=subdir,
                repo_id=repo_id,
                repo_type="model",
                subdir=Path(subdir),
                allow_patterns=patterns,
            )
        )
    return items


def build_download_plan(args: argparse.Namespace) -> list[DownloadItem]:
    plan: list[DownloadItem] = []
    if not args.skip_tokenizer:
        cfg = DEFAULT_ASSETS["tokenizer"]
        plan.append(
            DownloadItem(
                name="tokenizer",
                repo_id=cfg["repo_id"],
                repo_type=cfg["repo_type"],
                subdir=Path(cfg["subdir"]),
                allow_patterns=cfg["allow"],
            )
        )
    if not args.skip_diego_base:
        cfg = DEFAULT_ASSETS["diego-d20-base"]
        plan.append(
            DownloadItem(
                name="diego-d20-base",
                repo_id=cfg["repo_id"],
                repo_type=cfg["repo_type"],
                subdir=Path(cfg["subdir"]),
                allow_patterns=cfg["allow"],
            )
        )
    if args.extra:
        plan.extend(parse_extra_specs(args.extra))
    return plan


def download_item(base_dir: Path, item: DownloadItem, token: str | None, resume: bool):
    destination = base_dir / item.subdir
    destination.mkdir(parents=True, exist_ok=True)
    print(f"→ {item.name}: {item.repo_id} → {destination}")

    snapshot_download(
        repo_id=item.repo_id,
        repo_type=item.repo_type,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        allow_patterns=item.allow_patterns,
        token=token,
        resume_download=resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Download nanochat assets from Hugging Face.")
    parser.add_argument(
        "--base-dir",
        help="Destination root directory. Defaults to NANOCHAT_BASE_DIR or ~/.cache/nanochat.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip downloading tokenizer assets.",
    )
    parser.add_argument(
        "--skip-diego-base",
        action="store_true",
        help="Skip downloading the diego-d20-base checkpoint.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        help="Additional downloads formatted as repo_id::subdir::pattern1,pattern2",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face token. Otherwise relies on cached login/environment.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted downloads (pass-through to snapshot_download).",
    )
    args = parser.parse_args()

    plan = build_download_plan(args)
    if not plan:
        print("Nothing to download.")
        return

    base_dir = resolve_base_dir(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using base directory: {base_dir}")

    for item in plan:
        download_item(base_dir, item, token=args.token, resume=args.resume)

    print("All downloads complete.")


if __name__ == "__main__":
    main()
