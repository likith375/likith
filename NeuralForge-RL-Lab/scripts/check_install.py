#!/usr/bin/env python3
"""Check NeuralForge's installed runtime without loading a model or using the network."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys


CORE = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pypdf": "pypdf",
    "python-docx": "docx",
    "openpyxl": "openpyxl",
    "python-pptx": "pptx",
}
ML = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
}


def check_package(distribution: str, module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
        return True, version
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true", help="Skip optional model-training packages.")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required.")
        return 1

    failed = []
    for group_name, packages in (("Core", CORE), ("ML", {} if args.core_only else ML)):
        print(f"\n{group_name} packages:")
        if not packages:
            print("  skipped")
            continue
        for distribution, module in packages.items():
            ok, detail = check_package(distribution, module)
            print(f"  {'OK' if ok else 'MISSING'} {distribution}: {detail}")
            if not ok and group_name == "Core":
                failed.append(distribution)

    if not args.core_only:
        try:
            import torch
            print(f"\nPyTorch device: {'CUDA ' + str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'}")
        except Exception as exc:
            print(f"\nPyTorch device check unavailable: {type(exc).__name__}: {exc}")

    if failed:
        print("\nInstall missing core packages with: python -m pip install -r requirements.txt")
        return 1
    print("\nCore runtime check passed. Optional ML packages are required only for the built-in Transformers adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
