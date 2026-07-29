#!/usr/bin/env python3
"""Validate the isolated SProf-GO runtime and optionally load the local ProtT5 model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MINIMUM_TORCH = (2, 6)


def release_tuple(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"Unrecognized PyTorch version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def require_safe_torch(version: str) -> None:
    if release_tuple(version) < MINIMUM_TORCH:
        raise RuntimeError(
            "SProf-GO requires PyTorch >=2.6 to load its PyTorch/ProtT5 "
            "checkpoints safely (CVE-2025-32434). Run: "
            "sbatch 'arc slurms/setup_sprof_go_arc.sh'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prott5-root", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--load-prott5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy
    import scipy
    import sentencepiece
    import sklearn
    import torch
    import transformers

    require_safe_torch(torch.__version__)
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("SProf-GO requested a GPU, but PyTorch cannot access CUDA")

    print(
        "SProf-GO environment:",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__}",
        f"sentencepiece={sentencepiece.__version__}",
        f"numpy={numpy.__version__}",
        f"scipy={scipy.__version__}",
        f"sklearn={sklearn.__version__}",
        f"cuda={torch.cuda.is_available()}",
    )

    if args.load_prott5:
        if args.prott5_root is None or not args.prott5_root.is_dir():
            raise SystemExit("--load-prott5 requires an existing --prott5-root")
        from transformers import T5EncoderModel, T5Tokenizer

        root = str(args.prott5_root.resolve())
        T5Tokenizer.from_pretrained(root, do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(root, local_files_only=True)
        del model
        print(f"Secure ProtT5 load passed: {root}")


if __name__ == "__main__":
    main()
