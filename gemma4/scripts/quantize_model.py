#!/usr/bin/env python3

import sys
from pathlib import Path
from argparse import ArgumentParser


def main(path: str | Path, suffix: str):
    try:
        from gemma4_jax import chkpt_utils as utils
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from gemma4_jax import chkpt_utils as utils

    path = Path(path).expanduser().absolute()
    dest_path = path.parent / f"{path.name}{suffix}"
    utils.quantize_model(path, dest_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path", default="~/gemma-4-26B-A4B-it", required=True, help="Existing JAX model checkpoint path"
    )
    parser.add_argument(
        "--suffix",
        default="quant",
        help="Suffix for a new checkpoint directory, e.g., path=~/model, suffix=-quant -> ~/model-quant",
    )

    args = parser.parse_args()
    main(args.path, args.suffix if args.suffix.startswith("-") else f"-{args.suffix}")
