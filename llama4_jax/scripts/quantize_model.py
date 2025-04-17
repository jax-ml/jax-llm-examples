#!/usr/bin/env python3

import os.path
from pathlib import Path
from argparse import ArgumentParser

from llama4_jax import chkpt_utils as utils


def main(path: str | Path, suffix: str):
    path = Path(path).expanduser().absolute()
    dest_path = path.parent / f"{path.name}{suffix}"
    utils.quantize_model(path, dest_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path", default=os.path.join(os.path.expanduser("~"), "DeepSeek-R1-Distill-Llama-70B"),
        required=True, help="Existing JAX model checkpoint path"
    )
    parser.add_argument(
        "--suffix",
        default="quant",
        help="Suffix for a new checkpoint directory, e.g., path=~/model, suffix=-quant -> ~/model-quant",
    )

    args = parser.parse_args()
    main(args.path, args.suffix if args.suffix.startswith("-") else f"-{args.suffix}")
