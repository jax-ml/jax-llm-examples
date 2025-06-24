#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

example_models = [
  "meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
]

def main(model_id: str, dest_root_path: str | Path):
    from huggingface_hub import snapshot_download

    local_dir = Path(dest_root_path).expanduser().absolute() / str(model_id).replace("/", "--")
    snapshot_download(repo_id=model_id, local_dir=local_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id", required=True, help=f"HuggingFace model / repo id. Examples include: {example_models}"
    )
    parser.add_argument(
        "--dest-root-path",
        required=True,
        default="~/",
        help="Destination root directory, the model will be saved into its own directory.",
    )
    args = parser.parse_args()
    main(args.model_id, args.dest_root_path)
