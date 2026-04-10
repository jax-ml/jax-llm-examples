#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from argparse import ArgumentParser
import dataclasses
import shutil

def main(model_path: str | Path, ckpt_path: str | Path):
    try:
        from gemma4_jax import model as g4jax
        from gemma4_jax import chkpt_utils as utils
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from gemma4_jax import model as g4jax
        from gemma4_jax import chkpt_utils as utils

    from transformers import AutoConfig
    from safetensors import safe_open
    from tqdm import tqdm

    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()
    files = list(model_path.glob("**/*safetensors"))
    assert len(files) >= 1
    config_files = list(model_path.glob("**/config.json"))
    assert len(config_files) == 1, "Must have only one `config.json` file in the model path"
    # config = AutoConfig.from_pretrained(config_files[0])
    config = json.loads(config_files[0].read_text())["text_config"]
    cfg = g4jax.hf_to_jax_config(config)

    # Gemma 4 model checkpoints are distributed unquantized
    weights = g4jax.Weights.abstract(dataclasses.replace(cfg, quant_moe=False, quant_mlp=False, quant_attn=False))

    if not ckpt_path.exists():
        model = {}
        for file in tqdm(files):
            with safe_open(file, framework="torch") as f:
                for key in tqdm(f.keys(), leave=False):
                    if all(keyword not in key for keyword in ["audio", "vision"]):
                        model[key] = f.get_tensor(key)
        converted_weights = utils.convert_model_or_layer(weights, model, cfg, sequential=True, allow_unconverted_parameters=False)
        g4jax.save_pytree(converted_weights, ckpt_path)

    additional_files = [f for f in model_path.glob("*") if f.is_file() and f.suffix in (".json", ".jinja")]
    for additional_file in additional_files:
        shutil.copyfile(additional_file, ckpt_path / additional_file.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-path", default="~/gemma-4-26B-A4B-it", required=True, help="HF model directory path"
    )
    parser.add_argument(
        "--dest-path",
        default="~/gemma4_jax/gemma-4-26B-A4B-it",
        required=True,
        help="JAX model model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)
