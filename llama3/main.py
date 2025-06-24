# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import os

from etils import epath
import json
from pprint import pprint

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import use_mesh, AxisType, PartitionSpec as P
import numpy as np

from llama3_jax import model as l3jax


def encode_input(tokenizer, texts: list[str], model_name: str, pad_id: int = 0):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}])
        + tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")
        + ([] if "deepseek" not in model_name.lower() else tokenizer.encode("<think>"))
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    jax.distributed.initialize()
    quant = os.environ.get("QUANT") in (None, 't', 'T', 'True', 'true', 'TRUE', 1)

    ckpt_path = epath.Path("~").expanduser() / "bucket" / "DeepSeek-R1-Distill-Llama-3.1-70B-Instruct"
    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"
    tokenizer = l3jax.load_tokenizer(ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json")

    mesh = jax.make_mesh(
        (1, 8, jax.device_count() // 8), ("x", "y", "z"), devices=jax.devices(), axis_types=(AxisType.Explicit,) * 3
    )
    cfg = l3jax.llama_to_jax_config(json.loads((ckpt_path / "config.json").read_text()))
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_layer=quant, quant_cache=quant)
    weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(cfg))

    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
        model_name=ckpt_path.name,
    )

    with use_mesh(cfg.mesh):
        zero_cache = l3jax.KVCache.init(random.key(1), cfg, input.shape[0], cfg.max_seq_len)
        next_tokens, logits, cache = l3jax.prefill(input, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.length - 1 : cache.length].get(out_sharding=P(None, None))
        tokens_list = []
        for _ in range(16):
            tokens_list.append(curr_tokens)
            curr_tokens, cache = l3jax.decode_step(curr_tokens, weights, cache, cfg)
        tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:")
    pprint(responses)