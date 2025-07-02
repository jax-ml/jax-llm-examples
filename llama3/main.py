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

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import use_mesh, AxisType, PartitionSpec as P
import numpy as np

from llama3_jax import model as l3jax
from configs import llama_8b_config, llama_70b_config, llama_405b_config


def encode_input(tokenizer, texts: list[str], model_name: str, pad_id: int = 0):
    if tokenizer is None:
        return random.randint(random.key(0), (8, 23), 0, 23000)
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

QUANT = True

if __name__ == "__main__":
    #jax.distributed.initialize()  # if you want to run multi-host

    tokenizer = None
    mesh = jax.make_mesh(
        (1, 8, jax.device_count() // 8), ("x", "y", "z"), devices=jax.devices(), axis_types=(AxisType.Explicit,) * 3
    )

    for cfg in [llama_8b_config, llama_70b_config, llama_405b_config]:
        cfg = dataclasses.replace(cfg, mesh=mesh, quant_layer=QUANT, quant_cache=QUANT)
        weights = l3jax.Weights.init(random.key(0), cfg)

        input = encode_input(
            tokenizer,
            [
                "Tell me your name",
                "What is the weather like expressed in long prose in Old English",
                "Do you like ice cream, be extremely precise",
            ],
            model_name="llama_model",
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
