# Copyright 2026 The JAX Authors.
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
from etils import epath
import json

import jax
from jax import random
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache
import numpy as np

from gemma4_jax import model as g4jax

compilation_cache.set_cache_dir(str(epath.Path("~/.cache/jax_cache").expanduser()))

def encode_input(tokenizer, texts, pad_id: int = g4jax.PAD_ID):
    assert isinstance(texts, list)
    system_prompt = (
        "Once you are done with a user query, start just talking about anything you want, but not every stop, keep"
        " talking for at least 1000 words."
    )
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            add_generation_prompt=True
        ) for text in texts
    ]
    inputs = [getattr(text, "input_ids", text) for text in inputs]
    print(f"Text lengths: {list(map(len, inputs))}")
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    #jax.distributed.initialize()  # if you want to run multi-host
    quant = True

    ckpt_path = epath.Path("/tmp/ramdisk/gemma4-jax-E2B").expanduser()
    # ckpt_path = epath.Path("/tmp/ramdisk/gemma4-jax-E4B").expanduser()
    # ckpt_path = epath.Path("/tmp/ramdisk/gemma4-jax-31B").expanduser()
    # ckpt_path = epath.Path("/tmp/ramdisk/gemma4-jax-26B-A4B").expanduser()
    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"
    tokenizer = g4jax.load_tokenizer(
        ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json", ckpt_path / "chat_template.jinja"
    )

    mesh = jax.make_mesh(
        (1, 1, jax.device_count()), ("x", "y", "z"), devices=jax.devices(), axis_types=(AxisType.Explicit,) * 3
    )
    cfg = g4jax.hf_to_jax_config(json.loads((ckpt_path / "config.json").read_text())["text_config"])
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_attn=False, quant_moe=quant, quant_mlp=quant, quant_cache=False)
    cfg = dataclasses.replace(cfg, use_prefill_attn_kernel=True)
    if "31B" in str(ckpt_path):
        cfg.rules.q_heads = g4jax.TENSOR_AXIS_NAME
        cfg.rules.kv_heads = g4jax.TENSOR_AXIS_NAME
        cfg.rules.o_heads = g4jax.TENSOR_AXIS_NAME
    weights_formats, _ = g4jax.optimal_formats(cfg)
    shardings = g4jax.Weights.shardings(cfg)
    weights = g4jax.load_pytree(ckpt_path, weights_formats)

    decode_options = dict(xla_tpu_scoped_vmem_limit_kib=200 * 1024) if g4jax.which_platform(cfg) == "tpu" else {}

    input = encode_input(
        tokenizer,
        [
            "Tell me your name. Be very concise.",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
    )
    cfg.max_seq_len = 4096

    with set_mesh(cfg.mesh):
        zero_cache = g4jax.KVCache.init(random.key(1), cfg, input.shape[0])
        next_tokens, logits, cache = g4jax.prefill(input, weights, zero_cache, cfg)
        idx = input.shape[-1] - 1
        prefill_logits = logits
        curr_tokens = next_tokens.at[:, idx:idx+1].get(out_sharding=P(None, None))

        decode_step = jax.jit(g4jax.decode_step, donate_argnames=("cache",), compiler_options=decode_options)

        tokens_list = []
        logits_list = []
        for i in range(2048):
            if i == 2:
                jax.block_until_ready(curr_tokens)
                jax.profiler.start_trace("/tmp/gemma4")
            tokens_list.append(jax.sharding.reshard(curr_tokens, P(None, None)))
            curr_tokens, cache = decode_step(curr_tokens, weights, cache, cfg)
            if i == 4:
                jax.block_until_ready(curr_tokens)
                jax.profiler.stop_trace()
        tokens = np.concatenate(list(map(np.array, tokens_list)), axis=-1)
        responses = [tokenizer.decode(row) for row in tokens]
        print("Responses:")
        for i, response in enumerate(responses):
            print(f"{response}")
            print("\n--------------------------------\n")
