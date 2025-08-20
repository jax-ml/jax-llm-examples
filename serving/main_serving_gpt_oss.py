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
import socket
import threading
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import serving_jax as serving
from gpt_oss_jax import model as gpt_jax
from jax import random
from jax.sharding import AxisType
from serving_jax import attention_cache_utils as attn_utils

Config = Any

jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

try:  # newer JAX only
    my_id = int(socket.gethostname().split("-")[-1])
    my_ip = socket.getaddrinfo(socket.gethostname(), 80)[0][-1][0]
    jax.config.update("jax_cross_host_transfer_socket_address", f"{my_ip}:{17007 + my_id}")
    jax.config.update("jax_cross_host_transport_addresses", ",".join([f"{my_ip}:0"] * 8))
except:  # noqa: E722
    pass


def encode_input(tokenizer, texts, pad_id: int = gpt_jax.PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True) for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    return np.array([(max_len - len(x)) * [pad_id] + x for x in inputs])


tokenizer_encode = lambda tokenizer, text: encode_input(tokenizer, [text])[0].tolist()
tokenizer_decode = lambda tokenizer, tokens: tokenizer.decode(tokens)


def distributed_init():
    # for TPU
    jax.distributed.initialize()

    # for GPU/CPU
    # process_idx = int(socket.gethostname().split("-")[-1]) - 1  # a scheme where hosts are (host-1, host-2, ...)
    # jax.distributed.initialize(os.environ["COORDINATOR_ADDRESS"], 2, process_idx)
    # jax.distributed.initialize()


def main():
    parser = ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Make this node the main server.", default=False)
    ARGS = parser.parse_args()

    distributed_init()
    devices = jax.devices()  # this helps catch distributed errors quickly

    model_name = "gpt_oss_20b-quant"
    ckpt_path = Path(f"~/bucket/gpt_oss_jax/{model_name}").expanduser()
    cfg = gpt_jax.load_config(ckpt_path / "config.json")
    tokenizer = gpt_jax.load_tokenizer(ckpt_path)
    assert ckpt_path.is_dir()
    print("---> Model config loaded")

    # two hosts, different device and host meshes
    decode_mesh = jax.make_mesh((1, 2, 2), ("x", "y", "z"), devices=devices, axis_types=(AxisType.Explicit,) * 3)
    prefill_mesh = jax.make_mesh((1, 2, 2), ("x", "y", "z"), devices=devices, axis_types=(AxisType.Explicit,) * 3)
    cfg = dataclasses.replace(cfg, mesh=decode_mesh, quant_moe=True, quant_cache=True)
    cfg = dataclasses.replace(cfg, use_prefill_attn_kernel=False, use_decode_attn_kernel=False, max_seq_len=2048)
    cfg.quant_cache = True

    weights_formats, decode_cache_formats = gpt_jax.optimal_formats(dataclasses.replace(cfg, mesh=decode_mesh))
    decode_weights = gpt_jax.load_pytree(ckpt_path, weights_formats)
    weights_formats, prefill_cache_formats = gpt_jax.optimal_formats(dataclasses.replace(cfg, mesh=prefill_mesh))
    # prefill_weights = gpt_jax.load_pytree(ckpt_path, weights_formats)
    prefill_weights = decode_weights

    print("---> Weights loaded")

    serve_cfg = serving.ServingConfig(decode_steps=32, max_decode_length=64, prefix_chunk_size=64)
    decode_cache = gpt_jax.KVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size, cfg.max_seq_len)
    decode_cache = jax.tree.map(lambda x, sds: jax.device_put(x, sds.sharding), decode_cache, decode_cache_formats)
    decode_cache = serving.AttentionWrapper(
        decode_cache, attn_utils.kvcache_get_sequence, attn_utils.kvcache_insert_sequences
    )
    # decode_cache = gpt_jax.PagedKVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size, 2048, 32)
    # decode_cache = serving.AttentionWrapper(
    #     decode_cache, attn_utils.paged_kvcache_get_sequence, attn_utils.paged_kvcache_insert_sequences
    # )

    prefill_cache = gpt_jax.KVCache.init(random.key(0), dataclasses.replace(cfg, mesh=prefill_mesh), serve_cfg.prefill_batch_size, 2048)
    prefill_cache = jax.tree.map(lambda x, sds: jax.device_put(x, sds.sharding), prefill_cache, prefill_cache_formats)
    prefill_cache = serving.AttentionWrapper(
        prefill_cache, attn_utils.kvcache_get_sequence, attn_utils.kvcache_insert_sequences
    )

    sampler = partial(jnp.argmax, axis=-1)

    @partial(jax.jit, donate_argnames=("cache",))
    def forward_fn(inputs, weights, cache, cfg):
        logits, cache = gpt_jax.forward(inputs, (inputs != gpt_jax.PAD_ID).astype(jnp.int32), weights, cfg, cache)
        return sampler(logits), cache

    serve_loop = serving.ServingLoop(
        serve_cfg, cfg, forward_fn, prefill_weights, prefill_cache, decode_weights, decode_cache, ARGS.server
    )
    print("---> Created the serving loop")

    shutdown_signal = threading.Event()
    serve_loop.serve_forever(shutdown_signal)

    serving.run_http_server(
        serve_loop,
        partial(tokenizer_encode, tokenizer),
        partial(tokenizer_decode, tokenizer),
        ARGS.server,
        shutdown_signal=shutdown_signal,
    )

    return


if __name__ == "__main__":
    main()

########################################################################################################################
