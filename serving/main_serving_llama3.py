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
from jax import random
from jax.sharding import AxisType
from llama3_jax import model as l3jax
from serving_jax import attention_cache_utils as attn_utils

Config = Any

jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))

try:  # newer JAX only
    my_id = int(socket.gethostname().split("-")[-1])  # a scheme where hosts end with -HOST_NUM (host-0, host-1, ...)
    my_ip = socket.getaddrinfo(socket.gethostname(), 80)[0][-1][0]
    jax.config.update("jax_cross_host_transfer_socket_address", f"{my_ip}:{17007 + my_id}")
    jax.config.update("jax_cross_host_transport_addresses", ",".join([f"{my_ip}:0"] * 8))
except:  # noqa: E722
    pass


def encode_input(tokenizer, texts, pad_id: int = 0):
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

    model_name = "Llama-3.1-8B-Instruct-quant"
    ckpt_path = Path(f"~/bucket/llama3_jax/{model_name}").expanduser()
    cfg = l3jax.load_config(ckpt_path / "config.json")
    tokenizer = l3jax.load_tokenizer(ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json")
    assert ckpt_path.is_dir()
    print("---> Model config loaded")

    # two hosts, different device and host meshes
    decode_mesh = jax.make_mesh((1, 8, 1), ("x", "y", "z"), devices=devices[:8], axis_types=(AxisType.Explicit,) * 3)
    prefill_mesh = jax.make_mesh((1, 8, 1), ("x", "y", "z"), devices=devices[8:], axis_types=(AxisType.Explicit,) * 3)
    cfg = dataclasses.replace(cfg, mesh=decode_mesh, quant_layer=True, quant_cache=True, max_seq_len=2048)
    cfg_decode, cfg_prefill = dataclasses.replace(cfg, mesh=decode_mesh), dataclasses.replace(cfg, mesh=prefill_mesh)

    decode_weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(cfg_decode))
    prefill_weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(cfg_prefill))
    # prefill_weights = decode_weights

    print("---> Weights loaded")

    serve_cfg = serving.ServingConfig(
        decode_steps=32, max_decode_length=64, prefix_chunk_size=64, max_ondevice_buffers=2048, max_buffers=2048
    )
    decode_cache = l3jax.KVCache.init(random.key(0), cfg_decode, serve_cfg.decode_batch_size)
    decode_cache = attn_utils.AttentionInterface(
        decode_cache, attn_utils.kvcache_get_sequence, attn_utils.kvcache_insert_sequences
    )
    # decode_cache = l3jax.PagedKVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size, 2048, 32)
    # decode_cache = attn_utils.AttentionInterface(
    #     decode_cache, attn_utils.paged_kvcache_get_sequence, attn_utils.paged_kvcache_insert_sequences
    # )

    prefill_cache = l3jax.KVCache.init(random.key(0), cfg_prefill, serve_cfg.prefill_batch_size)
    prefill_cache = attn_utils.AttentionInterface(
        prefill_cache, attn_utils.kvcache_get_sequence, attn_utils.kvcache_insert_sequences
    )

    sampler = partial(jnp.argmax, axis=-1)

    @partial(jax.jit, donate_argnames=("cache",))
    def forward_fn(inputs, weights, cache, cfg):
        logits, cache = l3jax.forward(inputs, (inputs != 0).astype(jnp.int32), weights, cfg, cache)
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


if __name__ == "__main__":
    main()

########################################################################################################################
