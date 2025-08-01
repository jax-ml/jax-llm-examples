import dataclasses
import time
from pathlib import Path
import threading
import asyncio
import socket
import signal
import time
from typing import AsyncGenerator
from contextlib import asynccontextmanager
import os
from argparse import ArgumentParser

import jax
from jax import random
from jax.sharding import PartitionSpec as P, AxisType, NamedSharding
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import uvicorn

from llama3_jax import model as l3jax
import serving_jax as serving
from serving_jax import attention_cache_utils


TOKENIZER, SERVE_LOOP, SERVING_THREAD, ARGS = None, None, None, None

jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
jax.config.update("jax_enable_empty_arrays", True)

try:  # newer JAX only
    assert False
    my_id = int(socket.gethostname().split("-")[-1]) - 1
    my_ip = socket.getaddrinfo(socket.gethostname(), 80)[0][-1][0]
    jax.config.update("jax_cross_host_transfer_socket_address", f"{my_ip}:{17007 + my_id}")
    jax.config.update("jax_cross_host_transport_addresses", ",".join([f"{my_ip}:0"] * 8))
except: # noqa: E722
    pass

shutdown_signal = threading.Event()

def encode_input(tokenizer, texts, pad_id: int = 0):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True) for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    return np.array([(max_len - len(x)) * [pad_id] + x for x in inputs])


def load_model():
    global SERVE_LOOP, SERVING_THREAD, TOKENIZER, ARGS

    parser = ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Make this node the main server.", default=False)
    ARGS = parser.parse_args()

    #process_idx = int(socket.gethostname().split("-")[-1]) - 1  # a scheme where hosts are (host-1, host-2, ...)
    #jax.distributed.initialize(os.environ["COORDINATOR_ADDRESS"], 2, process_idx)
    print(jax.devices())
    print("-" * 80)
    print(jax.local_devices())

    model_name = "Llama-3.1-8B-Instruct"
    ckpt_path = Path(f"~/{model_name}").expanduser()
    cfg = l3jax.load_config(ckpt_path / "config.json")
    TOKENIZER = l3jax.load_tokenizer(ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json")
    assert ckpt_path.is_dir()
    print("---> Model config loaded")

    # two hosts, different device and host meshes
    local_mesh = jax.make_mesh((1, 8, 1), P("x", "y", "z"), devices=jax.local_devices(), axis_types=(AxisType.Explicit,) * 3)
    decode_mesh, prefill_mesh = local_mesh, local_mesh
    cfg = dataclasses.replace(cfg, mesh=decode_mesh, quant_layer=True, quant_cache=True)
    cfg = dataclasses.replace(cfg, use_prefill_attn_kernel=False, use_decode_attn_kernel=False, max_seq_len=8192)
    cfg = dataclasses.replace(cfg, quant_layer=False, quant_cache=False)
    cfg.quant_cache = True

    decode_weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(dataclasses.replace(cfg, mesh=decode_mesh)))
    prefill_weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(dataclasses.replace(cfg, mesh=prefill_mesh)))

    print("---> Weights loaded")

    serve_cfg = serving.ServingConfig(decode_steps=32, max_decode_length=64)
    #decode_cache = l3jax.KVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size)
    #decode_cache.get_sequence = attention_cache_utils.kvcache_get_entry
    #decode_cache.insert_sequences = attention_cache_utils.kvcache_update_cache
    decode_cache = l3jax.PagedKVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size, 2048, 32)
    decode_cache.get_sequence = attention_cache_utils.batch_paged_get_entry
    decode_cache.insert_sequences = attention_cache_utils.batch_paged_update_sequences
    SERVE_LOOP = serving.ServingLoop(
        serve_cfg, cfg, l3jax.prefill, prefill_weights, l3jax.decode_step, decode_weights, decode_cache, ARGS.server
    )
    print("---> Created the serving loop")

    def serve_forever():
        try:
            while not shutdown_signal.is_set():
                SERVE_LOOP.serving_step()
        except:  # noqa: E722
            import traceback
            print(traceback.format_exc(), flush=True)
        finally:
            print("Received a shutdown signal")
            time.sleep(0.1)
            signal.raise_signal(signal.SIGKILL)  # shut down the web server
        print("Exiting the serving loop")

    SERVING_THREAD = threading.Thread(target=serve_forever)
    SERVING_THREAD.start()


########################################################################################################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    shutdown_signal.set()


_ = load_model()
APP = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    id: int
    text: str


#async def generate_generator(params: GenerateRequest, request: Request) -> AsyncGenerator[str, None]:
async def generate_generator(id: int, text: str, request: Request) -> AsyncGenerator[str, None]:
    if id in SERVE_LOOP.results:
        del SERVE_LOOP.results[id]

    input = encode_input(TOKENIZER, [text])[0].tolist()
    iter = len(input)
    SERVE_LOOP.add_request(serving.UserRequestPrompt(id, input))
    while id not in SERVE_LOOP.results:
        await asyncio.sleep(0.1)
    try:
        result: serving.DecodeResult = SERVE_LOOP.results[id]
        while not result.done:
            if await request.is_disconnected():  # Check if client disconnected
                print("Client disconnected.")
                break
            if len(result.token_list) > iter:
                new_segment, iter = TOKENIZER.decode(result.token_list[iter:]), len(result.token_list)
                yield f"{new_segment}"
            await asyncio.sleep(0.1)  # Stream a new message every 1 second
        if len(result.token_list) > iter:
            new_segment, iter = TOKENIZER.decode(result.token_list[iter:]), len(result.token_list)
            yield f"{new_segment}"
    except asyncio.CancelledError:
        pass
    finally:
        pass


@APP.get("/stream")
async def stream_response(params: GenerateRequest, request: Request):
    return StreamingResponse(generate_generator(params.id, params.text, request), media_type="text/event-stream")


@APP.get("/generate")
async def generate(id: int, text: str):  # generate without output
    print(f"Input text: {text}")
    SERVE_LOOP.add_request(serving.UserRequestPrompt(id, encode_input(TOKENIZER, [text])[0].tolist()))
    return Response("OK")


@APP.get("/retrieve")
async def retrieve(id: int):
    if id in SERVE_LOOP.results:
        return Response(TOKENIZER.decode(SERVE_LOOP.results[id].token_list))
    return Response("NO TEXT")


@APP.get("/set_generation_length")
async def set_generation_length(length: int):
    SERVE_LOOP.serve_cfg.max_decode_length = max(length, 32)
    return Response("OK")


@APP.get("/profile")
async def profile(request: Request):
    del request
    SERVE_LOOP.profile_start_time = time.perf_counter()
    return Response("OK")


@APP.get("/")
async def root():
    return {"message": "Welcome! Try the /stream-text endpoint."}


if __name__ == "__main__":
    if ARGS.server:
        uvicorn.run(APP, host="0.0.0.0", port=8081, reload=False, server_header=False)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown_signal.set()
