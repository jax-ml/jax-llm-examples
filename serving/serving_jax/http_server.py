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

import asyncio
import signal
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from serving_jax import serving_loop


class GenerateRequest(BaseModel):
    id: int
    text: str


def run_http_server(
    serve_loop: serving_loop.ServingLoop,
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    is_server: bool = False,
    shutdown_signal: threading.Event | None = None,
) -> None:
    # lifetime management
    def signal_listener():
        while not shutdown_signal.is_set():
            time.sleep(1)
        signal.raise_signal(signal.SIGKILL)

    threading.Thread(target=signal_listener).start()

    def interrupt_handler(signum, frame):
        if shutdown_signal is not None:
            shutdown_signal.set()

    signal.signal(signal.SIGINT, interrupt_handler)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if shutdown_signal is not None:
            shutdown_signal.set()

    # the HTTP server
    APP = FastAPI(lifespan=lifespan)

    async def generate_generator(id: int, text: str, request: Request) -> AsyncGenerator[str, None]:
        if id in serve_loop.results:  # delete previous request if it exists
            del serve_loop.results[id]

        input = tokenizer_encode(text)
        iter = len(input)  # iterator for finding our current place in a append-only output text
        serve_loop.add_request(serving_loop.UserRequestPrompt(id, input))
        while id not in serve_loop.results:  # wait for the request to be prefilled
            await asyncio.sleep(0.1)
        try:
            result_ref: serving_loop.DecodeResult = serve_loop.results[id]
            while not result_ref.done:  # return text to the client as it becomes available
                if await request.is_disconnected():  # Check if client disconnected
                    print("Client disconnected.")
                    break
                if len(result_ref.token_list) > iter:
                    new_segment, iter = tokenizer_decode(result_ref.token_list[iter:]), len(result_ref.token_list)
                    yield f"{new_segment}"
                await asyncio.sleep(0.1)

            # return the final piece of generate text to the client
            if len(result_ref.token_list) > iter:
                new_segment, iter = tokenizer_decode(result_ref.token_list[iter:]), len(result_ref.token_list)
                yield f"{new_segment}"
        except asyncio.CancelledError:
            pass

    @APP.get("/stream")
    async def stream_response(params: GenerateRequest, request: Request):
        return StreamingResponse(generate_generator(params.id, params.text, request), media_type="text/event-stream")

    @APP.get("/generate")
    async def generate(id: int, text: str):  # generate without output
        print(f"Input text: {text}")
        serve_loop.add_request(serving_loop.UserRequestPrompt(id, tokenizer_encode(text)))
        return Response("OK")

    @APP.get("/retrieve")
    async def retrieve(id: int):
        if id in serve_loop.results:
            return Response(tokenizer_decode(serve_loop.results[id].token_list))
        return Response("NO TEXT")

    @APP.get("/set_generation_length")
    async def set_generation_length(length: int):
        serve_loop.serve_cfg.max_decode_length = max(length, 32)
        return Response("OK")

    @APP.get("/profile")
    async def profile(request: Request):
        del request
        serve_loop.profile_start_time = time.perf_counter()
        return Response("OK")

    @APP.get("/")
    async def root():
        return {"message": "Welcome! Try the /stream-text endpoint."}

    if is_server:
        uvicorn.run(APP, host="0.0.0.0", port=8081, reload=False, server_header=False)
    else:
        while not shutdown_signal.is_set():
            time.sleep(0.1)
