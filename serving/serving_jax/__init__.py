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
from functools import partial
from typing import Any, Callable
import math
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import json
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, use_mesh

try:
    from jax.experimental.shard import auto_axes
except ModuleNotFoundError:
    from jax.sharding import auto_axes
from jax._src import distributed

from jax._src.lib import xla_client as xc
import numpy as np

from .cross_host import transfer_tree_A2B


KVCache, Weights, Config = Any, Any, Any
PyTree, PyTreeStruct = Any, Any

TIME_AXIS = 2
USE_PREFIX_CACHE = True  # the eviction mechanism is extremely simple right now
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)

########################################################################################################################
# device put for cross-process/hosts transfers #########################################################################
########################################################################################################################


def unsafe_device_put(xs: PyTree, spec: PyTree, dest_mesh: Mesh):
    """Fastest, but local single-process JAX only for now."""
    xs_flat, xs_struct = jax.tree.flatten(xs)
    shardings_list = [NamedSharding(dest_mesh, s) for s in jax.tree.leaves(spec)]
    devices_list = [s._internal_device_list for s in shardings_list]
    copy_semantics = [xc.ArrayCopySemantics.ALWAYS_COPY] * len(devices_list)
    out = xc.batched_copy_array_to_devices_with_sharding(xs_flat, devices_list, shardings_list, copy_semantics)
    return jax.tree.unflatten(xs_struct, out)


def jax_device_put(xs: PyTree, sharding: PyTree):
    """Async, available in future JAX."""
    is_source = len(getattr(jax.tree.leaves(xs)[0], "addressable_shards", [])) > 0
    if is_source:
        return jax.device_put(xs, sharding)
    else:
        empty_arrays = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(x.shape, x.sharding, [], dtype=x.dtype), xs
        )
        return jax.device_put(empty_arrays, sharding)


def jit_device_put(xs: PyTree, sharding: PyTree):
    """Most compatabile, uses jit, so requires blocking dispatch."""
    jax.sharding.set_mesh(None)  # not compatible with context mesh
    meshA, meshB = jax.tree.leaves(xs)[0].sharding.mesh, jax.tree.leaves(sharding)[0].mesh
    return transfer_tree_A2B(xs, meshA, meshB)


device_put = jit_device_put  # the most compatible options currently, but NOT async, need


def _ensure_all_args_on_mesh(*args, mesh: Mesh):
    args_len = len(args)
    if not all(jax.tree.leaves(arg)[0].sharding.mesh == mesh for arg in args):
        _correct_mesh = lambda value: jax.tree.leaves(value)[0].sharding.mesh == mesh
        _args = {i: arg for i, arg in enumerate(args) if not _correct_mesh(arg)}
        if len(_args) > 0:
            args = dict(enumerate(args)) | device_put(_args, like_shard(_args, mesh))
            args = tuple(args[i] for i in range(len(args)))
    return args if args_len > 1 else args[0]


########################################################################################################################
# trie utils ###########################################################################################################
########################################################################################################################

_GLOBAL_NODE_ID = 0


@dataclasses.dataclass
class OffloadedValue:
    ref: str | np.ndarray
    spec: Any
    shape_dtypes: Any


@dataclasses.dataclass
class TrieNode:
    id: int
    key: jax.Array
    value: PyTree | OffloadedValue
    children: list["TrieNode"] = dataclasses.field(default_factory=list)
    child_keys: jax.Array | None = None
    lock: "threading.Lock | None" = None
    usage: int = 1

    def __repr__(self, indent: int = 0):
        lines = ["  " * indent + "TrieNode("]
        lines.append(("  " * indent) + f"  key={str(self.key.tolist() if hasattr(self.key, 'tolist') else self.key)},")
        lines.append(("  " * indent) + f"  usage={self.usage},")
        if is_type(self.value, OffloadedValue):
            lines.append(("  " * indent) + f"  value={self.value.ref},")
        else:
            lines.append(
                ("  " * indent)
                + f"  value={jax.tree.map(jax.typeof, self.value) if self.value is not None else 'None'},"
            )
        lines.append(
            ("  " * indent) + f"  child_keys={jax.typeof(self.child_keys) if self.child_keys is not None else 'None'},"
        )
        lines.append(("  " * indent) + "  children=[")
        if self.children:
            for child in self.children:
                lines.append(f"{child.__repr__(indent + 2)},")
            lines.append("  " * indent + "  ],")
        else:
            lines[-1] += "],"
        lines.append("  " * indent + ")")
        return "\n".join(lines)

    @staticmethod
    def new_id():
        global _GLOBAL_NODE_ID
        _GLOBAL_NODE_ID += 1
        return _GLOBAL_NODE_ID - 1

    @staticmethod
    def _dist_to_key(key, keys, mask, pad_idx: int):
        invalid_rows = np.all(keys == pad_idx, axis=-1)
        return np.where(invalid_rows, 2**30, np.sum(mask * np.abs(key - keys), axis=-1))

    @staticmethod
    def _append_key(keys, new_key, keys_len: int, pad_idx: int):
        if keys is None:
            return new_key[None, ...]  # 2 ** 0 power of 2
        if keys_len == keys.shape[0]:  # need to double the keys buffer
            new_buf = np.pad(
                new_key[None, ...], ((0, keys.shape[0] - 1), (0, 0)), mode="constant", constant_values=pad_idx
            )
            return np.concatenate([keys, new_buf], 0)
        else:
            keys[keys_len, ...] = new_key
            return keys

    @staticmethod
    def _pad_to_multiple_of(sequence: jax.Array, chunk_size: int, pad_idx: int):
        sequence_pad_len = math.ceil(sequence.size / chunk_size) * chunk_size
        return np.pad(sequence, ((0, sequence_pad_len - sequence.shape[-1])), mode="constant", constant_values=pad_idx)

    @staticmethod
    def _overlap_dist(key1, key2, mask):
        return np.sum(np.cumsum(np.logical_not(mask & (key1 == key2)), axis=-1) == 0, axis=-1)


@partial(jax.jit, static_argnames=("axis", "chunk_size", "ns"))
def _split(val: jax.Array | list[jax.Array], axis: int, chunk_size: int, ns: int) -> list[jax.Array]:
    spec = jax.tree.map(lambda x: [x] * ns, like_spec(val))

    def _fn(val):
        axis_ = axis % val.ndim
        size = val.shape[axis_]
        if size < chunk_size * ns:
            min_len = chunk_size * ns
            val = jnp.pad(val, [(0, 0) if i != axis_ else (0, min_len - val.shape[axis_]) for i in range(val.ndim)])
        index = [slice(None) if i != axis_ else slice(0, ns * chunk_size) for i in range(val.ndim)]
        return jnp.split(val[*index], ns, axis=axis_)[:ns]

    return auto_axes(lambda vals: jax.tree.map(_fn, vals), out_sharding=spec)(val)


@partial(jax.jit, static_argnames=("split_axis",))
def _concat(values, split_axis: int):
    _fn = lambda vals: jax.tree.map(lambda *args: jnp.concatenate(args, axis=split_axis), *vals)
    return auto_axes(_fn, out_sharding=like_spec(values[0]))(values)


def insert_prefix(
    prefix_cache: TrieNode,
    sequence: jax.Array,
    value: PyTree,
    *,
    chunk_size: int,
    split_axis: int,
    pad_idx: int = 2**30,
    executor: ThreadPoolExecutor | None = None,
    mesh: Any | None = None,
):
    del executor
    sequence = np.array(sequence)
    assert sequence.ndim == 1
    sequence = TrieNode._pad_to_multiple_of(sequence, chunk_size, pad_idx=pad_idx)
    ns = sequence.shape[-1] // chunk_size
    sequence_chunks = np.split(sequence, ns)

    # split the value, but only if it's needed for non-cache hit
    value_chunks = None

    def lazy_get_value(idx):
        nonlocal value_chunks
        if value_chunks is None:
            value_leaves, value_struct = jax.tree.flatten(value)
            with use_mesh(mesh):
                split_leaves = _split(value_leaves, axis=split_axis, chunk_size=chunk_size, ns=ns)
            value_chunks = [jax.tree.unflatten(value_struct, [x[i] for x in split_leaves]) for i in range(ns)]
        return value_chunks[idx]

    # walk the prefix cache tree
    with prefix_cache.lock:
        node = prefix_cache
        for seq_idx, seq in enumerate(sequence_chunks):
            if len(node.children) == 0:
                node.child_keys = TrieNode._append_key(node.child_keys, seq, len(node.children), pad_idx=pad_idx)
                node.children.append(TrieNode(TrieNode.new_id(), seq, lazy_get_value(seq_idx)))
                node = node.children[-1]
                continue
            left_mask, right_mask = (seq != pad_idx), (node.child_keys != pad_idx)
            left_dist = TrieNode._dist_to_key(seq, node.child_keys, left_mask, pad_idx=pad_idx)
            right_dist = TrieNode._dist_to_key(seq, node.child_keys, right_mask, pad_idx=pad_idx)
            left_idx, right_idx = np.argmin(left_dist), np.argmin(right_dist)
            if node.children and right_dist[right_idx] == 0:  # this sequence is longer
                if left_dist[right_idx] > 0:
                    node.children[right_idx].key = seq
                    node.children[right_idx].value = lazy_get_value(seq_idx)
                    node.child_keys[right_idx, :] = seq
                else:  # exact sequence exists
                    node.children[right_idx].usage += 1
                    pass
                node = node.children[right_idx]
            elif left_dist[left_idx] == 0:  # longer sequence already exists
                node.children[left_idx].usage += 1
                assert seq_idx == len(sequence_chunks) - 1
                return
            else:  # no exact match
                node.child_keys = TrieNode._append_key(node.child_keys, seq, len(node.children), pad_idx=pad_idx)
                node.children.append(TrieNode(TrieNode.new_id(), seq, lazy_get_value(seq_idx)))
                node = node.children[-1]


def retrieve_prefix(
    prefix_cache: TrieNode,
    sequence: jax.Array,
    *,
    chunk_size: int,
    split_axis: int,
    pad_idx: int = 2**30,
    executor: ThreadPoolExecutor | None = None,
    mesh: Any | None = None,
):
    sequence, total_match = np.array(sequence), 0
    assert sequence.ndim == 1
    sequence_len, sequence = sequence.size, TrieNode._pad_to_multiple_of(sequence, chunk_size, pad_idx=pad_idx)
    ns = sequence.shape[-1] // chunk_size
    values, sequence_chunks = [], np.split(sequence, ns)

    def _construct_output():
        if sequence_len != total_match:
            return None, total_match
        for i, value in enumerate(values):
            if is_type(value, OffloadedValue):
                _load = lambda value: jax.block_until_ready(device_put(value.ref, like_shard(value.spec, mesh)))
                values[i] = _load(value) if executor is None else executor.submit(_load, value)

        values_future = lambda: [value.result() if hasattr(value, "result") else value for value in values]
        return (executor.submit(values_future) if executor is not None else values_future()), total_match

    node = prefix_cache
    for seq in sequence_chunks:
        if len(node.children) == 0:  # cache ran out of node
            return _construct_output()
        left_mask = seq != pad_idx
        overlaps = TrieNode._overlap_dist(node.child_keys, seq, left_mask)
        max_idx = np.argmax(overlaps)
        max_overlap = overlaps[max_idx]
        if max_overlap == 0:
            return _construct_output()
        with prefix_cache.lock:
            node.children[max_idx].usage += 1
        values.append(node.children[max_idx].value)
        node, total_match = node.children[max_idx], total_match + max_overlap
        # exit early if the entire chunk wasn't found
        if max_overlap != np.sum(left_mask):
            break
    return _construct_output()


def offload_nodes(prefix_cache: TrieNode, how_many: int = 3):
    # work in progress, not tested, will probably not work
    # TODO: switch to [memories](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)
    node_queue, all_nodes = [prefix_cache], []
    with prefix_cache.lock:
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            for child in node.children:
                node_queue.append(child)
                all_nodes.append(child)
        sorted_nodes = sorted(all_nodes, key=lambda x: x.usage)
        offloaded = 0
        for i, node in enumerate(sorted_nodes):
            if offloaded >= how_many:
                break
            if is_type(node.value, OffloadedValue):
                continue
            value = jax.tree.map(partial(np.asarray, copy=False), jax.device_put(node.value, jax.devices("cpu")[0]))
            node.value = OffloadedValue(value, like_spec(node.value), jax.tree.map(jax.typeof, node.value))


########################################################################################################################
# serving loop #########################################################################################################
########################################################################################################################

next_power_of_2 = lambda x: 2 ** round(math.ceil(math.log2(x)))
like_spec = lambda z: jax.tree.map(lambda x: jax.typeof(x).sharding.spec, z)
like_shard = lambda z, mesh: jax.tree.map(lambda x: NamedSharding(mesh, jax.typeof(x).sharding.spec), z)


@dataclasses.dataclass
class ServingConfig:
    decode_steps: int = 10
    decode_batch_size: int = 16
    prefill_batch_size: int = 4
    prefix_chunk_size: int = 512
    eos_tokens: tuple[int, ...] | jax.Array = ()
    token_pad_idx: int = 0
    max_decode_length: int = 64


@dataclasses.dataclass
class UserRequestPrompt:
    id: int
    text: str


@dataclasses.dataclass
class DecodeResult:
    id: int
    token_list: list[int]
    tokens_decoded: int = 0
    done: bool = False


@dataclasses.dataclass
class PrefillResult:
    id: int
    input: np.ndarray
    next_token: jax.Array
    cache_entry: Any
    len: int


@dataclasses.dataclass
class DecodeWork:
    curr_tokens: jax.Array  # [B, 1] to conform with the general forward fn expecting a sequence dimension
    cache: KVCache
    active_results: list[DecodeResult | None]


@dataclasses.dataclass
class PrefillWork:
    requests: list[UserRequestPrompt]
    to_prefill: list[UserRequestPrompt]
    to_decode: list[PrefillResult]
    pending_prefill: Future | None = None
    pending_cache_retrievals: list[tuple[UserRequestPrompt, Future]] = dataclasses.field(default_factory=list)


def return_request(resp: DecodeResult):
    # an optional callback called with results available on decode nodes only
    # something happens here to output the response to the global queue
    # print(f"Finished request: {resp.id}")
    pass


class SyncServer:
    """A regular local network server for syncing between JAX processes in the multi-process JAX setup."""

    CLIENT = None
    TIMEOUT_SEC = 60

    @staticmethod
    def _get_client():
        if SyncServer.CLIENT is None:
            SyncServer.CLIENT = distributed.global_state.client
        return SyncServer.CLIENT

    @staticmethod
    def barrier(key: str, current_it: int) -> None:
        client = SyncServer._get_client()
        if client is None:
            return
        client.wait_at_barrier(key + str(current_it), timeout_in_ms=SyncServer.TIMEOUT_SEC * 1000)

    @staticmethod
    def broadcast(key: str, current_it: int, value: Any, is_source: bool = False, jsonify: bool = True) -> None:
        client = SyncServer._get_client()
        if client is None:
            return value
        if is_source:
            client.key_value_set(key + str(current_it), json.dumps(value) if jsonify else value)
            return value
        else:
            value = client.blocking_key_value_get(key + str(current_it), SyncServer.TIMEOUT_SEC * 1000)
            return json.loads(value) if jsonify else value


def _make_multistep_decode_fn(decode_fn):
    @partial(jax.jit, static_argnames=("steps",), donate_argnames=("cache",))
    def multistep_decode_fn(curr_tokens, decode_weights, cache, cfg, steps: int = 32):
        def body(carry, _):
            curr_tokens, cache = carry
            next_tokens, cache = decode_fn(curr_tokens, decode_weights, cache, cfg)
            return (next_tokens, cache), next_tokens

        (curr_tokens, cache), output_tokens = jax.lax.scan(body, (curr_tokens, cache), length=steps)
        return (curr_tokens, cache), output_tokens[..., 0].T

    return multistep_decode_fn


def _make_stacked_prefill(prefill_fn):
    def _numpy_pad_tokens(tokens):
        opts = dict(mode="constant", constant_values=0)
        return np.pad(tokens, [(0, 0), (0, next_power_of_2(tokens.shape[-1]) - tokens.shape[-1])], **opts)

    @jax.jit
    def stacked_prefill(inputs, weights, cfg):
        next_tokens, logits, kv_list = prefill_fn(inputs, weights, None, cfg)
        assert len(kv_list) == cfg.num_layers, "The output kv values have to be in a list kv pairs."
        stacked_kv = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *kv_list)
        return next_tokens, logits, stacked_kv

    return lambda inputs, weights, cfg: stacked_prefill(_numpy_pad_tokens(inputs), weights, cfg)


class ServingLoop:
    def __init__(
        self,
        serve_cfg: ServingConfig,
        cfg: Config,
        prefill_fn: Callable,
        prefill_weights: Weights,
        decode_fn: Callable,
        decode_weights: Weights,
        decode_cache: KVCache,
    ):
        self.serve_cfg, self.cfg = serve_cfg, cfg

        # setup decode
        self.decode_fn, self.decode_weights = decode_fn, decode_weights
        self.decode_mesh = [x for x in jax.tree.leaves(decode_weights) if hasattr(x, "sharding")][0].sharding.mesh
        with use_mesh(self.decode_mesh):
            self.decode_work = DecodeWork(None, decode_cache, [None for _ in range(serve_cfg.decode_batch_size)])
            self.decode_work.curr_tokens = jax.device_put(
                jnp.zeros((serve_cfg.decode_batch_size, 1), dtype=jnp.int32), P()
            )
        self.multistep_decode_fn = _make_multistep_decode_fn(self.decode_fn)
        self._update_index = jax.jit(lambda x, i, new: x.at[i, ...].set(new[:, None], mode="drop"))

        def _update_cache_and_index(cache: KVCache, curr_tokens: jax.Array, new_tokens, kvs, batch_idxs, actual_lens):
            length_sort = sorted(
                range(len(kvs)), key=lambda i: jax.tree.leaves(kvs[i])[0].shape[-2]
            )  # sort to minimize variants num
            new_cache = decode_cache.insert_sequences(
                cache, *[[x[i] for i in length_sort] for x in (kvs, batch_idxs, actual_lens)]
            )
            with use_mesh(self.decode_mesh):
                new_curr_tokens = self._update_index(curr_tokens, np.array(batch_idxs), new_tokens)
            return new_cache, new_curr_tokens

        self._update_cache_and_index = _update_cache_and_index
        self.decode_output = (None, None)

        # setup prefill
        self.prefill_fn = staticmethod(_make_stacked_prefill(prefill_fn))
        self.prefill_weights = prefill_weights
        self.prefill_mesh = [x for x in jax.tree.leaves(prefill_weights) if hasattr(x, "sharding")][0].sharding.mesh
        self.prefill_work = PrefillWork([], [], [])
        self.prefix_cache = TrieNode(TrieNode.new_id(), None, None, lock=threading.Lock())
        self._get_index = jax.jit(lambda z, idx: jax.tree.map(lambda x: x[:, idx, ...], z))
        self._get_cache_entry = jax.jit(self.decode_work.cache.get_sequence)

        # setup misc
        self.pending_requests, self.requests_lock, self.results = [], threading.Lock(), {}
        self.pad_id, self.eos_tokens, self.time_axis = 0, np.array(serve_cfg.eos_tokens), TIME_AXIS
        self._background = ThreadPoolExecutor(max_workers=1024)

        # setup profiling
        self.profile_start_time, self.profiling = -1, False

        # setup cache management
        # -1 for missing batch dimensiona and + 1 for layers being stacked
        self.prefix_cache, self._retrieve_prefix, self._insert_prefix = None, None, None
        self.new_prefix_cache()

        # setup the sync server for multi-host
        self._it, self.roles = 0, (("server",) if jax.process_index() == 0 else ())  # main server
        if any(d.id in [d_.id for d_ in self.decode_mesh.devices.reshape(-1)] for d in jax.local_devices()):
            self.roles += ("decode",)  # any node which has decode mesh devices
        if any(d.id in [d_.id for d_ in self.prefill_mesh.devices.reshape(-1)] for d in jax.local_devices()):
            self.roles += ("prefill",)  # any node which has prefill devices
        if any(d.id == min([d_.id for d_ in self.decode_mesh.devices.reshape(-1)]) for d in jax.local_devices()):
            self.roles += ("decode_coordinator",)  # the decode node which holds the smallest decode mesh device
        if any(d.id == min([d_.id for d_ in self.prefill_mesh.devices.reshape(-1)]) for d in jax.local_devices()):
            self.roles += ("prefill_coordinator",)  # the prefill node which holds the smallest prefill mesh device
        self.total_requests = 0

    def decode_step(self):
        # TODO: a more intelligent decision between decode and prefill (adaptive strategies, prefill queue size)

        # 1. add outstanding ready to decode prefill result to the active decode
        #   - some cache entries require some computation, so they're a callable
        #   - some cache entries are not on the correct decode_mesh
        if len(self.prefill_work.to_decode) > 0:
            batch_cache_updates = []
            for i, active_result in enumerate(self.decode_work.active_results):
                if active_result is not None:
                    continue
                if len(self.prefill_work.to_decode) == 0:
                    break
                result: PrefillResult = self.prefill_work.to_decode.pop(0)
                self.decode_work.active_results[i] = DecodeResult(result.id, result.input.tolist())
                with use_mesh(self.decode_mesh):
                    result.cache_entry = result.cache_entry() if callable(result.cache_entry) else result.cache_entry
                result.cache_entry = _ensure_all_args_on_mesh(result.cache_entry, mesh=self.decode_mesh)
                self.results[result.id] = self.decode_work.active_results[i]
                batch_cache_updates.append((result.cache_entry, i, result.len, result.next_token))
                if len(self.prefill_work.to_decode) == 0:
                    break
            if "decode" in self.roles and len(batch_cache_updates) > 0:  # batch cache update
                entries, batch_idxs, lens, next_tokens = map(list, zip(*batch_cache_updates))
                entries = [entry.result() if hasattr(entry, "result") else entry for entry in entries]  # maybe collect
                _control_args = (np.array(next_tokens), entries, batch_idxs, lens)
                self.decode_work.cache, self.decode_work.curr_tokens = self._update_cache_and_index(
                    self.decode_work.cache, self.decode_work.curr_tokens, *_control_args
                )

        if all(x is None for x in self.decode_work.active_results):
            return  # skip decoding if no decoding tasks are present

        # 2. run N decode steps
        output_tokens, output_mapping = [], []
        if "decode" in self.roles:  # cut a corner, don't issue the decode call on non-participating machines
            with use_mesh(self.decode_mesh):
                config = dict(cfg=self.cfg, steps=self.serve_cfg.decode_steps)
                (self.decode_work.curr_tokens, self.decode_work.cache), output_tokens = self.multistep_decode_fn(
                    self.decode_work.curr_tokens, self.decode_weights, self.decode_work.cache, **config
                )
                output_mapping = [
                    [getattr(result, "id", -1) for result in self.decode_work.active_results]
                ] * self.serve_cfg.decode_steps
                output_mapping = np.array(output_mapping).T
            print(
                f"Decoding with fill rate of {np.mean([result is not None for result in self.decode_work.active_results])}"
            )

        # 3. parse output tokens from previous decoding loop to allow for the tokens arrive (delayed EOS detection)
        self.decode_output, (output_tokens, output_mapping) = (output_tokens, output_mapping), self.decode_output
        if output_tokens is not None:
            SyncServer.barrier("output_tokens", self._it)
            if "decode" in self.roles:
                output_tokens = np.array(output_tokens)
                done = np.any(output_tokens[..., None] == self.eos_tokens, (-1, -2)).tolist()  # check for done
                done = [
                    d or getattr(result, "tokens_decoded", 0) >= self.serve_cfg.max_decode_length
                    for d, result in zip(done, self.decode_work.active_results)
                ]
            else:
                output_tokens, done = None, None
            done = SyncServer.broadcast("done_sync", self._it, done, is_source="decode_coordinator" in self.roles)
            if "server" in self.roles:
                for token, id in zip(output_tokens.reshape(-1).tolist(), output_mapping.reshape(-1).tolist()):
                    if id > 0:
                        self.results[id].token_list.append(token)
                        self.results[id].tokens_decoded += 1
            with use_mesh(self.decode_mesh):
                for i, result in enumerate(self.decode_work.active_results):
                    if result is None:
                        continue
                    # 2. check for done sequences; evict them if done and return them
                    if done[i]:
                        if USE_PREFIX_CACHE:
                            sequence = np.array(result.token_list)
                            with use_mesh(self.decode_mesh):
                                cache_entry, _ = self._get_cache_entry(self.decode_work.cache, i)
                            self._background.submit(self._insert_prefix, sequence, cache_entry, mesh=self.decode_mesh)
                        return_request(result)
                        result.done, self.decode_work.active_results[i] = True, None

    def prefill_step(self):
        # 1. check on any finished prefill jobs
        if self.prefill_work.pending_prefill is not None:
            prefill_is_done, is_source = self.prefill_work.pending_prefill.done(), "prefill_coordinator" in self.roles
            prefill_is_done = SyncServer.broadcast("prefill_done", self._it, prefill_is_done, is_source=is_source)
            if prefill_is_done:
                prefill_input, prefill_results = self.prefill_work.pending_prefill.result()
                for i, request in enumerate(prefill_input):
                    with use_mesh(self.prefill_mesh):
                        kv_list = self._get_index(prefill_results, i)
                    id, input = request.id, np.array(request.text)
                    self.prefill_work.to_decode.append(PrefillResult(id, input, input[-1], kv_list, len(input) - 1))
                self.prefill_work.pending_prefill = None

        # 2. triage requests queue into cached (-> decode) and not-cached work (-> prefill)
        new_pending_retrievals = []
        for request, cache_entry_fut in self.prefill_work.pending_cache_retrievals:
            if len(self.prefill_work.to_decode) < self.serve_cfg.decode_batch_size and cache_entry_fut.done():
                with use_mesh(self.decode_mesh):
                    # batch missing (-1) layers concatenated (+1)
                    cache_entry = partial(_concat, cache_entry_fut.result(), self.time_axis - 1 + 1)  # jit work future
                new_decode = PrefillResult(
                    request.id, np.array(request.text), request.text[-1], cache_entry, len(request.text) - 1
                )
                self.prefill_work.to_decode.append(new_decode)
            else:
                new_pending_retrievals.append((request, cache_entry_fut))  # not yet ready
        self.prefill_work.pending_cache_retrievals = new_pending_retrievals

        # 3. check if prefixes are in the cache
        retrieval_results = self._background.map(
            lambda request: (self._retrieve_prefix(np.array(request.text[:-1])), request), self.prefill_work.requests
        )
        for (cache_entry_fut, length), request in retrieval_results:
            if length == len(request.text) - 1:
                self.prefill_work.pending_cache_retrievals.append((request, cache_entry_fut))
                print(f"Found full prefill match in the cache")
            else:
                print(f"Need to prefill the request, only found a match for length {length / (len(request.text) - 1)}")
                self.prefill_work.to_prefill.append(request)
        self.prefill_work.requests.clear()

        if self.prefill_work.pending_prefill is not None:  # a current prefill is still running, skip scheduling another
            return

        # 4. prefill requests to be prefilled
        prefill_input = self.prefill_work.to_prefill[: self.serve_cfg.prefill_batch_size]
        self.prefill_work.to_prefill = self.prefill_work.to_prefill[len(prefill_input) :]
        if len(prefill_input) > 0:
            # disaggregated server via async on a subset of devices
            def _prefill_job():
                max_len = max([len(request.text) for request in prefill_input])
                inputs = [[self.pad_id] * (max_len - len(request.text)) + request.text for request in prefill_input]
                inputs = np.stack([np.array(input) for input in inputs], 0)
                row_pad = self.serve_cfg.prefill_batch_size - inputs.shape[0]
                col_pad = next_power_of_2(inputs.shape[-1]) - inputs.shape[-1]
                inputs = np.pad(inputs, ((0, row_pad), (0, col_pad)), mode="constant", constant_values=self.pad_id)
                cfg = dataclasses.replace(self.cfg, mesh=self.prefill_mesh)
                with use_mesh(self.prefill_mesh):
                    _, _, prefill_results = self.prefill_fn(inputs, self.prefill_weights, cfg)
                    prefill_results = jax.block_until_ready(prefill_results)
                return prefill_input, prefill_results

            self.prefill_work.pending_prefill = self._background.submit(_prefill_job)

    def serving_step(self):
        # this event loop relies on determinism for issuing computation to multiple processes (multi-process JAX)
        # frequent barriers should keep it in sync

        # potentially profile when received the request to #########################################
        should_start_profile = self.profile_start_time > 0 and not self.profiling
        should_start_profile = SyncServer.broadcast(
            "profile", self._it, should_start_profile, is_source="server" in self.roles
        )
        if should_start_profile:
            self.profile_start_time, self.profiling = time.perf_counter(), True
            jax.profiler.start_trace("/tmp/online")
            print("STARTING TRACE")
        should_stop_profile = self.profile_start_time > 0 and time.perf_counter() - self.profile_start_time > 5.0
        should_stop_profile = SyncServer.broadcast(
            "stop_profile", self._it, should_stop_profile, is_source="server" in self.roles
        )
        if should_stop_profile:
            self.profile_start_time, self.profiling = -1, False
            print("STOPPING TRACE")
            jax.profiler.stop_trace()
        # potentially profile when received the request to #########################################

        # sync on the server requests received #####################################################
        SyncServer.barrier("serving_step", self._it)
        self._it, requests = self._it + 1, None
        if "server" in self.roles:
            with self.requests_lock:
                self.pending_requests, requests = [], list(self.pending_requests)
        requests = SyncServer.broadcast("requests", self._it, requests, is_source="server" in self.roles)
        for request in requests:
            self.total_requests += 1
            self.prefill_work.requests.append(UserRequestPrompt(**request))
        # sync on the server requests received #####################################################

        # main event loop work #####################################################################
        self.decode_step()
        self.prefill_step()
        # main event loop work #####################################################################

        # manage cache #############################################################################
        # TODO: test and configure host offloading for the cache
        if USE_PREFIX_CACHE and len(self.prefix_cache.children) > 100:  # clear the cache after 100 root children
            self.new_prefix_cache()
        # manage cache #############################################################################

    def add_request(self, request: UserRequestPrompt):
        with self.requests_lock:
            self.pending_requests.append(dataclasses.asdict(request))

    def new_prefix_cache(self):
        self.prefix_cache = TrieNode(TrieNode.new_id(), None, None, lock=threading.Lock())
        _prefix_opts = dict(chunk_size=self.serve_cfg.prefix_chunk_size)
        _prefix_opts |= dict(split_axis=self.time_axis - 1 + 1, mesh=self.decode_mesh, executor=self._background)
        self._retrieve_prefix = partial(retrieve_prefix, self.prefix_cache, **_prefix_opts)
        self._insert_prefix = partial(insert_prefix, self.prefix_cache, **_prefix_opts)
