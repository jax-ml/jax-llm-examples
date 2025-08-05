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
import contextlib
from functools import partial
from typing import Any, Callable, Sequence
import math
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import json

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, set_mesh

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
# USE_PREFIX_CACHE = False
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
    # jax.sharding.set_mesh(None)  # not compatible with context mesh
    meshA, meshB = jax.tree.leaves(xs)[0].sharding.mesh, jax.tree.leaves(sharding)[0].mesh
    return transfer_tree_A2B(xs, meshA, meshB)


#device_put = jit_device_put  # the most compatible options currently, but NOT async, need
device_put = jax.device_put


def _ensure_all_args_on_mesh(args, mesh: Mesh):
    if not all(jax.tree.leaves(arg)[0].sharding.mesh == mesh for arg in args):
        _correct_mesh = lambda value: jax.tree.leaves(value)[0].sharding.mesh == mesh
        _args = {i: arg for i, arg in enumerate(args) if not _correct_mesh(arg)}
        if len(_args) > 0:
            args = dict(enumerate(args)) | device_put(_args, like_shard(_args, mesh))
            args = tuple(args[i] for i in range(len(args)))
    return args


########################################################################################################################
# kv cache buffer management ###########################################################################################
########################################################################################################################


@partial(jax.jit, static_argnames=("axis", "chunk_size", "ns"))
def _split(val: jax.Array | list[jax.Array], axis: int, chunk_size: int, ns: int) -> list[jax.Array]:
    def _fn(val):
        axis_ = axis % val.ndim
        size = val.shape[axis_]
        if size < chunk_size * ns:
            min_len = chunk_size * ns
            val = jnp.pad(val, [(0, 0) if i != axis_ else (0, min_len - val.shape[axis_]) for i in range(val.ndim)])
        index = [slice(None) if i != axis_ else slice(0, ns * chunk_size) for i in range(val.ndim)]
        return jnp.split(val[*index], ns, axis=axis_)[:ns]

    val_leaves, val_structure = jax.tree.flatten(val)
    spec = [[x] * ns for x in like_spec(val_leaves)]
    split_leaves = auto_axes(lambda vals: [_fn(val) for val in vals], out_sharding=spec)(val_leaves)
    return [jax.tree.unflatten(val_structure, [x[i] for x in split_leaves]) for i in range(ns)]


@partial(jax.jit, static_argnames=("split_axis",))
def _concat(values, split_axis: int):
    _fn = lambda vals: jax.tree.map(lambda *args: jnp.concatenate(args, axis=split_axis), *vals)
    return auto_axes(_fn, out_sharding=like_spec(values[0]))(values)


class KVBufferStore:
    def __init__(self):
        self.usecount, self.ondevice, self._store, self.unique_id, self.livecount = {}, {}, {}, 18, 0

    def _get_unique_buffer_ids(self, n: int):
        ids = list(range(self.unique_id, self.unique_id + n))
        self.unique_id += n
        return ids

    def offload_buffers(self, how_many: int):
        if how_many == 0:
            return
        candidates = sorted(self._store.keys(), key=lambda i: self.usecount[i] if self.ondevice[i] else 2**60)
        for i in candidates[:how_many]:
            if self.ondevice[i]:
                host_shrd = jax.tree.map(lambda x: x.sharding.with_memory_kind("pinned_host"), self._store[i])
                self._store[i] = jax.device_put(self._store[i], host_shrd)
                self.ondevice[i] = False
                self.livecount -= 1

    def load(self, id: int):
        if isinstance(id, (tuple, list)):
            return [self.load(i) for i in id]
        if self.ondevice[id]:
            return self._store[id]
        self.ondevice[id] = True
        self.livecount += 1
        device_shrd = jax.tree.map(lambda x: x.sharding.with_memory_kind("device"), self._store[id])
        self._store[id] = jax.device_put(self._store[id], device_shrd)
        return self._store[id]

    def delete(self, id: int):
        if isinstance(id, (list, tuple)):
            return [self.delete(i) for i in id]
        self.livecount -= self.ondevice[id]
        del self.usecount[id], self.ondevice[id], self._store[id]

    def store(self, id: int, val: Any):
        if isinstance(id, (tuple, list)):
            return [self.store(i, v) for i, v in zip(id, val)]
        self.livecount += 1
        self.usecount[id], self.ondevice[id], self._store[id] = 1, True, val

    def mark_visited(self, id: int):
        if isinstance(id, (list, tuple)):
            return [self.mark_visited(i) for i in id]
        self.usecount[id] += 1


BUFFER_STORE = KVBufferStore()

########################################################################################################################
# trie utils ###########################################################################################################
########################################################################################################################

EMPTY, HASH_BITWIDTH = -1, 1

@dataclasses.dataclass
class ChildKeys:
    keys: np.ndarray
    keys_hash: np.ndarray
    keys_hash_mask: np.ndarray
    key_lens: np.ndarray
    num: int = 0


def _hash_encode(v: np.ndarray, hash_bitwidth: int = HASH_BITWIDTH, pad_idx: int = EMPTY):
    v, last_dim = v.astype(np.int64), min(64 // hash_bitwidth, v.shape[-1])
    v_, el_mask = v.reshape(v.shape[:-1] + (-1, last_dim)), (1 << hash_bitwidth) - 1
    mask = np.bitwise_or.reduce(((v_ != pad_idx) * el_mask) << (hash_bitwidth * np.arange(v_.shape[-1])), axis=-1)
    h = np.bitwise_or.reduce((v_ & el_mask) << (hash_bitwidth * np.arange(v_.shape[-1])), axis=-1)
    return h, mask


def _prefilter_on_hash(
    w: np.ndarray,
    keys: np.ndarray,
    vh: np.ndarray,
    vm: np.ndarray,
    hash_bitwidth: int = HASH_BITWIDTH,
    pad_idx: int = EMPTY,
):
    wh, wm = _hash_encode(w, hash_bitwidth=hash_bitwidth, pad_idx=pad_idx)
    inv_match = (wh ^ vh) & vm & wm
    # count full hash chunk matches, but don't miss sequences not matching at least one full hash
    match_len = np.sum(np.cumsum(inv_match, axis=-1) == 0, axis=-1) + (w[0] == keys[:, 0])
    max_match_len = max(np.max(match_len), 1)
    return np.where(match_len == max_match_len)[0]


def _fast_pad(x, size, axis, pad_val=0):
    new_buf = pad_val * np.ones([size - s if i == axis else s for i, s in enumerate(x.shape)], dtype=x.dtype)
    return np.concat([x, new_buf], axis)


@dataclasses.dataclass
class TrieNode:
    value: int
    children: list["TrieNode"] = dataclasses.field(default_factory=list)
    child_keys: ChildKeys | None = None
    lock: "threading.Lock | None" = None
    usage: int = 1

    def __repr__(self, indent: int = 0):
        lines = [f"TrieNode(value={self.value}, usage={self.usage}, children={{"]
        if len(self.children) == 0:
            lines[-1] = lines[-1][:-1] + "})"
        else:
            for i, child in enumerate(self.children):
                child_key = self.child_keys.keys[i, : self.child_keys.key_lens[i]].tolist()
                lines.append(f"{' ' * indent}  {child_key}: {child.__repr__(indent + 2).strip()},")
            lines.append(")")
        return "\n".join([(" " * indent) + line for line in lines])

    @staticmethod
    def _overlap(child_keys: ChildKeys, key, key_len, pad_idx: int = EMPTY):
        keys = child_keys.keys[: child_keys.num, :]
        keys_hash = child_keys.keys_hash[: child_keys.num, :]
        keys_hash_mask = child_keys.keys_hash_mask[: child_keys.num, :]

        # pre-filter sequences
        relevant_idx = _prefilter_on_hash(key, keys, keys_hash, keys_hash_mask, pad_idx=pad_idx)
        if len(relevant_idx) == 0:
            return np.zeros((child_keys.num,), dtype=np.int32), np.zeros((child_keys.num,), dtype=np.int32)
        keys = keys[relevant_idx, :]

        mask = np.cumsum((key == keys) | (key == pad_idx) | (keys == pad_idx), -1) == np.arange(1, key.shape[-1] + 1)
        overlap = np.zeros((child_keys.num,), dtype=np.int32)
        overlap[relevant_idx] = np.sum(mask, axis=-1)
        return np.minimum(overlap, key_len), np.minimum(overlap, child_keys.key_lens[: child_keys.num])

    @staticmethod
    def _append_key(keys: ChildKeys | None, new_key: np.ndarray, key_len: int, pad_idx: int = EMPTY):
        if keys is None:
            key_hash, key_hash_mask = _hash_encode(new_key[None, :], pad_idx=pad_idx)
            return ChildKeys(new_key[None, :], key_hash, key_hash_mask, np.array([key_len], dtype=np.int32), 1)
        if keys.num == keys.keys.shape[0]:  # need to double the keys buffer
            keys.keys = _fast_pad(keys.keys, 2 * keys.num, 0, 0)
            keys.key_lens = _fast_pad(keys.key_lens, 2 * keys.num, 0)
            keys.keys_hash = _fast_pad(keys.keys_hash, 2 * keys.num, 0, 0)
            keys.keys_hash_mask = _fast_pad(keys.keys_hash_mask, 2 * keys.num, 0, 0)
        keys.keys[keys.num, :], keys.key_lens[keys.num] = new_key, key_len
        keys.keys_hash[keys.num, :], keys.keys_hash_mask[keys.num, :] = _hash_encode(new_key, pad_idx=pad_idx)
        keys.num += 1
        return keys

    @staticmethod
    def _delete_keys(keys: ChildKeys, delete_idxs: np.ndarray):
        if keys is None:
            return
        mask = np.ones(keys.keys.shape[0], dtype=bool)
        mask[np.array(list(delete_idxs) if isinstance(delete_idxs, set) else delete_idxs, int)] = False
        if np.sum(mask) == 0:
            return None
        num = max(keys.num - sum(1 for idx in set(delete_idxs) if idx < keys.num), 0)
        return ChildKeys(*(z[mask, ...] for z in [keys.keys, keys.keys_hash, keys.keys_hash_mask, keys.key_lens]), num)

    @staticmethod
    def _pad_to_multiple_of(sequence: np.ndarray, chunk_size: int, pad_idx: int = EMPTY):
        sequence_pad_len = math.ceil(sequence.size / chunk_size) * chunk_size
        return _fast_pad(sequence, sequence_pad_len, 0, pad_idx)


def insert_prefix(root: TrieNode, sequence: np.ndarray, ref_vals: list[int], *, chunk_size: int, pad_idx: int = 2**30):
    if len(sequence) == 0:
        return [], [], []
    sequence = np.array(sequence)
    assert sequence.ndim == 1
    sequence_len, sequence = sequence.size, TrieNode._pad_to_multiple_of(sequence, chunk_size, pad_idx=pad_idx)
    ns = sequence.shape[-1] // chunk_size
    seq_actual_lens = [(chunk_size if i != ns - 1 else (sequence_len - (ns - 1) * chunk_size)) for i in range(ns)]
    sequence_chunks = np.split(sequence, ns)
    if len(ref_vals) < ns:
        msg = f"Pass at least as many references as there are chunks (size={chunk_size}) in the sequence "
        msg += f" (size={sequence_len}), so expected at least {ns} references, got {len(ref_vals)=} instead."
        raise ValueError(msg)
    visited_refs, store_refs, delete_refs = [], [], []  # which refs to retain and which to delete

    # walk the prefix cache tree
    with root.lock:
        node = root
        for seq_idx, (seq, seq_len) in enumerate(zip(sequence_chunks, seq_actual_lens)):
            if len(node.children) > 0:
                left_match, right_match = TrieNode._overlap(node.child_keys, seq, seq_len, pad_idx=pad_idx)
                best_idx = np.argmax(left_match)
                left_match, right_match = left_match[best_idx], right_match[best_idx]
            else:
                left_match, right_match, best_idx = 0, 0, 2**30  # case 0: no children, add new child
            if left_match != seq_len:  # append new node
                node.child_keys = TrieNode._append_key(node.child_keys, seq, seq_len, pad_idx=pad_idx)
                node.children.append(TrieNode(int(ref_vals[seq_idx])))
                store_refs.append(int(ref_vals[seq_idx]))
                node = node.children[-1]
            elif right_match < left_match:  # replace the node
                delete_refs.append(node.children[best_idx].value)
                node.children[best_idx] = TrieNode(int(ref_vals[seq_idx]))
                node.child_keys.keys[best_idx, :], node.child_keys.key_lens[best_idx] = seq, seq_len
                store_refs.append(int(ref_vals[seq_idx]))
                node = node.children[best_idx]
            else:  # full match, do nothing
                if best_idx > len(node.children):
                    break
                visited_refs.append(int(node.children[best_idx].value))
                node = node.children[best_idx]
    visited_refs = list(set(visited_refs) | set(store_refs))
    return visited_refs, store_refs, delete_refs


def retrieve_prefix(root: TrieNode, sequence: np.ndarray, *, chunk_size: int, pad_idx: int = 2**30):
    sequence, total_match, ref_vals = np.array(sequence), 0, []
    assert sequence.ndim == 1
    sequence_len, sequence = sequence.size, TrieNode._pad_to_multiple_of(sequence, chunk_size, pad_idx=pad_idx)
    ns = sequence.shape[-1] // chunk_size
    seq_actual_lens = [(chunk_size if i != ns - 1 else (sequence_len - (ns - 1) * chunk_size)) for i in range(ns)]
    visited_refs = []

    with root.lock:
        node = root
        for seq, seq_len in zip(np.split(sequence, ns), seq_actual_lens):
            if len(node.children) == 0:  # cache ran out of node
                return (total_match, ref_vals), visited_refs
            left_match, right_match = TrieNode._overlap(node.child_keys, seq, seq_len, pad_idx=pad_idx)
            exact_match = np.minimum(left_match, right_match)
            best_idx = np.argmax(exact_match)
            match_length = exact_match[best_idx]
            if match_length > 0:
                visited_refs.append(int(node.children[best_idx].value))
            if match_length == 0:
                break
            node = node.children[best_idx]
            total_match += int(match_length)
            ref_vals.append(node.value)
            if match_length != seq_len:
                break
    return (total_match, ref_vals), visited_refs

def remove_prefix_nodes(node: TrieNode, refs_to_delete: Sequence[int]):
    refs_to_delete, deleted_refs = set(refs_to_delete), set()
    ctx = node.lock if node.lock is not None else contextlib.nullcontext()
    with ctx:
        for child in node.children:
            deleted_refs |= remove_prefix_nodes(child, refs_to_delete)
            deleted_refs |= set(child.value for child in node.children if child.value in refs_to_delete)
            delete_idxs = set([i for i, child in enumerate(node.children) if child.value in refs_to_delete])
            for idx in delete_idxs:  # if we're removing a full child, tell it to remove all its children first
                deleted_refs |= remove_prefix_nodes(node.children[idx], [c.value for c in node.children[idx].children])
            node.child_keys = TrieNode._delete_keys(node.child_keys, delete_idxs)
            node.children = [child for i, child in enumerate(node.children) if i not in delete_idxs]
    return set(deleted_refs)

########################################################################################################################
# serving loop #########################################################################################################
########################################################################################################################

next_power_of_2 = lambda x: 2 ** round(math.ceil(math.log2(x)))
like_spec = lambda z: jax.tree.map(lambda x: jax.typeof(x).sharding.spec, z)
like_shard = lambda z, mesh: jax.tree.map(lambda x: NamedSharding(mesh, jax.typeof(x).sharding.spec), z)
_make_empty = lambda x, mesh: jax.make_array_from_single_device_arrays(
    x.shape, NamedSharding(mesh, jax.typeof(x).sharding.spec), [], dtype=x.dtype
)


@dataclasses.dataclass
class ServingConfig:
    decode_steps: int = 10
    decode_batch_size: int = 16
    prefill_batch_size: int = 4
    prefix_chunk_size: int = 512
    eos_tokens: tuple[int, ...] | jax.Array = ()
    token_pad_idx: int = 0
    max_decode_length: int = 64
    max_ondevice_buffers: int = 100
    max_buffers: int = 256


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
class PrefillJob:
    request: UserRequestPrompt
    cache_entry: Any
    match_len: int


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
    TIMEOUT_SEC = 600

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

def maybe_call(fn: Callable, mesh: Mesh):
    """Only call the program if the host worker is participating, get (truly) empty arrys with correct sharding."""
    mesh_devices = set(d.id for d in mesh.devices.flat)
    if any(d.id in mesh_devices for d in jax.local_devices()):  # host has some participating devices
        return fn
    return (lambda *args, **kw: jax.tree.map(partial(_make_empty, mesh=mesh), jax.eval_shape(fn, *args, **kw)))


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


class ServingLoop:
    def __init__(
        self,
        serve_cfg: ServingConfig,
        cfg: Config,
        forward_fn: Callable,
        prefill_weights: Weights,
        prefill_cache: KVCache,
        decode_weights: Weights,
        decode_cache: KVCache,
        is_server: bool = False,
    ):
        #self.init_cache = init_cache
        self.prefill_cache = prefill_cache
        if not SyncServer.broadcast("welcome", 0, is_server, is_server):
            raise ValueError("Neither this proccess nor any other processe is the main server, at least one must.")
        self.serve_cfg, self.cfg = serve_cfg, cfg

        # setup decode
        self.forward_fn, self.decode_weights = forward_fn, decode_weights
        self.decode_mesh = [x for x in jax.tree.leaves(decode_weights) if hasattr(x, "sharding")][0].sharding.mesh
        with set_mesh(self.decode_mesh):
            self.decode_work = DecodeWork(None, decode_cache, [None for _ in range(serve_cfg.decode_batch_size)])
            self.decode_work.curr_tokens = jax.device_put(
                jnp.zeros((serve_cfg.decode_batch_size, 1), dtype=jnp.int32), P()
            )
        self.multistep_decode_fn = _make_multistep_decode_fn(self.forward_fn)
        self._update_index = jax.jit(lambda x, i, new: x.at[i, ...].set(new[:, None], mode="drop"))

        def _update_cache_and_index(cache: KVCache, curr_tokens: jax.Array, new_tokens, kvs, batch_idxs, actual_lens):
            # sort to minimize variants num
            length_sort = sorted(range(len(kvs)), key=lambda i: jax.tree.leaves(kvs[i])[0].shape[-2])
            sorted_args = [[x[i] for i in length_sort] for x in (kvs, batch_idxs, actual_lens)]
            new_cache = decode_cache.insert_sequences(cache, *sorted_args)
            with set_mesh(self.decode_mesh):
                new_curr_tokens = self._update_index(curr_tokens, np.array(batch_idxs), np.array(new_tokens))
            return new_cache, new_curr_tokens

        self._update_cache_and_index = _update_cache_and_index
        self.decode_output = (None, None)

        # setup prefill
        self.prefill_weights = prefill_weights
        self.prefill_mesh = [x for x in jax.tree.leaves(prefill_weights) if hasattr(x, "sharding")][0].sharding.mesh
        self.prefill_work = PrefillWork([], [], [])
        self._get_index = jax.jit(lambda z, idx: jax.tree.map(lambda x: x[:, idx, ...], z))
        self._get_cache_entry = jax.jit(self.decode_work.cache.get_sequence)

        # setup misc
        self.pending_requests, self.state_lock, self.results = [], threading.Lock(), {}
        self.pad_id, self.eos_tokens, self.time_axis = 0, np.array(serve_cfg.eos_tokens), TIME_AXIS
        self._background = ThreadPoolExecutor(max_workers=1024)

        # setup profiling
        self.profile_start_time, self.profiling = -1, False

        # setup cache management
        self.prefix_cache, self._retrieve_prefix, self._insert_prefix = None, None, None
        self.new_prefix_cache()

        # setup the sync server for multi-host
        self._it, self.roles = 0, (("server",) if is_server else ())  # main server
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
                with set_mesh(self.decode_mesh):
                    result.cache_entry = result.cache_entry() if callable(result.cache_entry) else result.cache_entry
                self.results[result.id] = self.decode_work.active_results[i]
                batch_cache_updates.append((result.cache_entry, i, result.len, result.next_token))
                if len(self.prefill_work.to_decode) == 0:
                    break
            if "decode" in self.roles and len(batch_cache_updates) > 0:  # batch cache update
                entries, batch_idxs, lens, next_tokens = map(list, zip(*batch_cache_updates))
                entries = [entry.result() if hasattr(entry, "result") else entry for entry in entries]  # maybe collect
                self.decode_work.cache, self.decode_work.curr_tokens = self._update_cache_and_index(
                    self.decode_work.cache, self.decode_work.curr_tokens, next_tokens, entries, batch_idxs, lens
                )

        if all(x is None for x in self.decode_work.active_results):
            return  # skip decoding if no decoding tasks are present

        # 2. run N decode steps
        output_tokens, output_mapping = [], []
        with set_mesh(self.decode_mesh):
            # config = dict(cfg=self.cfg, steps=self.serve_cfg.decode_steps, participate="decode" in self.roles)
            config = dict(cfg=self.cfg, steps=self.serve_cfg.decode_steps)
            decode_fn = maybe_call(self.multistep_decode_fn, self.decode_mesh)
            #decode_fn = self.multistep_decode_fn
            (self.decode_work.curr_tokens, self.decode_work.cache), output_tokens = decode_fn(
                self.decode_work.curr_tokens, self.decode_weights, self.decode_work.cache, **config
            )
            output_mapping = [
                [getattr(result, "id", -1) for result in self.decode_work.active_results]
            ] * self.serve_cfg.decode_steps
            output_mapping = np.array(output_mapping).T
        print(f"Decoding with fill rate: {np.mean([result is not None for result in self.decode_work.active_results])}")

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
                output_tokens_flat = output_tokens.reshape(-1).tolist()
                output_mapping_flat = output_mapping.reshape(-1).tolist()
            else:
                output_tokens, done, output_tokens_flat, output_mapping_flat = None, None, None, None
            output_tokens_flat, output_mapping_flat, done = SyncServer.broadcast(
                "decode_output",
                self._it,
                (output_tokens_flat, output_mapping_flat, done),
                is_source="decode_coordinator" in self.roles,
            )
            for token, id in zip(output_tokens_flat, output_mapping_flat):
                if id > 0:
                    self.results[id].token_list.append(token)
                    self.results[id].tokens_decoded += 1
            with set_mesh(self.decode_mesh):
                for i, result in enumerate(self.decode_work.active_results):
                    if result is None:
                        continue
                    # 2. check for done sequences; evict them if done and return them
                    if done[i]:
                        if USE_PREFIX_CACHE:  # store the results in the prefix cache buffer store
                            sequence = np.array(result.token_list)
                            cache_entry, _ = self._get_cache_entry(self.decode_work.cache, i)
                            ns = math.ceil(sequence.size / self.serve_cfg.prefix_chunk_size)
                            buffer_ids = BUFFER_STORE._get_unique_buffer_ids(ns)
                            visited_ids, store_ids, del_ids = self._insert_prefix(sequence, buffer_ids)
                            if len(store_ids) > 0:
                                axis = self.time_axis - 1 + 1  # batch missing (-1) layers concatenated (+1)
                                chunked_cache_entry = _split(cache_entry, axis, self.serve_cfg.prefix_chunk_size, ns)
                                vals = [chunked_cache_entry[buffer_ids.index(id)] for id in store_ids]
                                BUFFER_STORE.store(store_ids, vals)
                            BUFFER_STORE.delete(del_ids)
                            BUFFER_STORE.mark_visited(visited_ids)
                        return_request(result)
                        result.done, self.decode_work.active_results[i] = True, None

    def prefill_step(self):
        # 1. prefill requests to be prefilled (do this first to overlap with decode)
        prefill_input: list[PrefillJob] = self.prefill_work.to_prefill[: self.serve_cfg.prefill_batch_size]
        self.prefill_work.to_prefill = self.prefill_work.to_prefill[len(prefill_input) :]
        if len(prefill_input) > 0:
            prefill_texts = [job.request.text[job.match_len :] for job in prefill_input]
            max_len = max([len(text) for text in prefill_texts])
            inputs = [text + [self.pad_id] * (max_len - len(text)) for text in prefill_texts]
            inputs = np.stack([np.array(input) for input in inputs], 0)
            row_pad = self.serve_cfg.prefill_batch_size - inputs.shape[0]
            col_pad = max(next_power_of_2(inputs.shape[-1]), 64) - inputs.shape[-1]
            inputs = np.pad(inputs, ((0, row_pad), (0, col_pad)), mode="constant", constant_values=self.pad_id)

            with set_mesh(self.prefill_mesh):
                actual_cache_len = np.array(max(job.match_len for job in prefill_input), dtype=np.int32)
                self.prefill_cache.iter = actual_cache_len  # TODO: make this explictly cache public interface
                kvs = [job.cache_entry() if job.cache_entry is not None else None for job in prefill_input]
                batch_idxs = np.array([i for i, kv in enumerate(kvs) if kv is not None])
                actual_lens = np.array([job.match_len for kv, job in zip(kvs, prefill_input) if kv is not None])
                kvs = [kv for kv in kvs if kv is not None]

                if len(kvs) > 0:
                    # sort to minimize variants num
                    length_sort = sorted(range(len(kvs)), key=lambda i: jax.tree.leaves(kvs[i])[0].shape[-2])
                    sorted_args = [[x[i] for i in length_sort] for x in (kvs, batch_idxs, actual_lens)]
                    insert_sequences = maybe_call(self.prefill_cache.insert_sequences, self.prefill_mesh)
                    self.prefill_cache = insert_sequences(self.prefill_cache, *sorted_args)
                cfg = dataclasses.replace(self.cfg, mesh=self.prefill_mesh)
                forward_fn = maybe_call(self.forward_fn, self.prefill_mesh)
                _, self.prefill_cache = forward_fn(inputs, self.prefill_weights, self.prefill_cache, cfg)

            with set_mesh(self.prefill_mesh):
                for i, job in enumerate(prefill_input):
                    request = job.request
                    cache_entry, _ = maybe_call(self._get_cache_entry, self.prefill_mesh)(self.prefill_cache, i)
                    cache_entry = _ensure_all_args_on_mesh(cache_entry, self.decode_mesh)
                    sequence = np.array(request.text)
                    new_decode = PrefillResult(
                        request.id, sequence, request.text[-1], cache_entry, len(request.text) - 1
                    )
                    self.prefill_work.to_decode.append(new_decode)

        # 2. triage requests based on whether they need to go to prefill or there's a cache match, so decode directly
        while len(self.prefill_work.requests) > 0:
            request = self.prefill_work.requests.pop(0)
            sequence = np.array(request.text)
            (total_match, buffer_ids), visited_ids = self._retrieve_prefix(sequence)
            assert total_match <= sequence.size
            BUFFER_STORE.mark_visited(visited_ids)
            _axis = self.time_axis - 1 + 1  # batch missing (-1) layers concatenated (+1)
            buffers = BUFFER_STORE.load(buffer_ids)
            if total_match == sequence.size:
                cache_entry = partial(_concat, _ensure_all_args_on_mesh(buffers, mesh=self.decode_mesh), _axis)
                new_decode = PrefillResult(request.id, sequence, request.text[-1], cache_entry, len(request.text) - 1)
                self.prefill_work.to_decode.append(new_decode)
                print(f"Found a full match")
            else:
                print(f"Need to prefill, only found a match for length {total_match / (len(request.text) - 1):.2%}")
                print(f"That equals {len(buffer_ids)} buffers or {total_match=}")
                if total_match > 0:
                    cache_entry = partial(_concat, _ensure_all_args_on_mesh(buffers, mesh=self.prefill_mesh), _axis)
                else:
                    cache_entry = None
                self.prefill_work.to_prefill.append(PrefillJob(request, cache_entry, total_match))

    def serving_step(self):
        # this event loop relies on determinism for issuing computation to multiple processes (multi-process JAX)
        # frequent barriers should keep it in sync

        # potentially profile when received the request to #########################################
        is_server = "server" in self.roles
        should_start_profile = self.profile_start_time > 0 and not self.profiling
        should_start_profile = SyncServer.broadcast("profile", self._it, should_start_profile, is_source=is_server)
        if should_start_profile:
            self.profile_start_time, self.profiling = time.perf_counter(), True
            jax.profiler.start_trace("/tmp/online")
            print("STARTING TRACE")
        should_stop_profile = self.profile_start_time > 0 and time.perf_counter() - self.profile_start_time > 5.0
        should_stop_profile = SyncServer.broadcast("stop_profile", self._it, should_stop_profile, is_source=is_server)
        if should_stop_profile:
            self.profile_start_time, self.profiling = -1, False
            print("STOPPING TRACE")
            jax.profiler.stop_trace()
        # potentially profile when received the request to #########################################

        # sync on the server requests received #####################################################
        SyncServer.barrier("serving_step", self._it)
        self._it, requests = self._it + 1, None
        if "server" in self.roles:
            with self.state_lock:
                self.pending_requests, requests = [], list(self.pending_requests)
        serve_cfg, requests = SyncServer.broadcast(
            "requests", self._it, (dataclasses.asdict(self.serve_cfg), requests), is_source="server" in self.roles
        )
        with self.state_lock:
            self.serve_cfg = dataclasses.replace(self.serve_cfg, **serve_cfg)
        for request in requests:
            self.total_requests += 1
            self.prefill_work.requests.append(UserRequestPrompt(**request))
        # sync on the server requests received #####################################################

        # main event loop work #####################################################################
        self.decode_step()
        self.prefill_step()
        # main event loop work #####################################################################

        # offload buffers to keep a max of N #######################################################
        BUFFER_STORE.offload_buffers(max(0, BUFFER_STORE.livecount - self.serve_cfg.max_ondevice_buffers))
        extra_buffer_count = max(len(BUFFER_STORE.usecount) - self.serve_cfg.max_buffers, 0)
        if extra_buffer_count > 0:
            refs_to_delete = sorted(BUFFER_STORE.usecount.keys())[:extra_buffer_count]
            deleted_buffers = remove_prefix_nodes(self.prefix_cache, refs_to_delete)
            BUFFER_STORE.delete(list(deleted_buffers))
            if len(BUFFER_STORE._store) > self.serve_cfg.max_buffers:
                raise ValueError()
        # offload buffers to keep a max of N #######################################################

    def add_request(self, request: UserRequestPrompt):
        with self.state_lock:
            self.pending_requests.append(dataclasses.asdict(request))

    def update_params(self, params: dict[str, Any]):
        with self.state_lock:
            self.serve_cfg = dataclasses.replace(self.serve_cfg, **params)

    def new_prefix_cache(self):
        self.prefix_cache = TrieNode(None, lock=threading.Lock())
        self._retrieve_prefix = partial(retrieve_prefix, self.prefix_cache, chunk_size=self.serve_cfg.prefix_chunk_size)
        self._insert_prefix = partial(insert_prefix, self.prefix_cache, chunk_size=self.serve_cfg.prefix_chunk_size)
