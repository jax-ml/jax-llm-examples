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
import math
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import auto_axes

QuantArray, PyTree, KVCache, PagedKVCache = Any, Any, Any, Any


@dataclasses.dataclass
class AttentionInterface:
    cache: KVCache
    get_sequence: Callable
    insert_sequences: Callable


next_power_of_2 = lambda x: 2 ** math.ceil(math.log2(max(x, 1)))
_pad_after = lambda x, l, axis: jnp.pad(x, [(0, 0) if i != axis else (0, l - x.shape[i]) for i in range(x.ndim)])


def _transpose_attention_tree(kv_list: list[PyTree], time_axis: int):
    "From a list of cache entries stacked along layer idx (in transit) to stacked along batch, layers split into list."

    _split = lambda x: jnp.split(x, x.shape[0], axis=0)
    max_seq_len = max([jax.tree.leaves(kv)[0].shape[time_axis] for kv in kv_list])
    kv_list = [jax.tree.map(lambda x: _pad_after(x, max_seq_len, time_axis), kv) for kv in kv_list]
    out = [None for _ in kv_list[0]]
    for i, c in enumerate(kv_list[0]):
        els = [[_split(z) for z in jax.tree.leaves(kv[i])] for kv in kv_list]  # [B, R_flat, L]
        els = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *els)  # [R_flat, L]
        leaves_list = list(zip(*els, strict=True))  # [L, R_flat]
        out[i] = [jax.tree.unflatten(jax.tree.structure(c), leaves) for leaves in leaves_list]  # [L, R]
    return tuple(out), max_seq_len


########################################################################################################################
# KV cache utils #######################################################################################################
########################################################################################################################


@partial(jax.jit, donate_argnames=("cache",))
def __kvcache_insert_sequences(
    cache: KVCache,
    kvs: list[tuple[jax.Array | QuantArray, ...]],
    batch_idxs: list[jax.Array],
    actual_lens: list[jax.Array],
    update_mask: list[bool] | None = None,
    erase: bool = False,
):
    assert len(kvs) == len(batch_idxs) == len(actual_lens)
    batch_idxs, actual_lens, update_mask = jnp.array(batch_idxs), jnp.array(actual_lens), jnp.array(update_mask)
    uninitialized_cache = jnp.logical_or(cache.iter < 0, erase)
    start_time = jnp.where(
        uninitialized_cache, jnp.max(actual_lens) - actual_lens, (cache.iter - actual_lens) % cache.size
    )
    batch_idxs = jnp.where(update_mask, batch_idxs, 2**30)  # send masked to nowhere
    kvs, max_seq_len = _transpose_attention_tree(kvs, time_axis=cache.time_axis)
    time_indices = (jnp.arange(max_seq_len)[None, :] + start_time[:, None]) % cache.size

    def _update_element(x, u):
        update_permute = [0, cache.time_axis] + [i for i in range(u.ndim) if i not in (0, cache.time_axis)]
        return x.at[batch_idxs[:, None], :, time_indices, ...].set(u.transpose(update_permute), mode="drop")

    buffers_sharding = jax.tree.map(lambda x: jax.typeof(x).sharding, cache.buffers)
    cache_kvs = jax.tree.map(_update_element, cache.buffers, kvs)
    cache_starts = cache.starts.at[batch_idxs].set(start_time, mode="drop")
    cache_iter = jnp.where(uninitialized_cache, jnp.max(actual_lens), cache.iter)

    buffer_names = [field.name for field in dataclasses.fields(cache)][: len(cache_kvs)]
    return dataclasses.replace(
        cache, **dict(zip(buffer_names, cache_kvs, strict=True)), iter=cache_iter, starts=cache_starts
    )


@partial(jax.jit, donate_argnames=("cache",))
def _kvcache_insert_sequences(
    cache: KVCache,
    kvs: list[tuple[jax.Array | QuantArray, ...]],
    batch_idxs: list[jax.Array],
    actual_lens: list[jax.Array],
    update_mask: list[bool] | None = None,
    erase: bool = False,
):
    cache_shardings = jax.tree.map(lambda x: jax.typeof(x).sharding, cache)
    return auto_axes(__kvcache_insert_sequences, out_sharding=cache_shardings)(
        cache, kvs, batch_idxs, actual_lens, update_mask, erase
    )


@partial(jax.jit, donate_argnames=("cache",))
def _maybe_erase_only(cache: KVCache, erase: bool):
    return dataclasses.replace(cache, iter=jnp.where(erase, -1, cache.iter))


def kvcache_insert_sequences(
    cache: KVCache,
    kvs: list[tuple[jax.Array | QuantArray, ...]],
    batch_idxs: list[jax.Array],
    actual_lens: list[jax.Array],
    erase: bool = False,
):
    if len(kvs) == 0:
        return _maybe_erase_only(cache, erase)
    pad_len = max(next_power_of_2(len(kvs)), 4) - len(kvs)  # an update of power of 2 and at least 4
    update_mask = [i < len(kvs) for i in range(len(kvs) + pad_len)]
    kvs = kvs + [kvs[-1]] * pad_len
    batch_idxs, actual_lens = batch_idxs + [batch_idxs[-1]] * pad_len, actual_lens + [actual_lens[-1]] * pad_len
    return _kvcache_insert_sequences(cache, kvs, batch_idxs, actual_lens, update_mask, erase=erase)


@jax.jit
def kvcache_get_sequence(cache: KVCache, batch_idx: jax.Array):
    shift = -cache.starts[batch_idx]
    assert cache.time_axis > 0
    kvs = jax.tree.map(lambda x: jnp.roll(x[batch_idx, ...], shift=shift, axis=cache.time_axis - 1), cache.buffers)
    kvs = tuple(jax.tree.map(lambda *xs: jnp.stack(xs, 0), *z) for z in kvs)
    true_len = cache.fill_len()[batch_idx]
    return kvs, true_len


########################################################################################################################
# Paged KV cache utils #################################################################################################
########################################################################################################################

PagedKVCache = Any


def _find_empty_pages(free_pages: jax.Array, k: int, proposal_pages: jax.Array | None = None):
    if proposal_pages is not None:
        assert proposal_pages.size == k
        proposal_mask = free_pages[proposal_pages]
        indicies = jnp.where(~proposal_mask, jnp.cumsum(~proposal_mask, axis=-1) - 1, k - 1)
        newly_free_pages = free_pages.at[jnp.where(proposal_mask, proposal_pages, 2**30)].set(False, mode="drop")
        return jnp.where(proposal_mask, proposal_pages, jax.lax.top_k(newly_free_pages, k)[1][indicies])
    else:
        return jax.lax.top_k(free_pages, k)[1]


@partial(jax.jit, donate_argnames=("cache",))
def __paged_kvcache_insert_sequences(
    cache: PagedKVCache,
    kvs: list[tuple[jax.Array | QuantArray, ...]],
    batch_idxs: list[jax.Array],
    actual_lens: list[jax.Array],
    update_mask: list[bool] | None = None,
) -> PagedKVCache:
    update_mask = jnp.array(update_mask)
    batch_idxs = jnp.where(update_mask, jnp.array(batch_idxs), 2**30)  # send masked to nowhere
    actual_lens = jnp.minimum(jnp.array(actual_lens), jnp.array([jax.tree.leaves(kv)[0].shape[2] for kv in kvs]))

    kvs, max_seq_len = _transpose_attention_tree(kvs, time_axis=2)  # undo stack along layer dimension in transit

    # clear existing pages
    actual_page_num = jnp.rint(jnp.ceil(cache.lengths[batch_idxs] / cache.page_size)).astype(jnp.int32)
    occupied_mask = jnp.arange(cache.block_tables.shape[-1])[None, :] < actual_page_num[:, None]
    indices_to_free = jnp.where(occupied_mask & update_mask[:, None], cache.block_tables[batch_idxs, :], 2**30)
    new_free_pages = cache.free_pages.at[indices_to_free.reshape(-1)].set(True, mode="drop")

    # get the length of the new sequence and find empty pages for the new sequence ideally contiguous
    upper_bound_page_num = math.ceil(max_seq_len / cache.page_size)
    actual_page_num = jnp.rint(jnp.ceil(actual_lens / cache.page_size)).astype(jnp.int32)
    avg_pages_per_batch_entry = round(jax.tree.leaves(cache)[0].shape[1] / cache.batch_size)
    proposal_pages = batch_idxs[:, None] * avg_pages_per_batch_entry + jnp.arange(upper_bound_page_num)[None, :]
    pages_idx = _find_empty_pages(
        new_free_pages, upper_bound_page_num * batch_idxs.size, proposal_pages=proposal_pages.reshape(-1)
    ).reshape(proposal_pages.shape)
    pages_arange = jnp.arange(upper_bound_page_num)
    pages_idx = jnp.where(update_mask[:, None] & (pages_arange[None, :] < actual_page_num[:, None]), pages_idx, 2**30)

    # reshape the new pages for insertion and possibly quantize
    b, h, s, e = jax.tree.leaves(kvs)[0].shape
    kvs = jax.tree.map(lambda x: x.reshape((b, h, s // cache.page_size, cache.page_size) + x.shape[3:]), kvs)

    def _update_element(x, u):
        # we're updating (batch, page_entries) with (BATCH, heads, PAGE, page_size, head_dim), so (BATCH, PAGE) go first
        update_permute = [1, 0, 2] + [i for i in range(u.ndim) if i not in (0, 1, 2)]
        return x.at[:, pages_idx, ...].set(u.transpose(update_permute), mode="drop")

    new_buffers = jax.tree.map(_update_element, cache.buffers, kvs)
    block_tables_idx = jnp.where(
        update_mask[:, None] & (pages_arange[None, :] < actual_page_num[:, None]), pages_arange[None, :], 2**30
    )
    new_block_tables = cache.block_tables.at[batch_idxs[:, None], block_tables_idx].set(pages_idx, mode="drop")
    new_free_pages = new_free_pages.at[pages_idx.reshape(-1)].set(False, mode="drop")
    new_lengths = cache.lengths.at[batch_idxs].set(actual_lens, mode="drop")

    named_buffers = dict(zip([field.name for field in dataclasses.fields(cache)][: len(new_buffers)], new_buffers))
    return dataclasses.replace(
        cache, **named_buffers, lengths=new_lengths, block_tables=new_block_tables, free_pages=new_free_pages
    )


@partial(jax.jit, donate_argnames=("cache",))
def _paged_kvcache_insert_sequences(
    cache: PagedKVCache,
    kvs: list[tuple[jax.Array | QuantArray, ...]],
    batch_idxs: list[jax.Array],
    actual_lens: list[jax.Array],
    update_mask: list[bool] | None = None,
) -> PagedKVCache:
    cache_sharding = jax.tree.map(lambda x: jax.typeof(x).sharding, cache)
    return auto_axes(__paged_kvcache_insert_sequences, out_sharding=cache_sharding)(
        cache, kvs, batch_idxs, actual_lens, update_mask
    )


def paged_kvcache_insert_sequences(
    cache: KVCache,
    kvs: list[tuple[jax.Array | QuantArray, ...]],
    batch_idxs: list[jax.Array],
    actual_lens: list[jax.Array],
    erase: bool = False,
):
    del erase  # inapplicable
    if len(kvs) == 0:
        return cache
    pad_len = max(next_power_of_2(len(kvs)), 4) - len(kvs)  # an update of power of 2 and at least 4
    update_mask = [i < len(kvs) for i in range(len(kvs) + pad_len)]
    kvs = kvs + [kvs[-1]] * pad_len
    batch_idxs, actual_lens = batch_idxs + [batch_idxs[-1]] * pad_len, actual_lens + [actual_lens[-1]] * pad_len
    return _paged_kvcache_insert_sequences(cache, kvs, batch_idxs, actual_lens, update_mask)


@partial(jax.jit, static_argnames=("max_seq_len",))
def paged_kvcache_get_sequence(cache: PagedKVCache, batch_idx: jax.Array, max_seq_len: int = -1):
    true_len = cache.fill_len()[batch_idx]
    max_seq_len = max_seq_len if max_seq_len > 0 else cache.page_size * cache.block_tables.shape[-1]
    max_seq_len = min(max_seq_len, cache.page_size * cache.block_tables.shape[-1])  # cache capacity
    page_indices = cache.block_tables[batch_idx, : round(math.ceil(max_seq_len / cache.page_size))]
    _reshape_out = lambda x: x.reshape((x.shape[0], max_seq_len) + x.shape[3:])
    mask = jnp.arange(max_seq_len) < true_len
    _get = lambda x: jnp.where(mask[None, :, *([None] * (x.ndim - 3))], _reshape_out(x[:, page_indices, ...]), 0)

    # stack along layer dimensions for transit
    kvs = tuple(jax.tree.map(lambda *xs: jnp.stack(xs, 0), *z) for z in jax.tree.map(_get, cache.buffers))
    return kvs, true_len
