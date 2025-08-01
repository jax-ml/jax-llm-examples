"""This file implements cross-host device_put in multi-process JAX - a temporary workaround until jax.device_put update."""

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

#jax.config.update("jax_enable_empty_arrays", True)
PyTree = Any


@lru_cache
def combine_meshes(meshA, meshB):
    if not (meshA.devices.shape == meshB.devices.shape and meshA.axis_names == meshB.axis_names):
        raise ValueError("Meshes shapes and specs must match")
    devices = np.stack([meshA.devices, meshB.devices], axis=0)
    axis_names = ("cross_mesh",) + tuple(meshA.axis_names)
    axis_types = tuple(meshA.axis_types)[:1] + tuple(meshA.axis_types)
    return Mesh(devices, axis_names, axis_types=axis_types)


@jax.jit
def _prepare_arrays(xs: list[jax.Array]):
    return jax.tree.map(lambda x: x[None, ...], xs)


@lru_cache
def _make_zeros(sds, shardings):
    new_shardings = tuple(NamedSharding(sd.mesh, P(None, *sd.spec)) for sd in shardings)
    new_sds = tuple(jax.ShapeDtypeStruct((1,) + sd.shape, sd.dtype) for sd in sds)
    return jax.jit(
        lambda: jax.tree.map(lambda s: jnp.zeros(s.shape, dtype=s.dtype), new_sds), out_shardings=new_shardings
    )()


@jax.jit
def _combine(xs: list[jax.Array]):
    return jax.tree.map(lambda x: jnp.sum(x, axis=0).astype(x.dtype), xs)


def transfer_tree_A2B(xs: PyTree, meshA, meshB):
    if meshA == meshB:
        return xs
    meshC = combine_meshes(meshA, meshB)
    xs, xs_struct = jax.tree.flatten(xs)
    combined_sharding = [NamedSharding(meshC, P("cross_mesh", *x.sharding.spec)) for x in xs]
    dest_sharding = [NamedSharding(meshB, x.sharding.spec) for x in xs]
    dest_arrays = _make_zeros(tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in xs), tuple(dest_sharding))
    all_arrays = [x_src._arrays + x_dest._arrays for x_src, x_dest in zip(_prepare_arrays(xs), dest_arrays)]
    xs_combined = [
        jax.make_array_from_single_device_arrays((2,) + x.shape, sharding, arrays, dtype=x.dtype)
        for (x, arrays, sharding) in zip(xs, all_arrays, combined_sharding)
    ]
    xs_repl = _combine(xs_combined)  # issue collectives under jit
    xs_new = [
        jax.make_array_from_single_device_arrays(
            x_src.shape, sharding, x_new._arrays[len(x_src._arrays) :], dtype=x_src.dtype
        )
        for x_new, x_src, sharding, x_dest in zip(xs_repl, xs, dest_sharding, dest_arrays)
    ]
    return jax.tree.unflatten(xs_struct, xs_new)
