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

# pylint: disable=bad-continuation
# pylint: disable=missing-module-docstring
# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=g-importing-member
# pylint: disable=unused-imports
# pylint: disable=line-too-long
# pylint: disable=g-docstring-first-line-too-long
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=g-bare-generic
# pylint: disable=g-short-doctring-punctuation

"""Minimal model definition."""

from collections import OrderedDict as odict
import dataclasses
from functools import lru_cache
from functools import partial
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
from jax import random
from jax.experimental.array_serialization import pytree_serialization as ser
from jax.experimental.layout import Format, Layout
import jax.numpy as jnp
from jax.sharding import auto_axes, reshard, PartitionSpec as P

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
except ImportError:
    splash = None, None

PAD_ID = 0

AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]
PathT = str | os.PathLike[str] | Path

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "z"
TENSOR_ONLY_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y", "z")


@dataclasses.dataclass(unsafe_hash=True)
class ShardingRules:
    """Mapping from logical data axes to physical mesh axes.

    To manage the different shardings in the model, we define the "logical"
    dimensions of various arrays (each dimension for each layer's weights,
    etc.). Each of these logical axes may then be sharded over a physical mesh
    axis, i.e. over multiple devices. For example, any values with a batch
    dimension should always be sharded over the batch axis of the mesh.

    Defining the shardings this way allows us to easily try out new sharding
    strategies by just changing this mapping. The rest of the code handles
    taking this mapping and eventually turning it into the correct JAX shardings
    and sharding contraints.
    """

    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None
    head_dim: AxisName = None
    # attention
    qkv_embed: AxisName = None
    q_heads: AxisName = ATTN_HEADS_AXIS_NAME
    kv_heads: AxisName = ATTN_HEADS_AXIS_NAME
    o_heads: AxisName = ATTN_HEADS_AXIS_NAME
    o_embed: AxisName = None
    # MLP layer
    mlp_up_embed: AxisName = None
    mlp_up_ffw: AxisName = TENSOR_AXIS_NAME
    mlp_down_ffw: AxisName = TENSOR_AXIS_NAME
    mlp_down_embed: AxisName = None
    # MoE layer
    moe_e_experts: AxisName = EXPERT_AXIS_NAME
    moe_e_up_embed: AxisName = None
    moe_e_up_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_e_down_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_e_down_embed: AxisName = None
    moe_e_tp: AxisName = TENSOR_ONLY_AXIS_NAME  # moe forward function tensor parallelism
    moe_e_ep: AxisName = EXPERT_AXIS_NAME  # moe forward function expert parallelism
    # vocab
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME


def logical_to_physical(logical: Axes, rules: ShardingRules) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    spec = jax.tree.map(lambda x: getattr(rules, x) if x is not None else None, logical)
    # `spec` may contain tuples, flatten to check that `spec` maps each physical mesh axis to at most one logical axis.
    flat_axes = jax.tree.leaves(spec)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(f"Colliding physical axes from translating logical spec {logical} -> {spec}")
    return P(*spec)


def logical_to_sharding(logical: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules) -> jax.sharding.Sharding:
    """Returns the sharding for a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    assert mesh is not None
    return jax.sharding.NamedSharding(mesh, logical_to_physical(logical, rules))


static_field = lambda val=dataclasses.MISSING: dataclasses.field(default=val, metadata=dict(static=True))


@jax.tree_util.register_static
@dataclasses.dataclass
class Config:
    embed: int
    q_heads: int
    local_kv_heads: int
    local_head_dim: int
    global_kv_heads: int | None
    global_head_dim: int | None
    global_k_eq_v: bool
    num_layers: int
    vocab_size: int
    max_seq_len: int
    causal: bool
    # moe
    moe_ffw_size: int
    moe_experts_per_tok: int
    moe_num_experts: int
    norm_eps: float = 1e-6
    # mlp
    mlp_ffw_size: int = -1
    per_layer_input_dim: int = 0
    vocab_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    # attention
    attention_types: tuple[str, ...] = ()
    sliding_window_size: int | None = None
    final_logit_softcap: float | None = None
    tie_embed: bool = False
    # compute strategy
    moe_gate_dtype: jax.typing.DTypeLike = jnp.float32
    ep_strategy: str = "decode"
    use_prefill_attn_kernel: bool = False
    use_decode_attn_kernel: bool = False
    use_ragged_dot_kernel: bool = False
    weight_dtype: jax.typing.DTypeLike = jnp.bfloat16
    dtype: jax.typing.DTypeLike = jnp.bfloat16  # for compute
    # sharding
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    mesh: jax.sharding.Mesh | None = None
    # quantization
    quant_moe: bool = False
    quant_mlp: bool = False
    quant_attn: bool = False
    quant_cache: bool = False
    quant_scale_dtype: jax.typing.DTypeLike = jnp.bfloat16
    # positional embedding parameters
    rope_theta: float = 500000.0
    global_rope_proportion: float | None = None
    local_rope_proportion: float | None = None
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000
    # sampling
    sample_topk: int = 4
    sample_temp: float = 0.7


def hf_to_jax_config(hf_config: Any | dict[str, Any]) -> "Config":
    hf_config = hf_config if "text_config" not in hf_config else hf_config
    _get = lambda x, k, default=None: (
        getattr(x, k, default) if not isinstance(hf_config, dict) else hf_config.get(k, default)
    )
    rope_params = _get(hf_config, "rope_parameters", None) or {}
    local_rope = rope_params.get("sliding_attention", {}) if isinstance(rope_params, dict) else {}
    global_rope = rope_params.get("full_attention", {}) if isinstance(rope_params, dict) else {}
    layer_types = _get(hf_config, "layer_types", None) or ()
    attention_types = tuple(
        {"sliding_attention": "local_attention", "full_attention": "global_attention"}.get(t, t) for t in layer_types
    )
    local_theta = local_rope.get("rope_theta", _get(hf_config, "rope_theta", 10_000))
    global_theta = global_rope.get("rope_theta", 1_000_000)
    intermediate_size = _get(hf_config, "intermediate_size", -1)
    use_double_wide_mlp = _get(hf_config, "use_double_wide_mlp", False)
    global_kv_heads = _get(hf_config, "num_global_key_value_heads")
    if global_kv_heads is None:
        global_kv_heads = _get(hf_config, "num_key_value_heads")
    return Config(
        embed=_get(hf_config, "hidden_size"),
        mlp_ffw_size=intermediate_size,
        moe_ffw_size=_get(hf_config, "moe_intermediate_size", _get(hf_config, "expert_intermediate_size", -1)) or -1,
        q_heads=_get(hf_config, "num_attention_heads"),
        local_kv_heads=_get(hf_config, "num_key_value_heads"),
        num_layers=_get(hf_config, "num_hidden_layers"),
        local_head_dim=_get(hf_config, "head_dim"),
        vocab_size=_get(hf_config, "vocab_size"),
        tie_embed=_get(hf_config, "tie_word_embeddings", False),
        norm_eps=_get(hf_config, "rms_norm_eps", 1e-6),
        moe_experts_per_tok=_get(hf_config, "top_k_experts", 0) or _get(hf_config, "num_experts_per_tok", 0) or 0,
        moe_num_experts=_get(hf_config, "num_experts", 0) or 0,
        max_seq_len=2048,
        weight_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=float(local_theta),
        attention_types=attention_types,
        sliding_window_size=_get(hf_config, "sliding_window", None),
        final_logit_softcap=_get(hf_config, "final_logit_softcapping", None),
        global_kv_heads=global_kv_heads,
        global_head_dim=_get(hf_config, "global_head_dim", None),
        global_k_eq_v=_get(hf_config, "attention_k_eq_v", False),
        global_rope_proportion=global_rope.get("partial_rotary_factor", None),
        local_rope_proportion=local_rope.get("partial_rotary_factor", None),
        local_base_frequency=int(local_theta),
        global_base_frequency=int(global_theta),
        per_layer_input_dim=_get(hf_config, "hidden_size_per_layer_input", 0) or 0,
        vocab_size_per_layer_input=_get(hf_config, "vocab_size_per_layer_input", 0) or 0,
        num_kv_shared_layers=_get(hf_config, "num_kv_shared_layers", 0) or 0,
        use_double_wide_mlp=bool(use_double_wide_mlp),
    )


def load_config(config_path: PathT) -> "Config":
    config = json.loads(Path(config_path).read_text())
    return hf_to_jax_config(config["text_config"] if "text_config" in config else config)


def load_tokenizer(
    tokenizer_path: PathT | None, tokenizer_config_path: PathT | None, chat_template: PathT | None = None
) -> "PreTrainedTokenizerFast":  # noqa: F821
    from transformers import PreTrainedTokenizerFast

    config = json.loads(Path(tokenizer_config_path).read_text())
    if chat_template is not None:
        config["chat_template"] = Path(chat_template).read_text()
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ArrayInfo:
    shape: tuple[int, ...] = static_field()
    dtype: jax.typing.DTypeLike = static_field()
    logical_axes: tuple = static_field()
    initializer: Callable | None = static_field(None)


# module reload friendly isinstance check
specof = lambda x: jax.typeof(x).sharding.spec
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, ArrayInfo)
_count_left_padding = lambda ids, pad_id=PAD_ID: auto_axes(
    lambda ids: jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1), out_sharding=P(specof(ids)[0])
)(ids)
_length_minus_right_padding = lambda segment_ids: auto_axes(
    lambda segment_ids: jnp.sum(jnp.cumsum(jnp.flip(segment_ids != 0, -1), axis=-1) > 0, -1),
    out_sharding=P(specof(segment_ids)[0]),
)(segment_ids)
which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform
_he_normal = lru_cache(jax.nn.initializers.he_normal)
_const_init = lru_cache(jax.nn.initializers.constant)


@lru_cache
def _init_leaves(abstract, shardings):
    @partial(jax.jit, out_shardings=shardings)
    def _init_fn(key):
        num_leaves = len(jax.tree.leaves(abstract, is_leaf=is_param))  # one new RNG key per tensor
        k = iter(random.split(key, num_leaves))
        return jax.tree.map(lambda info: info.initializer(next(k), info.shape, info.dtype), abstract, is_leaf=is_param)

    return _init_fn


class _Init:
    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Config, *args, **kw):
        abstract = cls.abstract(cfg, *args, **kw)
        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules), abstract, is_leaf=is_param
        )

    @classmethod
    def init(cls, key: jax.typing.ArrayLike, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules), abstract, is_leaf=is_param
        )
        abstract_leaves, abstract_struct = jax.tree.flatten(abstract, is_leaf=is_param)
        shardings_leaves = jax.tree.leaves(shardings, is_leaf=is_param)
        return jax.tree.unflatten(abstract_struct, _init_leaves(tuple(abstract_leaves), tuple(shardings_leaves))(key))


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class QuantArray:
    quant: jax.Array | ArrayInfo
    scale: jax.Array | ArrayInfo
    out_scaling: bool = static_field(False)
    scale_expand_dims: int | tuple[int, ...] = static_field(())
    shape = property(lambda self: self.quant.shape)
    ndim = property(lambda self: self.quant.ndim)


def einsum(subscripts: str, lhs: jax.Array, rhs: jax.Array | QuantArray, out_sharding: P | None = None):
    """jnp.einsum wrapper that handles regular arrays and QuantArrays"""
    if is_type(rhs, QuantArray):
        scale = jnp.expand_dims(rhs.scale, rhs.scale_expand_dims)
        if rhs.out_scaling:
            return jnp.einsum(subscripts, lhs, rhs.quant, out_sharding=out_sharding) * scale
        else:
            return jnp.einsum(subscripts, lhs * scale, rhs.quant, out_sharding=out_sharding)
    else:
        return jnp.einsum(subscripts, lhs, rhs, out_sharding=out_sharding)


_int8_quant_init = lambda key, shape, dtype=jnp.int8: random.randint(key, shape, -128, 128, dtype=dtype)
_int8_scale_init = lambda key, shape, dtype: random.normal(key, shape, dtype=dtype) / math.sqrt(math.prod(shape)) / 127


def quantize(x: jax.Array | ArrayInfo, axis: int | tuple[int, ...], scale_dtype=jnp.bfloat16, zero_init: bool = False):
    if is_type(x, QuantArray):
        raise ValueError("Attempting to quantize an already quantized QuantArray.")
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    axis = tuple(z % len(x.shape) for z in axis)

    if isinstance(x, jax.Array):
        axis = tuple(z % x.ndim for z in axis)
        amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
        quant = jnp.round(x / scale).astype(jnp.int8)
        scale = scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])
        return quant, scale

    if is_type(x, ArrayInfo):
        new_shape = tuple(ax for i, ax in enumerate(x.shape) if i not in axis)
        new_logical_axes = tuple(ax for i, ax in enumerate(x.logical_axes) if i not in axis)
        if zero_init:
            quant_init, scale_init = jax.nn.initializers.zeros, jax.nn.initializers.ones
        else:
            quant_init, scale_init = _int8_quant_init, _int8_scale_init
        quant = dataclasses.replace(x, shape=x.shape, dtype=jnp.int8, initializer=quant_init)
        scale = ArrayInfo(new_shape, scale_dtype, new_logical_axes, scale_init)
        return quant, scale
    raise ValueError(f"quantize got unexpected type: {type(x)}")


def update_slice(
    x: jax.Array | QuantArray, y: jax.Array | QuantArray, pos: int | jax.Array, update_axis: int, quant_axis: int = -1
):
    """dynamic_update_slice wrapper that handles regular arrays and QuantArrays"""
    if is_type(x, QuantArray):
        assert x.quant.ndim == y.ndim
        quant_axis, update_axis = quant_axis % x.quant.ndim, update_axis % x.quant.ndim  # normalize axis numbers
        if is_type(y, QuantArray):
            y_quant, y_scale = y.quant, y.scale
        else:
            y_quant, y_scale = quantize(y, axis=quant_axis, scale_dtype=x.scale.dtype)  # quantize rhs
        y_quant = reshard(y_quant.astype(x.quant.dtype), specof(x.quant))
        y_scale = reshard(y_scale.astype(x.scale.dtype), specof(x.scale))
        scale_update_axis = [ax for ax in range(x.quant.ndim) if ax != quant_axis][update_axis]
        new_quant = update_slice(x.quant, y_quant, pos, update_axis=update_axis)
        new_scale = update_slice(x.scale, y_scale, pos, update_axis=scale_update_axis)
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        assert x.ndim == y.ndim
        y = reshard(y.astype(x.dtype), specof(x))
        return jax.lax.dynamic_update_slice_in_dim(x, y, pos, axis=update_axis)


def _zeros_like(x: jax.Array | QuantArray, new_size: int, update_axis: int, quant_axis: int = -1):
    new_shape = [s if i != update_axis else new_size for i, s in enumerate(x.shape)]
    if is_type(x, QuantArray):
        new_scale_shape = [s for i, s in enumerate(new_shape) if i != (quant_axis % len(new_shape))]
        quant_new = jnp.zeros_like(x.quant, shape=new_shape, out_sharding=specof(x.quant))
        scale_new = jnp.zeros_like(x.scale, shape=new_scale_shape, out_sharding=specof(x.scale))
        return dataclasses.replace(x, quant=quant_new, scale=scale_new)
    else:
        return jnp.zeros_like(x, shape=new_shape, out_sharding=specof(x))


def _dynamic_slice(x: jax.Array | QuantArray, pos: int | jax.Array, size: int, axis: int, quant_axis: int = -1):
    if is_type(x, QuantArray):
        axis, quant_axis = axis % x.ndim, quant_axis % x.ndim
        assert quant_axis > axis
        new_quant = jax.lax.dynamic_slice_in_dim(x.quant, pos, size, axis)
        new_scale = jax.lax.dynamic_slice_in_dim(x.scale, pos, size, axis)
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        return jax.lax.dynamic_slice_in_dim(x, pos, size, axis)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AttentionLayer(_Init):
    q: jax.Array | ArrayInfo | QuantArray
    k: jax.Array | ArrayInfo | QuantArray | None
    v: jax.Array | ArrayInfo | QuantArray | None
    o: jax.Array | ArrayInfo | QuantArray
    q_gamma: jax.Array | ArrayInfo | QuantArray
    k_gamma: jax.Array | ArrayInfo | QuantArray | None

    ######################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config, idx: str) -> "AttentionLayer":
        layer_type = cfg.attention_types[idx]
        if layer_type not in ("local_attention", "global_attention"):
            raise ValueError(f"Unsupported layer type: {layer_type=} not in ('local_attention', 'global_attention').")
        _init, dtype = _he_normal, cfg.weight_dtype
        is_shared = idx >= cfg.num_layers - max(cfg.num_kv_shared_layers, 0)
        is_global = layer_type == "global_attention"
        head_dim = cfg.global_head_dim if is_global else cfg.local_head_dim
        kv_heads = cfg.global_kv_heads if is_global else cfg.local_kv_heads
        use_k_eq_v = cfg.global_k_eq_v and is_global
        kv_proj_info = ArrayInfo(
            (cfg.embed, kv_heads, head_dim), dtype, ("qkv_embed", "kv_heads", "head_dim"), _init(0, (1, 2))
        )
        layer = AttentionLayer(
            q=ArrayInfo(
                (cfg.embed, cfg.q_heads, head_dim), dtype, ("qkv_embed", "q_heads", "head_dim"), _init(0, (1, 2))
            ),
            k=None if is_shared else kv_proj_info,
            v=None if (is_shared or use_k_eq_v) else kv_proj_info,
            o=ArrayInfo(
                (cfg.q_heads, head_dim, cfg.embed), dtype, ("o_heads", "head_dim", "o_embed"), _init(0, (1, 2))
            ),
            q_gamma=ArrayInfo((head_dim,), dtype, ("head_dim",), jax.nn.initializers.ones),
            k_gamma=None if is_shared else ArrayInfo((head_dim,), dtype, ("head_dim",), jax.nn.initializers.ones),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "AttentionLayer", cfg: Config):
        if not cfg.quant_attn:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        v_q = (
            QuantArray(*quantize(layer.v, 0, scale_dtype), out_scaling=True, scale_expand_dims=-2)
            if layer.v is not None
            else None
        )
        k_q = (
            QuantArray(*quantize(layer.k, 0, scale_dtype), out_scaling=True, scale_expand_dims=-2)
            if layer.k is not None
            else None
        )
        q_q = QuantArray(*quantize(layer.q, 0, scale_dtype), out_scaling=True, scale_expand_dims=-2)
        o_q = QuantArray(*quantize(layer.o, (0, 1), scale_dtype), out_scaling=True)
        return dataclasses.replace(layer, q=q_q, k=k_q, v=v_q, o=o_q)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MLPLayer(_Init):
    w_gate: jax.Array | ArrayInfo | QuantArray
    w_up: jax.Array | ArrayInfo | QuantArray | None
    w_down: jax.Array | ArrayInfo | QuantArray

    ######################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config, up_layer: bool = True) -> "MLPLayer":
        _init, dtype = _he_normal, cfg.weight_dtype
        w_up = None
        if up_layer:
            w_up = ArrayInfo((cfg.embed, cfg.mlp_ffw_size), dtype, ("mlp_up_embed", "mlp_up_ffw"), _init(0, 1))
        layer = MLPLayer(
            w_gate=ArrayInfo((cfg.embed, cfg.mlp_ffw_size), dtype, ("mlp_up_embed", "mlp_up_ffw"), _init(0, 1)),
            w_up=w_up,
            w_down=ArrayInfo((cfg.mlp_ffw_size, cfg.embed), dtype, ("mlp_down_ffw", "mlp_down_embed"), _init(0, 1)),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MLPLayer", cfg: Config):
        if not cfg.quant_mlp:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        w_up = None if layer.w_up is None else QuantArray(*quantize(layer.w_up, 0, scale_dtype), out_scaling=True)
        return dataclasses.replace(
            layer,
            w_gate=QuantArray(*quantize(layer.w_gate, 0, scale_dtype), out_scaling=True),
            w_up=w_up,
            w_down=QuantArray(*quantize(layer.w_down, 0, scale_dtype), out_scaling=True),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoELayer(_Init):
    w_router: jax.Array | ArrayInfo | QuantArray
    w_router_scale: jax.Array | ArrayInfo
    per_expert_scale: jax.Array | ArrayInfo
    we_gate: jax.Array | ArrayInfo | QuantArray
    we_up: jax.Array | ArrayInfo | QuantArray
    we_down: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _einit, _sinit, dtype = (
            _he_normal(in_axis=0, out_axis=(1, 2)),
            _he_normal(in_axis=0, out_axis=1),
            cfg.weight_dtype,
        )
        assert cfg.moe_ffw_size > 0
        layer = MoELayer(
            w_router=ArrayInfo((cfg.embed, cfg.moe_num_experts), cfg.moe_gate_dtype, ("moe_e_up_embed", None), _sinit),
            w_router_scale=ArrayInfo((cfg.embed,), cfg.moe_gate_dtype, (None,), _const_init(1.0)),
            we_gate=ArrayInfo(
                (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size),
                dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_up=ArrayInfo(
                (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size),
                dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_down=ArrayInfo(
                (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed),
                dtype,
                ("moe_e_experts", "moe_e_down_ffw", "moe_e_down_embed"),
                _einit,
            ),
            per_expert_scale=ArrayInfo((cfg.moe_num_experts,), dtype, (None,), _const_init(1.0)),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MoELayer", cfg: Config):
        if not cfg.quant_moe:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            we_gate=QuantArray(*quantize(layer.we_gate, 1, scale_dtype), out_scaling=True),
            we_up=QuantArray(*quantize(layer.we_up, 1, scale_dtype), out_scaling=True),
            we_down=QuantArray(*quantize(layer.we_down, 1, scale_dtype), out_scaling=True),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Layer(_Init):
    mlp: MLPLayer
    moe: MoELayer | None
    attn: AttentionLayer
    attn_pre_gamma: jax.Array | ArrayInfo
    attn_post_gamma: jax.Array | ArrayInfo
    ffw_pre_gamma_mlp: jax.Array | ArrayInfo
    ffw_post_gamma_mlp: jax.Array | ArrayInfo | None  # only if moe is not None
    ffw_pre_gamma_moe: jax.Array | ArrayInfo | None  # only if moe is not None
    ffw_post_gamma_moe: jax.Array | ArrayInfo | None  # only if moe is not None
    ffw_post_gamma: jax.Array | ArrayInfo
    post_mlp: MLPLayer | None
    post_mlp_post_gamma: jax.Array | ArrayInfo | None
    layer_scalar: jax.Array | ArrayInfo | None

    ######################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config, layer_idx: int) -> "Layer":
        is_moe = cfg.moe_num_experts > 0
        _gamma = lambda: ArrayInfo((cfg.embed,), cfg.weight_dtype, ("act_embed",), _const_init(1.0))
        has_per_layer_input = cfg.per_layer_input_dim > 0
        first_kv_shared = cfg.num_layers - cfg.num_kv_shared_layers
        is_kv_shared = cfg.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared
        mlp_ffw_size = cfg.mlp_ffw_size * (2 if cfg.use_double_wide_mlp and is_kv_shared else 1)
        post_mlp = None
        if has_per_layer_input:
            post_mlp = MLPLayer.abstract(dataclasses.replace(cfg, mlp_ffw_size=cfg.per_layer_input_dim), up_layer=False)
        layer = Layer(
            mlp=MLPLayer.abstract(dataclasses.replace(cfg, mlp_ffw_size=mlp_ffw_size)),
            moe=MoELayer.abstract(cfg) if is_moe else None,
            attn=AttentionLayer.abstract(cfg, layer_idx),
            attn_pre_gamma=_gamma(),
            attn_post_gamma=_gamma(),
            ffw_pre_gamma_mlp=_gamma(),
            ffw_post_gamma_mlp=_gamma() if is_moe else None,
            ffw_pre_gamma_moe=_gamma() if is_moe else None,
            ffw_post_gamma_moe=_gamma() if is_moe else None,
            ffw_post_gamma=_gamma(),
            post_mlp=post_mlp,
            post_mlp_post_gamma=_gamma() if has_per_layer_input else None,
            layer_scalar=ArrayInfo((), cfg.weight_dtype, (), _const_init(1.0)),
        )
        return layer

    @staticmethod
    def quantize(layer: "Layer", cfg: Config):
        return dataclasses.replace(
            layer,
            mlp=layer.mlp.quantize(layer.mlp, cfg),
            attn=layer.attn.quantize(layer.attn, cfg),
            moe=layer.moe.quantize(layer.moe, cfg) if layer.moe is not None else None,
            post_mlp=layer.post_mlp.quantize(layer.post_mlp, cfg) if layer.post_mlp is not None else None,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Weights(_Init):
    layers: list[Layer]
    embedding: jax.Array | ArrayInfo
    gamma_final: jax.Array | ArrayInfo
    lm_head: jax.Array | ArrayInfo | None
    per_layer_embed: jax.Array | ArrayInfo | None
    per_layer_proj: jax.Array | ArrayInfo | None
    per_layer_proj_gamma: jax.Array | ArrayInfo | None

    @classmethod
    def abstract(cls, cfg: Config):
        init, dtype = _he_normal, cfg.weight_dtype
        layers = [Layer.abstract(cfg, layer_idx) for layer_idx in range(cfg.num_layers)]
        lm_head = None
        if not cfg.tie_embed:
            lm_head = ArrayInfo((cfg.embed, cfg.vocab_size), dtype, ("vocab_in", "vocab_out"), init(1, 0))

        per_layer_embed, per_layer_proj, per_layer_proj_gamma = None, None, None
        if cfg.per_layer_input_dim > 0:
            pli_total = cfg.num_layers * cfg.per_layer_input_dim
            per_layer_embed = ArrayInfo((cfg.vocab_size_per_layer_input, pli_total), dtype, (None, None), init(0, 1))
            per_layer_proj = ArrayInfo((cfg.embed, pli_total), dtype, (None, None), init(0, 1))
            per_layer_proj_gamma = ArrayInfo((cfg.per_layer_input_dim,), dtype, (None,), _const_init(0.0))

        return Weights(
            layers=layers,
            embedding=ArrayInfo((cfg.vocab_size, cfg.embed), dtype, (None, "vocab_in"), init(0, 1)),
            gamma_final=ArrayInfo((cfg.embed,), dtype, ("act_embed",), _const_init(1.0)),
            lm_head=lm_head,
            per_layer_embed=per_layer_embed,
            per_layer_proj=per_layer_proj,
            per_layer_proj_gamma=per_layer_proj_gamma,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class KVCache(_Init):
    k: list[jax.Array]  # (batch_size, key_heads, max_seq_len, head_dim)
    v: list[jax.Array]  # (batch_size, key_heads, max_seq_len, head_dim)
    iters: list[jax.Array]  # []  # sequences are right-aligned for slice udpate performance
    starts: list[jax.Array]  # [batch_size]  # sequences are right-aligned, we need start indices
    time_axis: int = static_field(2)
    sizes: list[int] = dataclasses.field(default_factory=list, metadata=dict(static=True))
    _first_global_layer_idx: int = static_field(-1)

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int):
        local_max_seq_len = min(cfg.sliding_window_size or cfg.max_seq_len, cfg.max_seq_len)
        local_shape = (batch_size, cfg.local_kv_heads, local_max_seq_len, cfg.local_head_dim)
        global_shape = (batch_size, cfg.global_kv_heads, cfg.max_seq_len, cfg.global_head_dim)
        dtype = cfg.weight_dtype
        _zeros = jax.nn.initializers.zeros
        local_info = ArrayInfo(local_shape, dtype, ("batch", "kv_heads", "sequence", "head_dim"), _zeros)
        global_info = ArrayInfo(global_shape, dtype, ("batch", "kv_heads", "sequence", "head_dim"), _zeros)
        num_kv_layers = (cfg.num_layers - cfg.num_kv_shared_layers) if cfg.num_kv_shared_layers > 0 else cfg.num_layers
        is_local = [cfg.attention_types[i] == "local_attention" for i in range(num_kv_layers)]
        sizes = [local_max_seq_len if is_local[i] else cfg.max_seq_len for i in range(num_kv_layers)]
        cache = KVCache(
            k=[local_info if is_local[i] else global_info for i in range(num_kv_layers)],
            v=[local_info if is_local[i] else global_info for i in range(num_kv_layers)],
            iters=[ArrayInfo((), jnp.int32, (), _const_init(-1)) for _ in range(num_kv_layers)],
            starts=[ArrayInfo((batch_size,), jnp.int32, ("batch",), _const_init(0)) for _ in range(num_kv_layers)],
            sizes=sizes,
            _first_global_layer_idx=cfg.attention_types.index("global_attention"),
        )
        if cfg.quant_cache:
            _quantize = partial(quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype, zero_init=True)
            cache = dataclasses.replace(
                cache,
                k=[
                    QuantArray(*_quantize(cache.k[idx]), out_scaling=True, scale_expand_dims=(-2, -3))
                    for idx in range(len(cache.k))
                ],
                v=[
                    QuantArray(*_quantize(cache.v[idx]), out_scaling=False, scale_expand_dims=(-2, -3))
                    for idx in range(len(cache.v))
                ],
            )
        return cache

    def unroll_kvs(self, idx: int):
        min_start = jnp.min(self.starts[idx])
        k, v = self.k[idx], self.v[idx]
        kvs_unrolled = []
        for buf in [k, v]:
            if buf is None:
                kvs_unrolled.append(None)
                continue
            if is_type(k, QuantArray):
                quant_unrolled = jnp.roll(buf.quant, -min_start, axis=self.time_axis)
                scale_unrolled = jnp.roll(buf.scale, -min_start, axis=self.time_axis - 1)
                buf = dataclasses.replace(buf, quant=quant_unrolled, scale=scale_unrolled)
            else:
                buf = jnp.roll(buf, -min_start, axis=self.time_axis)
            kvs_unrolled.append(buf)
        new_starts = (self.starts[idx] - min_start) % self.sizes[idx]
        new_iter = jnp.maximum(self.iters[idx], 0) - min_start
        return tuple(kvs_unrolled), new_iter, new_starts

    @staticmethod
    def fill_len(iter, starts, size) -> jax.Array:
        return jnp.where(iter >= 0, (iter - (starts + 1)) % size + 1, 0)

    def global_fill_len(self) -> jax.Array:
        assert (idx := self._first_global_layer_idx) >= 0
        return self.fill_len(self.iters[idx], self.starts[idx], self.sizes[idx])

    @property
    def buffers(self) -> tuple[jax.Array | QuantArray, ...]:
        raise NotImplementedError


def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""
    scan_fun = lambda a, b: ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])
    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")


def _generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    cfg: Config,
    layer_type: str,
) -> tuple[jax.Array, jax.Array]:
    """Generate Sin/Cos for Rotary Embeddings."""
    if layer_type == "global_attention":
        rope_theta = cfg.global_base_frequency
        head_dim = cfg.global_head_dim or features
        rope_proportion = cfg.global_rope_proportion or 1.0
    elif layer_type == "local_attention":
        rope_theta = cfg.local_base_frequency
        head_dim = features
        rope_proportion = cfg.local_rope_proportion or 1.0
    else:
        raise ValueError(f"Unknown {layer_type=}. Supported values are 'global_attention' and 'local_attention'.")
    rope_angles = int(rope_proportion * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    rotary_freq = 1.0 / (rope_theta ** (jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32) / head_dim))
    rotational_frequency = jnp.concatenate([rotary_freq, jnp.zeros(nope_angles)]) if nope_angles > 0 else rotary_freq
    # Must use high precision einsum here, since rounding off to a bfloat16 is catastrophic. bfloat16 rounds 257 to 256,
    # but sin(257) is very different from sin(256).
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST, out_sharding=P()
    )
    sin, cos = jnp.sin(sinusoid_inp).astype(jnp.float32), jnp.cos(sinusoid_inp).astype(jnp.float32)
    return sin, cos


def apply_rotary_embedding(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_attention_mask(
    q_len, k_len, q_segment_ids, kv_segment_ids, q_offset, kv_offset, causal: bool, sliding_window: int | None = None
):
    segment_mask = (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])[:, None, :, :]  # [B, 1, t, T]
    if causal:
        qk = (1, 1, q_len, k_len)  # [b, h, t, T]
        q_positions = jax.lax.broadcasted_iota(jnp.int32, qk, 2) + q_offset[:, None, None, None]
        kv_positions = (jax.lax.broadcasted_iota(jnp.int32, qk, 3) + kv_offset[:, None, None, None]) % k_len
        causal_mask = q_positions >= kv_positions
        if sliding_window is not None:
            window_mask = (q_positions - kv_positions) < sliding_window
            causal_mask = causal_mask & window_mask
        return segment_mask & causal_mask
    return segment_mask


@partial(auto_axes, out_sharding=P(BATCH_AXIS_NAME, ATTN_HEADS_AXIS_NAME, None, None))
def attention(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
    cfg: Config,
    sliding_window: int | None = None,
) -> jax.Array:
    """
    Compute attention.

    Args:
      q: Query tensor of shape (batch_size, num_heads, q_len, head_dim)
      k: Key tensor of shape (batch_size, num_heads, k_len, head_dim)
      v: Value tensor of shape (batch_size, num_heads, k_len, head_dim)
      q_segment_ids: Query segment IDs of shape (batch_size, q_len)
      k_segment_ids: Key segment IDs of shape (batch_size, k_len)
      q_offset: Query offset of shape (batch_size,)
      kv_offset: Key offset of shape (batch_size,)
      cfg: Configuration object.
      sliding_window: If not None, limits attention to this many past positions.

    Returns:
      Attention output of shape (batch_size, num_heads, q_len, head_dim)
    """
    # grouped-query attention
    b, qh, t, d = q.shape
    _, kh, T, _ = k.shape
    scale = 1.0

    q_ = q.reshape((b, kh, qh // kh, t, d))
    qk = einsum("bhgtd,bhTd->bhgtT", q_, k) * scale
    qk = qk.reshape((b, qh, t, T))

    mask = make_attention_mask(t, T, q_segment_ids, kv_segment_ids, q_offset, kv_offset, cfg.causal, sliding_window)

    # Apply the combined mask
    qk = jnp.where(mask, qk, -1e30)
    # jax softmax impl includes max subtraction for numerical stability, no need to do it outside.
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)

    # grouped-query attention
    attn_ = attn.reshape((b, kh, qh // kh, t, T))
    qkv = einsum("bhgtN,bhNd->bhgtd", attn_, v).astype(cfg.dtype)
    return qkv.reshape((b, qh, t, d))


def attention_kernel(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
    *,
    cfg: Config,
    sliding_window: int | None = None,
) -> jax.Array:
    """Flash attention kernel!"""
    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)

    # handle grouped query attention
    assert q.shape[-3] % k.shape[-3] == 0
    scale = 1.0

    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)
    q_shape, kv_repeats = q.shape, q.shape[-3] // k.shape[-3]
    kv_repeats_spec = tuple(set(*l2p("q_heads")) - set(*l2p("kv_heads")))
    kv_repeats_spec = kv_repeats_spec if len(kv_repeats_spec) > 0 else (None,)
    q_spec = P(*(l2p("batch", "kv_heads") + kv_repeats_spec + l2p("sequence", "head_dim")))
    q = jax.lax.reshape(q, (q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1])), out_sharding=q_spec)

    # shard_map
    in_specs = (
        q_spec,  # q
        l2p("batch", "kv_heads", "sequence", "head_dim"),  # k
        l2p("batch", "kv_heads", "sequence", "head_dim"),  # v
        l2p("batch", "sequence"),  # q_segment_ids
        l2p("batch", "sequence"),  # kv_segment_ids
        l2p("batch"),  # q_offset
        l2p("batch"),  # kv_offset
    )
    in_specs += (None if k_scale is None else l2p("batch", "kv_heads", "sequence"),)  # k_scales
    in_specs += (None if v_scale is None else l2p("batch", "kv_heads", "sequence"),)  # v_scales

    @partial(jax.shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=q_spec, check_vma=False)
    def _f(q, k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset, k_scale, v_scale) -> jax.Array:
        q_seq, kv_seq, kv_heads = q.shape[-2], v.shape[-2], v.shape[-3]
        block_q, block_kv = min(q_seq, 512), min(kv_seq, 512)
        block_sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=min(512, block_kv))

        def attn_dynamic_fn(q, k, v, q_segment_ids, kv_segment_ids):
            segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
            mask = make_attention_mask(
                q_seq, kv_seq, segment_ids.q, segment_ids.kv, q_offset, kv_offset, cfg.causal, sliding_window
            )
            if which_platform(cfg) == "tpu":
                attn_fn = lambda q, k, v, segment_ids, mask: (
                    splash.make_splash_mqa_single_device(mask=mask, block_sizes=block_sizes)(q, k, v, segment_ids)
                )
            else:

                def attn_fn(q, k, v, _, mask):
                    q, k, v = q.swapaxes(0, 1), k[:, None, :], v[:, None, :]
                    impl = "cudnn" if which_platform(cfg) == "gpu" else "xla"
                    return jax.nn.dot_product_attention(q, k, v, mask=mask, implementation=impl).swapaxes(0, 1)

            attn_fn = jax.vmap(attn_fn, (0, 0, 0, None, None))  # map over kv heads for mqa
            attn_fn = jax.vmap(attn_fn, (0, 0, 0, 0, 0))  # map over batch
            return jnp.where(jnp.any(mask, axis=-1)[..., None, :, None], attn_fn(q, k, v, segment_ids, mask), 0.0)

        if k_scale is not None:
            k = (k * k_scale[..., None]).astype(jnp.bfloat16)
        if v_scale is not None:
            v = (v * v_scale[..., None]).astype(jnp.bfloat16)
        return attn_dynamic_fn(q * scale, k, v, q_segment_ids, kv_segment_ids)

    out = _f(q, k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset, k_scale, v_scale).astype(cfg.dtype)
    return jax.lax.reshape(out, q_shape, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim"))


def rms_norm(x: jax.Array, gamma: jax.Array | None, eps: jax.Array | float) -> jax.Array:
    """Apply RMS normalization."""
    normed = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return ((normed * gamma) if gamma is not None else normed).astype(x.dtype)


# A KV representation necessary for computing attention, so KV & mask metadata.
class _CacheLayerState(NamedTuple):
    k: jax.Array | QuantArray
    v: jax.Array | QuantArray | None
    q_segment_ids: jax.Array
    kv_segment_ids: jax.Array
    q_offset: jax.Array
    kv_offset: jax.Array

# A KV representation needed to update the KV-cache.
class _CacheLayerUpdate(NamedTuple):
    k: jax.Array | QuantArray
    v: jax.Array | QuantArray | None
    iter: jax.Array
    starts: jax.Array


def _build_cache_state(k, v, starts, fill_len, segment_ids, q_offset) -> _CacheLayerState:
    size = k.shape[-2]
    physical_idx = jnp.arange(size)[None, :]
    logical_pos = (physical_idx - starts[:, None]) % size
    kv_segment_ids = logical_pos < fill_len[:, None]
    q_segment_ids = jnp.where(segment_ids != 0, True, False)
    kv_offset = -starts
    return _CacheLayerState(k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset)


def update_cache_space_sufficient(k_update, v_update, segment_ids, cache: KVCache, idx: int, quant_axis: int = -1):
    k, v = cache.k[idx], cache.v[idx]
    it = jnp.maximum(cache.iters[idx], 0)  # iter might be uninitialized
    k_new = update_slice(k, k_update, it, update_axis=cache.time_axis, quant_axis=quant_axis)
    v_new = update_slice(v, v_update, it, update_axis=cache.time_axis, quant_axis=quant_axis)

    num_new_tokens_per_elem = _length_minus_right_padding(segment_ids)
    num_new_tokens_max = jnp.max(num_new_tokens_per_elem)

    old_fill_len = jnp.where(cache.iters[idx] >= 0, (cache.iters[idx] - (cache.starts[idx] + 1)) % cache.sizes[idx] + 1, 0)
    cache_spill = old_fill_len + num_new_tokens_max > cache.sizes[idx]

    new_fill_len = jnp.minimum(old_fill_len + num_new_tokens_per_elem, cache.sizes[idx])
    new_it = (it + num_new_tokens_max) % cache.sizes[idx]
    new_starts = jnp.where(cache_spill, new_it, cache.starts[idx])

    q_offset = jnp.where(cache.iters[idx] >= 0, old_fill_len, -_count_left_padding(segment_ids))
    q_offset = jnp.where(cache_spill, q_offset - num_new_tokens_per_elem, q_offset)

    cache_state = _build_cache_state(k_new, v_new, new_starts, new_fill_len, segment_ids, q_offset)
    cache_update = _CacheLayerUpdate(k_new, v_new, new_it, new_starts)
    return cache_update, cache_state


def update_cache_space_insufficient(k_update, v_update, segment_ids, cache: KVCache, idx: int, quant_axis: int = -1):
    k, v = cache.k[idx], cache.v[idx]
    new_size = cache.sizes[idx] + k_update.shape[cache.time_axis]
    new_size = ((new_size + 512 - 1) // 512) * 512
    k_new = _zeros_like(k, new_size, cache.time_axis, quant_axis)
    v_new = _zeros_like(v, new_size, cache.time_axis, quant_axis)
    (k_unrolled, v_unrolled), iter_unrolled, starts_unrolled = cache.unroll_kvs(idx)
    k_new = update_slice(k_new, k_unrolled, 0, update_axis=cache.time_axis, quant_axis=quant_axis)
    k_new = update_slice(k_new, k_update, iter_unrolled, update_axis=cache.time_axis, quant_axis=quant_axis)
    v_new = update_slice(v_new, v_unrolled, 0, update_axis=cache.time_axis, quant_axis=quant_axis)
    v_new = update_slice(v_new, v_update, iter_unrolled, update_axis=cache.time_axis, quant_axis=quant_axis)
    initialized = cache.iters[idx] >= 0
    starts = jnp.where(initialized, starts_unrolled, _count_left_padding(segment_ids))

    num_new_tokens_per_elem = jnp.sum(segment_ids != 0, axis=-1)
    num_new_tokens_max = jnp.max(_length_minus_right_padding(segment_ids))
    old_fill_len = jnp.where(initialized, (cache.iters[idx] - (cache.starts[idx] + 1)) % cache.sizes[idx] + 1, 0)
    logical_iter = jnp.where(initialized, iter_unrolled + num_new_tokens_max - starts, 0)
    fill_len = jnp.where(initialized, jnp.maximum(old_fill_len, logical_iter), num_new_tokens_per_elem)
    q_offset = jnp.where(initialized, old_fill_len, -_count_left_padding(segment_ids))
    cache_state = _build_cache_state(k_new, v_new, starts, fill_len, segment_ids, q_offset)

    slice_start = jnp.maximum(0, iter_unrolled + num_new_tokens_max - cache.sizes[idx])
    k_cache = _dynamic_slice(k_new, slice_start, cache.sizes[idx], cache.time_axis, quant_axis)
    v_cache = _dynamic_slice(v_new, slice_start, cache.sizes[idx], cache.time_axis, quant_axis)
    new_iter = iter_unrolled + num_new_tokens_max - slice_start
    new_starts = jnp.maximum(0, starts - slice_start)
    cache_update = _CacheLayerUpdate(k_cache, v_cache, new_iter % cache.sizes[idx], new_starts)
    return cache_update, cache_state


def attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: AttentionLayer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    *,
    idx: int,
    shared_kvs: _CacheLayerState | None = None,
):
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    x = x.astype(cfg.dtype)
    is_sliding = cfg.attention_types[idx] != "global_attention"
    sliding_window = cfg.sliding_window_size if is_sliding else None

    # Multi-head attention
    with jax.named_scope("qkv_matmul"):
        q = einsum("btd,dhq->bhtq", x, layer.q).astype(cfg.dtype)
        if shared_kvs is None:
            k = einsum("btd,dhq->bhtq", x, layer.k).astype(cfg.dtype)
            v = einsum("btd,dhq->bhtq", x, layer.v).astype(cfg.dtype) if layer.v is not None else k  # k_eq_v

    # Apply rotary embeddings and norms
    with jax.named_scope("rope"):
        q = rms_norm(q, layer.q_gamma, cfg.norm_eps)
        q = apply_rotary_embedding(q, sin, cos)
        if shared_kvs is None:
            k = rms_norm(k, layer.k_gamma, cfg.norm_eps)
            k = apply_rotary_embedding(k, sin, cos)
            v = rms_norm(v, None, cfg.norm_eps)  # v_norm: normalize-only, no learnable scale

    with jax.named_scope("cache_update"):
        if shared_kvs is None:
            if is_type(cache, KVCache):
                if q.shape[cache.time_axis] == 1:  # decode always fits in the cache
                    cache_update, cache_state = update_cache_space_sufficient(k, v, segment_ids, cache, idx, -1)
                else:
                    cache_update, cache_state = update_cache_space_insufficient(k, v, segment_ids, cache, idx, -1)
                k, v = cache_state.k, cache_state.v
                q_segment_ids, kv_segment_ids = cache_state.q_segment_ids, cache_state.kv_segment_ids
                q_offset, kv_offset = cache_state.q_offset, cache_state.kv_offset
            else:
                q_segment_ids, kv_segment_ids = segment_ids, segment_ids
                starts = _count_left_padding(kv_segment_ids, 0)  # pad_id=0 for segment_ids
                q_offset, kv_offset = -starts, -starts
                cache_update = (k, v)
                cache_state = _CacheLayerState(k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset)
        else:
            k, v = shared_kvs.k, shared_kvs.v
            q_segment_ids, kv_segment_ids = shared_kvs.q_segment_ids, shared_kvs.kv_segment_ids
            q_offset, kv_offset = shared_kvs.q_offset, shared_kvs.kv_offset
            cache_update, cache_state = None, shared_kvs

    with jax.named_scope("attention"):  # Compute attention
        attn_args = (q, k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset)
        if cfg.use_prefill_attn_kernel and q.shape[-2] != 1:
            attn_out = attention_kernel(*attn_args, cfg=cfg, sliding_window=sliding_window)
        else:
            attn_out = attention(*attn_args, cfg, sliding_window)

    # Project attention output
    with jax.named_scope("projection"):
        attn_out = einsum(
            "bhtq,hqd->btd", attn_out, layer.o, out_sharding=l2p("batch", "sequence", "act_embed")
        ).astype(cfg.dtype)
    return attn_out, (cache_update, cache_state)


@partial(jax.jit, static_argnames=("replicated_routing",))
def _route_tokens_to_moe_experts(
    x: jax.Array, weight: jax.Array, scale: jax.Array, replicated_routing: bool, cfg: Config
):
    l2p = lambda *axes: logical_to_physical(axes, cfg.rules)
    x_shape = x.shape
    x = x.reshape((-1, x.shape[-1]))
    # not distributing the routing work avoids communication for small batches
    x = reshard(x, l2p(None, None) if replicated_routing else P(TENSOR_AXIS_NAME, None))
    weight, scale = reshard(weight, l2p(None, None)), reshard(scale, l2p(None, None))

    z = rms_norm(x, None, cfg.norm_eps) * (cfg.embed**-0.5)
    scores = jnp.einsum("Sk,kj->Sj", z * scale, weight).astype(cfg.moe_gate_dtype)
    topk_weights, topk_idx = jax.lax.top_k(jax.nn.softmax(scores, axis=-1), cfg.moe_experts_per_tok)
    topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
    topk_weights = topk_weights.reshape(
        x_shape[:-1] + (cfg.moe_experts_per_tok,), out_sharding=l2p("batch", "sequence", None)
    )
    topk_idx = topk_idx.reshape(x_shape[:-1] + (cfg.moe_experts_per_tok,), out_sharding=l2p("batch", "sequence", None))
    return topk_weights, topk_idx


def _moe_gmm(lhs, rhs, group_sizes, topk_idx, cfg: Config):
    assert lhs.ndim == 2 and rhs.ndim == 3, f"{lhs.ndim=} != 2 and {rhs.ndim=} != 3"
    group_sizes = group_sizes.astype(jnp.int32)
    with jax.named_scope("ragged_dot"):
        if is_type(rhs, QuantArray):
            assert rhs.scale.ndim == 2 and rhs.scale.shape == (rhs.quant.shape[0], rhs.quant.shape[2])
            scale = jnp.take_along_axis(rhs.scale, topk_idx[:, None], axis=-2)
            ret = jax.lax.ragged_dot(lhs, rhs.quant, group_sizes) * scale
        else:
            ret = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    return ret.astype(cfg.dtype)


def moe_block(x: jax.Array, layer: MoELayer, cfg: Config):
    assert x.ndim == 3
    l2p = lambda *axes: logical_to_physical(axes, cfg.rules)
    _psc = lambda z, spec: reshard(z, P(*spec))
    _qpsc = lambda z, spec: dataclasses.replace(z, quant=_psc(z.quant, spec.quant), scale=_psc(z.scale, spec.scale))
    psc = lambda z, spec: _qpsc(z, spec) if is_type(z, QuantArray) else _psc(z, spec)

    # we're decoding or device count does not divide total token count
    replicated_routing = x.shape[-2] == 1 or (x.shape[-2] * x.shape[-3]) % jax.device_count() != 0
    topk_weights, topk_idx = _route_tokens_to_moe_experts(
        x, layer.w_router, layer.w_router_scale, replicated_routing, cfg
    )
    tensor_axname, expert_axname = l2p("moe_e_tp")[0], l2p("moe_e_ep")[0]

    x_spec = l2p("batch", "sequence", None)
    topk_weights_spec, topk_idx_spec = l2p("batch", "sequence", None), l2p("batch", "sequence", None)
    out_spec = l2p("batch", "sequence", None)

    we_gate_spec = l2p("moe_e_ep", None, "moe_e_tp")
    we_up_spec = l2p("moe_e_ep", None, "moe_e_tp")
    we_down_spec = l2p("moe_e_ep", "moe_e_tp", None)
    if all(is_type(z, QuantArray) for z in [layer.we_gate, layer.we_up, layer.we_down]):
        we_gate_spec = dataclasses.replace(layer.we_gate, quant=we_gate_spec, scale=P(we_gate_spec[0], we_gate_spec[2]))
        we_up_spec = dataclasses.replace(layer.we_up, quant=we_up_spec, scale=P(we_up_spec[0], we_up_spec[2]))
        we_down_spec = dataclasses.replace(layer.we_down, quant=we_down_spec, scale=P(we_down_spec[0], we_down_spec[2]))
    we_gate = psc(layer.we_gate, we_gate_spec)
    we_up = psc(layer.we_up, we_up_spec)
    we_down = psc(layer.we_down, we_down_spec)

    in_specs = (x_spec, we_gate_spec, we_up_spec, we_down_spec, topk_weights_spec, topk_idx_spec)

    is_embedding_sharded = l2p("act_embed")[0] is not None
    if is_embedding_sharded:  # activations are sharded
        out_spec = P(*(out_spec[:-1] + (tensor_axname,)))  # override last axis name
    if cfg.ep_strategy == "prefill":
        out_spec = P(*(out_spec[:-1] + (tensor_axname,)))  # override last axis name

    expert_count = cfg.mesh.shape.get(expert_axname, 1)
    tensor_count = cfg.mesh.shape.get(tensor_axname, 1)
    assert cfg.moe_num_experts % expert_count == 0
    expert_size = cfg.moe_num_experts // expert_count

    @partial(jax.shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_spec, check_vma=False)
    def _expert_fn(x, we_gate, we_up, we_down, topk_weights, topk_idx):
        (b, s, d), e = x.shape, cfg.moe_experts_per_tok
        expert_idx = jax.lax.axis_index(expert_axname) if expert_axname is not None else 0
        tensor_idx = jax.lax.axis_index(tensor_axname) if tensor_axname is not None else 0
        del tensor_idx
        topk_idx_ = topk_idx.reshape(-1)
        valid_group_mask_ = (topk_idx_ >= expert_size * expert_idx) & (topk_idx_ < expert_size * (expert_idx + 1))
        expert_mapped_topk_idx_ = jnp.where(valid_group_mask_, topk_idx_ - expert_idx * expert_size, 2**30)

        sort_idx_ = jnp.argsort(expert_mapped_topk_idx_, axis=-1)  # [b * s * e]
        isort_idx_ = jnp.argsort(sort_idx_)

        if cfg.ep_strategy == "prefill":
            truncate_size = round(2 * sort_idx_.size / expert_count)
            sort_idx_, isort_idx_ = sort_idx_[:truncate_size], isort_idx_[:truncate_size]

        topk_idx_sort_ = topk_idx_[sort_idx_]  # [b * s * e]
        expert_mapped_topk_idx_sort_ = expert_mapped_topk_idx_[sort_idx_]
        valid_group_mask_sort_ = expert_mapped_topk_idx_sort_ < 2**30
        expert_mapped_topk_idx_sort_ = jnp.where(expert_mapped_topk_idx_sort_ < 2**30, expert_mapped_topk_idx_sort_, 0)

        # equivalent to:
        # ```
        # x_repeat_ = jnp.repeat(x.reshape((-1, x.shape[-1])), e, axis=0)
        # x_repeat_sort_ = jnp.take_along_axis(x_repeat_, sort_idx_[:, None], axis=-2)  # [b * s, d]
        # ```
        x_repeat_sort_ = jnp.take_along_axis(
            x.reshape((-1, x.shape[-1])),
            sort_idx_[:, None] // e,
            axis=-2,  # index trick to avoid jnp.repeat
        )  # [b * s * e, d]

        group_sizes = jnp.bincount(topk_idx_sort_, length=cfg.moe_num_experts)
        group_sizes_shard = jax.lax.dynamic_slice_in_dim(group_sizes, expert_idx * expert_size, expert_size, 0)

        with jax.named_scope("we_gate"):
            ff_gate = _moe_gmm(x_repeat_sort_, we_gate, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg)
            ff_gate = jax.nn.gelu(ff_gate, approximate=True)
            ff_gate = jnp.where(valid_group_mask_sort_[..., None], ff_gate, 0)
        with jax.named_scope("we_up"):
            ff_up = _moe_gmm(x_repeat_sort_, we_up, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg)
        ff_gate_up = jnp.where(valid_group_mask_sort_[..., None], ff_gate * ff_up, 0)
        with jax.named_scope("we_down"):
            ff_out = _moe_gmm(ff_gate_up, we_down, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg)
            ff_out = jnp.where(valid_group_mask_sort_[..., None], ff_out, 0)  # expensive

        if cfg.ep_strategy == "prefill":
            rs_shape = math.ceil((ff_out.shape[-1] // tensor_count) / 256) * 256 * tensor_count
            pad_size = rs_shape - ff_out.shape[-1]
            ff_out = jnp.pad(ff_out, ((0, 0), (0, pad_size)))
            ff_out = jax.lax.psum_scatter(ff_out, axis_name=tensor_axname, scatter_dimension=1, tiled=True)
        ff_out = ff_out * topk_weights.reshape(-1)[sort_idx_][:, None]

        if cfg.ep_strategy == "prefill":
            with jax.named_scope("unpermute"):
                # unpermute tokens
                dtype = jnp.bfloat16
                dim_nums = jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1,), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
                )
                ff_out_expert = jax.lax.scatter_add(
                    jnp.zeros((b * s, ff_out.shape[-1]), dtype=dtype),
                    sort_idx_[..., None] // e,
                    ff_out.astype(dtype),
                    dim_nums,
                ).astype(dtype)
                ff_out_expert = ff_out_expert.astype(cfg.dtype)
        else:
            with jax.named_scope("unpermute"):
                ff_out = jnp.take_along_axis(ff_out, isort_idx_[..., None], axis=-2)
            with jax.named_scope("expert_summing"):
                ff_out_expert = jnp.sum(ff_out.reshape((b * s, e, d)), -2)
                ff_out_expert = ff_out_expert.astype(cfg.dtype)

        with jax.named_scope("experts_collective"):
            if cfg.ep_strategy == "prefill":
                if expert_axname is not None:
                    ff_out_expert = jax.lax.psum(ff_out_expert, expert_axname)
            else:
                # collectives
                if is_embedding_sharded:  # activations are supposed to be sharded on out
                    with jax.named_scope("tp_e_psum_scatter"):
                        ff_out_expert = jax.lax.psum_scatter(
                            ff_out_expert, tensor_axname, scatter_dimension=1, tiled=True
                        )
                    with jax.named_scope("ep_e_psum"):
                        if expert_axname is not None:
                            ff_out_expert = jax.lax.psum(ff_out_expert, expert_axname)
                else:
                    psum_axes = tensor_axname if expert_axname is None else (expert_axname, tensor_axname)
                    ff_out_expert = jax.lax.psum(ff_out_expert, psum_axes)
            ff_out_expert = ff_out_expert.reshape((b, s, ff_out_expert.shape[-1]))
            return ff_out_expert

    with jax.named_scope("moe_routed_expert"):
        x_ = psc(x, x_spec)
        per_expert_weight = auto_axes(
            lambda pes, idx: jnp.take_along_axis(pes[None, None, :], idx, axis=-1), out_sharding=topk_weights_spec
        )(layer.per_expert_scale, topk_idx)
        topk_weights_ = topk_weights * per_expert_weight
        ff_out_expert = _expert_fn(x_, we_gate, we_up, we_down, topk_weights_, topk_idx)[..., : x.shape[-1]]
    return psc(ff_out_expert, l2p("batch", "sequence", "act_embed"))


def mlp_block(x: jax.Array, layer: MLPLayer, cfg: Config):
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    with jax.named_scope("gate"):
        ff_gate = jax.nn.gelu(einsum("btd,df->btf", x, layer.w_gate), approximate=True).astype(cfg.dtype)
    if layer.w_up is not None:
        with jax.named_scope("up_proj"):
            ff_up = einsum("btd,df->btf", x, layer.w_up).astype(cfg.dtype)
        ff_gate = ff_gate * ff_up
    with jax.named_scope("down_proj"):
        ff_out = einsum(
            "btf,fd->btd", ff_gate, layer.w_down, out_sharding=l2p("batch", "sequence", "act_embed")
        ).astype(cfg.dtype)
    return ff_out


def forward_layer(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    idx: int,
    cfg: Config,
    cache: KVCache | None = None,
    per_layer_input: jax.Array | None = None,
    shared_kvs: tuple[jax.Array | QuantArray, jax.Array | QuantArray] | None = None,
):
    x = x.astype(cfg.dtype)

    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_pre_gamma, cfg.norm_eps)
    attn_kw = dict(idx=idx, shared_kvs=shared_kvs)
    attn_out, cache_updates = attention_block(attn_in, segment_ids, layer.attn, sin, cos, cfg, cache, **attn_kw)
    with jax.named_scope("attn_post_norm"):
        attn_out = rms_norm(attn_out, layer.attn_post_gamma, cfg.norm_eps)
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.dtype)

    residual = x
    with jax.named_scope("mlp"):
        ff_in_1 = rms_norm(x, layer.ffw_pre_gamma_mlp, cfg.norm_eps)
        ff_out_1 = mlp_block(ff_in_1, layer.mlp, cfg)
    if layer.moe is None:
        ff_combined = ff_out_1
    else:
        with jax.named_scope("moe"):
            ff_in_2 = rms_norm(x, layer.ffw_pre_gamma_moe, cfg.norm_eps)
            ff_out_2 = moe_block(ff_in_2, layer.moe, cfg)
            ff_out_2 = rms_norm(ff_out_2, layer.ffw_post_gamma_moe, cfg.norm_eps)
        ff_out_1 = rms_norm(ff_out_1, layer.ffw_post_gamma_mlp, cfg.norm_eps)
        ff_combined = ff_out_1 + ff_out_2
    ff_combined = rms_norm(ff_combined, layer.ffw_post_gamma, cfg.norm_eps)
    with jax.named_scope("residual"):
        x = residual + ff_combined.astype(cfg.dtype)

    if layer.post_mlp is not None:
        if per_layer_input is None:
            raise ValueError(f"When a layer has {layer.post_mlp=}, per_layer_input must be provided.")
        residual = x
        with jax.named_scope("per_layer_input_mlp"):
            pl_gate = einsum("btd,df->btf", x, layer.post_mlp.w_gate, out_sharding=specof(per_layer_input))
            pl_out = jax.nn.gelu(pl_gate, approximate=True).astype(cfg.dtype)
            pl_out = pl_out * per_layer_input
            pl_out = einsum("btf,fd->btd", pl_out, layer.post_mlp.w_down, out_sharding=specof(x)).astype(cfg.dtype)
        with jax.named_scope("post_mlp_post_norm"):
            pl_out = rms_norm(pl_out, layer.post_mlp_post_gamma, cfg.norm_eps)

        with jax.named_scope("residual"):
            x = residual + pl_out.astype(cfg.dtype)

    x = (x * layer.layer_scalar) if layer.layer_scalar is not None else x
    return x, cache_updates


def forward(x: jax.Array, segment_ids: jax.Array, weights: Weights, cfg: Config, cache: KVCache | None = None):
    l2p = lambda *args: logical_to_physical(args, cfg.rules)
    segment_ids = reshard(segment_ids, l2p("batch", "sequence"))
    input_ids = x
    # Embed input tokens [B, T] -> [B, T D]
    x = weights.embedding.at[x, :].get(out_sharding=l2p("batch", "sequence", "act_embed")) * (cfg.embed**0.5)

    per_layer_inputs = None
    if weights.per_layer_embed is not None:
        pli_dim = cfg.per_layer_input_dim
        pli_tok = weights.per_layer_embed.at[input_ids, :].get(out_sharding=l2p("batch", "sequence", None))
        pli_tok *= pli_dim ** 0.5
        pli_tok = pli_tok.reshape(
            (*x.shape[:2], cfg.num_layers, pli_dim), out_sharding=l2p("batch", "sequence", None, None)
        )
        pli_proj = einsum("btd,df->btf", x, weights.per_layer_proj) * (cfg.embed**-0.5)
        pli_proj = pli_proj.reshape(
            (*x.shape[:2], cfg.num_layers, pli_dim), out_sharding=l2p("batch", "sequence", None, None)
        )
        pli_proj = rms_norm(pli_proj, weights.per_layer_proj_gamma, cfg.norm_eps)
        per_layer_inputs = (pli_proj + pli_tok) * (2.0**-0.5)

    positions = segment_ids_to_positions(segment_ids)
    if is_type(cache, KVCache):
        positions = positions + cache.global_fill_len()[:, None]
    sin_local, cos_local = _generate_pos_embeddings(positions, cfg.local_head_dim, cfg, "local_attention")
    sin_global, cos_global = _generate_pos_embeddings(positions, cfg.local_head_dim, cfg, "global_attention")
    sin_local, cos_local = sin_local.astype(cfg.dtype), cos_local.astype(cfg.dtype)
    sin_global, cos_global = sin_global.astype(cfg.dtype), cos_global.astype(cfg.dtype)

    _last_idx = lambda xs, val: max([i for i, layer_type in enumerate(xs) if layer_type == val], default=-1)
    last_local_idx = _last_idx(cfg.attention_types[: -cfg.num_kv_shared_layers], "local_attention")
    last_global_idx = _last_idx(cfg.attention_types[: -cfg.num_kv_shared_layers], "global_attention")
    local_shared_kvs, global_shared_kvs = None, None

    all_kvs = []
    for idx, layer in enumerate(weights.layers):
        is_global = cfg.attention_types[idx] == "global_attention"
        sin, cos = (sin_global, cos_global) if is_global else (sin_local, cos_local)
        pli = per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
        shared_kvs = global_shared_kvs if is_global else local_shared_kvs
        x, cache_updates = forward_layer(
            x, segment_ids, layer, sin, cos, idx, cfg, cache, per_layer_input=pli, shared_kvs=shared_kvs
        )
        cache_update, cache_state = cache_updates
        global_shared_kvs = cache_state if (is_global and idx == last_global_idx) else global_shared_kvs
        local_shared_kvs = cache_state if (not is_global and idx == last_local_idx) else local_shared_kvs
        if idx < (cfg.num_layers - cfg.num_kv_shared_layers):
            if is_type(cache, KVCache):
                cache.k[idx], cache.v[idx] = cache_update.k, cache_update.v
                cache.iters[idx], cache.starts[idx] = cache_update.iter, cache_update.starts
            else:
                all_kvs.append(cache_update)

    x = rms_norm(x, weights.gamma_final, cfg.norm_eps)  # Final layer norm.
    head_weights = weights.embedding.T if cfg.tie_embed else weights.lm_head
    logits = einsum("btd,dv->btv", x, head_weights)  # Project to vocabulary size
    if cfg.final_logit_softcap is not None:
        logits = jnp.tanh(logits / cfg.final_logit_softcap) * cfg.final_logit_softcap
    return (logits, all_kvs) if not is_type(cache, KVCache) else (logits, cache)


def optimal_formats(cfg: Config):
    SDS, tree_map, bs = jax.ShapeDtypeStruct, partial(jax.tree.map, is_leaf=is_param), 16
    weights_abstract, cache_abstract = Weights.abstract(cfg), KVCache.abstract(cfg, bs)
    weights_shardings, cache_shardings = Weights.shardings(cfg), KVCache.shardings(cfg, bs)
    weights_shapes = tree_map(lambda x, s: SDS(x.shape, x.dtype, sharding=s), weights_abstract, weights_shardings)
    cache_shapes = tree_map(lambda x, s: SDS(x.shape, x.dtype, sharding=s), cache_abstract, cache_shardings)
    with jax.sharding.set_mesh(cfg.mesh):
        dummy_x = jax.ShapeDtypeStruct((bs, 1), jnp.int32, sharding=P(BATCH_AXIS_NAME, None))
        dummy_segment_ids = jax.ShapeDtypeStruct((bs, 1), jnp.int32, sharding=P(BATCH_AXIS_NAME, None))
        fn = jax.jit(
            forward, in_shardings=Format(Layout.AUTO), out_shardings=Format(Layout.AUTO), donate_argnames=("cache",)
        )
        fn_trace = fn.trace(dummy_x, dummy_segment_ids, weights_shapes, cfg=cfg, cache=cache_shapes)
        args_formats, kw_formats = fn_trace.lower().compile().input_formats
        (_, _, weights_formats), cache_formats = args_formats, kw_formats["cache"]
    weights = tree_map(lambda x, f: SDS(x.shape, x.dtype, sharding=f), weights_abstract, weights_formats)
    cache = tree_map(lambda x, f: SDS(x.shape, x.dtype, sharding=f), cache_abstract, cache_formats)
    return weights, cache


# serialization
def save_pytree(weights, path):
    flat_data = odict(("weights" + "".join(map(str, k)), v) for k, v in jax.tree.flatten_with_path(weights)[0])
    ser.save(flat_data, path)  # save a flatten-with-path to avoid custom nodes


def load_pytree(path, sharding=None):
    flat_sharding = odict(("weights" + "".join(map(str, k)), v) for k, v in jax.tree.flatten_with_path(sharding)[0])
    data = jax.tree.unflatten(jax.tree.structure(sharding), jax.tree.leaves(ser.load(path, flat_sharding)))
    return data


# Inference.
@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int):  # [bs, length] -> [bs, padded]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])], mode="constant", constant_values=pad_id)
    segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
    return chunk, segment_ids


def prefill(
    tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config, pad_id: int = PAD_ID
) -> tuple[jax.Array, jax.Array, KVCache]:
    """Samples from a prompt."""
    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)
    # Calculate the next power of 2 for padding, up to cfg.max_seq.
    assert tokens.shape[-1] <= cfg.max_seq_len
    pad_to = 2 ** math.ceil(math.log2((tokens.shape[-1])))
    prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)
    prompt = reshard(prompt, l2p("batch", "sequence"))
    assert prompt.ndim == 2

    cache_shardings = KVCache.shardings(cfg, prompt.shape[0])
    if is_type(cache, KVCache):
        uninitialized_iter = [-jnp.ones_like(iter) for iter in cache.iters]
        starts = [_count_left_padding(prompt, pad_id=pad_id) for _ in cache.starts]
        cache = dataclasses.replace(cache, starts=starts, iters=uninitialized_iter)
    else:
        kvs_layer_num = cfg.num_layers - max(cfg.num_kv_shared_layers, 0)
        cache_shardings = [(cache_shardings.k[idx], cache_shardings.v[idx]) for idx in range(kvs_layer_num)]
    logits_shardings = jax.sharding.NamedSharding(cfg.mesh, P(BATCH_AXIS_NAME, None, TENSOR_AXIS_NAME))
    logits, cache = jax.jit(forward, donate_argnums=(4,), out_shardings=(logits_shardings, cache_shardings))(
        prompt, prompt_segment_ids, weights, cfg, cache
    )
    next_tokens = jax.jit(partial(jnp.argmax, axis=-1))(logits)
    return next_tokens, logits, cache


def sample_top(key: jax.typing.ArrayLike, logits: jax.Array, *, k: int = 16, temp: float = 1.0):
    def sample_multinomial(logits):
        shard_vocab = P(BATCH_AXIS_NAME, None, TENSOR_AXIS_NAME)
        logits = reshard(logits / temp, shard_vocab)

        def sample(key, logits):
            idx = jax.lax.axis_index(TENSOR_AXIS_NAME)
            top_logits, top_tokens = jax.lax.approx_max_k(logits, k=k)
            top_tokens = top_tokens + logits.shape[-1] * idx
            top_logits = jax.lax.all_gather(top_logits, TENSOR_AXIS_NAME, axis=-1, tiled=True)
            top_tokens = jax.lax.all_gather(top_tokens, TENSOR_AXIS_NAME, axis=-1, tiled=True)
            top_logits, idx = jax.lax.top_k(top_logits, k=k)
            top_tokens = jnp.take_along_axis(top_tokens, idx, -1)

            idx = jnp.argmax(top_logits + jax.random.gumbel(key, top_logits.shape, top_logits.dtype), axis=-1)
            next_token = jnp.take_along_axis(top_tokens, idx[..., None], -1)[..., 0]
            return jax.lax.pmax(next_token, TENSOR_AXIS_NAME)  # make it invariant across TENSOR_AXIS_NAME

        return jax.shard_map(sample, in_specs=(P(), shard_vocab), out_specs=P(BATCH_AXIS_NAME, None))(key, logits)

    return jax.lax.cond(temp > 1e-3, sample_multinomial, partial(jnp.argmax, axis=-1), logits)


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(last_tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config, pad_id: int = PAD_ID, key=None):
    assert last_tokens.ndim == 2
    segment_ids = (last_tokens != pad_id).astype(jnp.int32)
    next_logits, cache = forward(last_tokens, segment_ids, weights, cfg=cfg, cache=cache)
    key = key if key is not None else random.key(cache.iters[0])  # poor man's random key
    next_tokens = sample_top(key, next_logits, k=cfg.sample_topk, temp=cfg.sample_temp)
    return reshard(next_tokens, P()), cache
