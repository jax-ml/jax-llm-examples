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

"""Minimal model definition."""

import dataclasses
import math
from dataclasses import field
from functools import partial
from typing import Callable
import tempfile
import gzip
import json
from pathlib import Path

from etils import epath

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from .decode_ragged_dot import decode_ragged_dot


AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]


# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
TENSOR_AXIS_NAME = ("y", "z")
TENSOR_ONLY_AXIS_NAME = "z"
EXPERT_AXIS_NAME = "y"


@dataclasses.dataclass
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
    head_dim: AxisName = None
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME
    act_embed: AxisName = None
    act_heads: AxisName = TENSOR_AXIS_NAME
    # attention layer
    qkv_heads: AxisName = TENSOR_AXIS_NAME
    qkv_embed: AxisName = None
    q_lora: AxisName = None
    kv_lora: AxisName = None
    o_heads: AxisName = TENSOR_AXIS_NAME
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
    moe_s_up_embed: AxisName = None
    moe_s_up_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_embed: AxisName = None
    moe_e_tp: AxisName = TENSOR_ONLY_AXIS_NAME  # moe forward function tensor parallelism
    moe_e_ep: AxisName = EXPERT_AXIS_NAME  # moe forward function expert parallelism


def logical_to_physical(logical: Axes, rules: ShardingRules) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
    # `spec` may contain tuples, flatten to check that `spec` maps each physical mesh axis to at most one logical array
    # axis.
    flat_axes = jax.tree.leaves(spec)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(f"Colliding physical axes from translating logical spec {logical} -> {spec}")
    return P(*spec)


def logical_to_sharding(logical: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules) -> jax.sharding.Sharding:
    """Returns the sharding for a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    return jax.sharding.NamedSharding(mesh, logical_to_physical(logical, rules))


def jax_pytree_struct(cls, meta_fields: tuple = ()):
    """jax.tree_util.register_dataclass wrapper that automatically infers data_fields."""
    assert not dataclasses.is_dataclass(cls)
    cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


@tree_util.register_static
@dataclasses.dataclass
class Config:
    embed: int = 7168
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    num_heads: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    vocab_size: int = 129280
    num_layers: int = 61
    max_seq_len: int = 8192
    causal: bool = True
    use_prefill_attn_kernel: bool = False
    use_decode_attn_kernel: bool = False
    weight_dtype: "jnp.dtype" = jnp.bfloat16
    # Sharding rules
    rules: ShardingRules = field(default_factory=lambda: ShardingRules())
    mesh: jax.sharding.Mesh | None = None
    # Deepseek Yarn RoPE
    rope_theta: float = 1e4
    rope_scaling_factor: float = 40.0
    rope_beta_fast: float = 32
    rope_beta_slow: float = 1
    rope_mscale: float = 1
    rope_mscale_all_dim: float = 1
    rope_original_max_position_embeddings: int = 4096
    # quantization
    quant_scale_dtype: "jnp.dtype" = jnp.float16
    quantize_moe: bool = True
    quantize_mlp: bool = False
    quantize_attn: bool = True
    quantize_cache: bool = True
    # attention
    causal: bool = True
    # MLP
    ffw_size: int = 18432
    # MoE
    first_k_dense: int = 3
    moe_gate_dtype: "jnp.dtype" = jnp.float32
    moe_ffw_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    n_shared_experts: int = 1
    psum_before_expert_reduce: bool = False
    strategy: str = "decode"


def load_tokenizer(
    tokenizer_path: Path | None = None, tokenizer_config_path: Path | None = None
) -> "PreTrainedTokenizerFast":
    from transformers import PreTrainedTokenizerFast

    if tokenizer_path is not None:
        tokenizer_path = epath.Path(tokenizer_path).expanduser().resolve()
    else:
        tokenizer_path = epath.Path(__file__).parent / "third_party" / "tokenizer" / "tokenizer.json.gz"
    if tokenizer_config_path is not None:
        tokenizer_config_path = epath.Path(tokenizer_config_path).expanduser().resolve()
    else:
        tokenizer_config_path = epath.Path(__file__).parent / "third_party" / "tokenizer" / "tokenizer_config.json"
    config = json.loads(tokenizer_config_path.expanduser().resolve().read_text())
    for k in list(config.keys()):
        v = config[k]
        if "token" in k and isinstance(v, dict):
            config[k] = v["content"]
    # return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)
    if tokenizer_path.suffix == ".gz":
        with tempfile.NamedTemporaryFile() as file:
            new_tokenizer_path = Path(file.name)
            new_tokenizer_path.write_bytes(gzip.decompress(tokenizer_path.read_bytes()))
            return PreTrainedTokenizerFast(tokenizer_file=str(new_tokenizer_path.resolve()), **config)
    else:
        return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)


# module reload friendly check for type(x) == cls
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, ArrayInfo)
which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform


@partial(jax_pytree_struct, meta_fields=("shape", "dtype", "logical_axes", "initializer", "metadata"))
class ArrayInfo:
    """Metadata describing a jax.Array, including its sharding.

    We create ArrayInfos before creating actual arrays, e.g. for model weights, so we can use the sharding and other
    metadata to set things up so we can efficiently create the actual arrays with the correct shardings.

    An alternative approach would be to use jax.eval_shape to more automatically generate the metadata we need. We use
    the ArrayInfo approach instead to decouple data and its sharding from the functions we'll apply the data to.

    """
    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    logical_axes: tuple
    initializer: Callable | None = None
    metadata: dict = field(default_factory=dict)


class _Init:
    """Base class for pytree data structures that will eventually contain jax.Arrays (e.g. layer definitions).

    Each subclass is responsible for defining abstract(), which returns an "abstract" version of the pytree containing
    ArrayInfos (i.e. metadata) instead of actual data. This class then helps generate the shardings and actual data.
    """

    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with ArrayInfos instead of jax.Arrays."""
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with Shardings instead of jax.Arrays.

        This is used to generate the Shardings needed for each array.
        """
        abstract = cls.abstract(cfg, *args, **kw)
        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key: jax.random.PRNGKey, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

        @partial(jax.jit, out_shardings=shardings)
        def _init():
            num_leaves = len(jax.tree.leaves(abstract, is_leaf=is_param))  # one new RNG key per tensor
            key_iter = iter(jax.random.split(key, num_leaves))
            return jax.tree.map(
                lambda info: info.initializer(next(key_iter), info.shape, info.dtype),
                abstract,
                is_leaf=is_param,
            )

        return _init()


@partial(jax_pytree_struct, meta_fields=("out_scaling", "scale_expand_dims"))
class QuantArray:
    quant: jax.Array | ArrayInfo
    scale: jax.Array | ArrayInfo
    out_scaling: bool = False
    scale_expand_dims: int | tuple[int, ...] = ()
    shape = property(lambda self: self.quant.shape)
    ndim = property(lambda self: self.quant.ndim)


def quantize(x: jax.Array | ArrayInfo, axis: int | tuple[int, ...], scale_dtype=jnp.float16):
    if is_type(x, QuantArray):
        raise ValueError("Attempting to quantize an already quantized QuantArray.")

    if isinstance(x, jax.Array):
        if not isinstance(axis, tuple): axis = (axis,)
        axis = tuple(z % x.ndim for z in axis)
        amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
        quant = jnp.round(x / scale).astype(jnp.int8)
        scale = scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])
        return quant, scale

    if is_type(x, ArrayInfo):
        if not isinstance(axis, tuple): axis = (axis,)
        axis = tuple(z % len(x.shape) for z in axis)
        new_shape = tuple(ax for i, ax in enumerate(x.shape) if i not in axis)
        new_logical_axes = tuple(ax for i, ax in enumerate(x.logical_axes) if i not in axis)
        quant_init = lambda key, shape, dtype=jnp.int8: random.randint(key, shape, -128, 128, dtype=dtype)
        scale_init = (
            lambda key, shape, dtype=scale_dtype: random.normal(key, shape, dtype=dtype)
            / math.sqrt(math.prod(shape))
            / 127
        )
        return (
            dataclasses.replace(x, shape=x.shape, dtype=jnp.int8, initializer=quant_init),
            ArrayInfo(new_shape, scale_dtype, new_logical_axes, scale_init, metadata={"quant_axis": axis}),
        )

    raise ValueError(f"quantize got unexpected type: {type(x)}")


def quantize_update_slice(x: QuantArray, y: jax.Array, pos: int, update_axis: int, quant_axis: int):
    assert x.quant.ndim == y.ndim
    quant_axis, update_axis = quant_axis % x.quant.ndim, update_axis % x.quant.ndim  # normalize axis numbers
    y_quant, y_scale = quantize(y, axis=quant_axis, scale_dtype=x.scale.dtype)  # quantize rhs
    scale_update_axis = [ax for ax in range(x.quant.ndim) if ax != quant_axis][update_axis]  # update axis in `scale`
    z_quant = jax.lax.dynamic_update_slice_in_dim(x.quant, y_quant.astype(x.quant.dtype), pos, axis=update_axis)
    z_scale = jax.lax.dynamic_update_slice_in_dim(x.scale, y_scale.astype(x.scale.dtype), pos, axis=scale_update_axis)
    return z_quant, z_scale


@jax_pytree_struct
class MLPLayer(_Init):
    w_gate: jax.Array | ArrayInfo | QuantArray
    w_up: jax.Array | ArrayInfo | QuantArray
    w_down: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _init = jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
        dtype = cfg.weight_dtype
        layer = MLPLayer(
            w_gate=ArrayInfo((cfg.embed, cfg.ffw_size), dtype, ("mlp_up_embed", "mlp_up_ffw"), _init),
            w_up=ArrayInfo((cfg.embed, cfg.ffw_size), dtype, ("mlp_up_embed", "mlp_up_ffw"), _init),
            w_down=ArrayInfo((cfg.ffw_size, cfg.embed), dtype, ("mlp_down_ffw", "mlp_down_embed"), _init),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MLPLayer", cfg: Config):
        if not cfg.quantize_mlp:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            w_gate=QuantArray(*quantize(layer.w_gate, 0, scale_dtype), out_scaling=True),
            w_up=QuantArray(*quantize(layer.w_up, 0, scale_dtype), out_scaling=True),
            w_down=QuantArray(*quantize(layer.w_down, 0, scale_dtype), out_scaling=True),
        )


@jax_pytree_struct
class MoELayer(_Init):
    # router
    w_router: jax.Array | ArrayInfo | QuantArray
    b_router: jax.Array | ArrayInfo | QuantArray
    # experts
    we_gate: jax.Array | ArrayInfo | QuantArray
    we_up: jax.Array | ArrayInfo | QuantArray
    we_down: jax.Array | ArrayInfo | QuantArray
    # shared experts
    ws_gate: jax.Array | ArrayInfo | QuantArray
    ws_up: jax.Array | ArrayInfo | QuantArray
    ws_down: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _einit = jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2))
        _sinit = jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
        dtype = cfg.weight_dtype
        layer = MoELayer(
            w_router=ArrayInfo(
                (cfg.embed, cfg.n_routed_experts), cfg.moe_gate_dtype, ("moe_e_up_embed", None), _sinit
            ),
            b_router=ArrayInfo(
                (cfg.n_routed_experts,), cfg.moe_gate_dtype, (None,), jax.nn.initializers.constant(0.0)
            ),
            we_gate=ArrayInfo(
                (cfg.n_routed_experts, cfg.embed, cfg.moe_ffw_size), dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_up=ArrayInfo(
                (cfg.n_routed_experts, cfg.embed, cfg.moe_ffw_size), dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_down=ArrayInfo(
                (cfg.n_routed_experts, cfg.moe_ffw_size, cfg.embed), dtype,
                ("moe_e_experts", "moe_e_down_ffw", "moe_e_down_embed"),
                _einit,
            ),
            ws_gate=ArrayInfo(
                (cfg.embed, cfg.n_shared_experts * cfg.moe_ffw_size), dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_up=ArrayInfo(
                (cfg.embed, cfg.n_shared_experts * cfg.moe_ffw_size), dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_down=ArrayInfo(
                (cfg.moe_ffw_size, cfg.n_shared_experts * cfg.embed), dtype,
                ("moe_s_down_ffw", "moe_s_down_embed"),
                _sinit,
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MoELayer", cfg: Config):
        if not cfg.quantize_moe:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            we_gate=QuantArray(*quantize(layer.we_gate, 1, scale_dtype), out_scaling=True),
            we_up=QuantArray(*quantize(layer.we_up, 1, scale_dtype), out_scaling=True),
            we_down=QuantArray(*quantize(layer.we_down, 1, scale_dtype), out_scaling=True),
            ws_gate=QuantArray(*quantize(layer.ws_gate, 0, scale_dtype), out_scaling=True),
            ws_up=QuantArray(*quantize(layer.ws_up, 0, scale_dtype), out_scaling=True),
            ws_down=QuantArray(*quantize(layer.ws_down, 0, scale_dtype), out_scaling=True),
        )


@jax_pytree_struct
class AttentionLayer(_Init):
    q_a: jax.Array | ArrayInfo | QuantArray
    q_gamma: jax.Array | ArrayInfo | QuantArray
    q_b: jax.Array | ArrayInfo | QuantArray
    kv_a: jax.Array | ArrayInfo | QuantArray
    k_pe: jax.Array | ArrayInfo | QuantArray
    kv_gamma: jax.Array | ArrayInfo | QuantArray
    k_b: jax.Array | ArrayInfo | QuantArray
    v_b: jax.Array | ArrayInfo | QuantArray
    o: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _init = lambda *out_ax: jax.nn.initializers.he_normal(in_axis=0, out_axis=out_ax)
        dtype = cfg.weight_dtype
        _ones_init = jax.nn.initializers.constant(1.0)
        q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        layer = AttentionLayer(
            q_a=ArrayInfo((cfg.embed, cfg.q_lora_rank), dtype, ("qkv_embed", "q_lora"), _init(1)),
            q_gamma=ArrayInfo((cfg.q_lora_rank,), dtype, ("q_lora",), _ones_init),
            q_b=ArrayInfo(
                (cfg.q_lora_rank, cfg.num_heads, q_head_dim), dtype,
                ("q_lora", "qkv_heads", "head_dim"),
                _init(1, 2),
            ),
            kv_a=ArrayInfo((cfg.embed, cfg.kv_lora_rank), dtype, ("qkv_embed", "kv_lora"), _init(1)),
            k_pe=ArrayInfo((cfg.embed, cfg.qk_rope_head_dim), dtype, ("qkv_embed", "head_dim"), _init(1)),
            kv_gamma=ArrayInfo((cfg.kv_lora_rank,), dtype, ("kv_lora",), _ones_init),
            k_b=ArrayInfo(
                (cfg.kv_lora_rank, cfg.num_heads, cfg.qk_nope_head_dim), dtype,
                ("kv_lora", "qkv_heads", "head_dim"),
                _init(1, 2),
            ),
            v_b=ArrayInfo(
                (cfg.kv_lora_rank, cfg.num_heads, cfg.v_head_dim),
                dtype,
                ("kv_lora", "qkv_heads", "head_dim"),
                _init(1, 2),
            ),
            o=ArrayInfo(
                (cfg.num_heads, cfg.v_head_dim, cfg.embed), dtype, ("o_heads", "head_dim", "o_embed"), _init(1, 2)
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "AttentionLayer", cfg: Config):
        if not cfg.quantize_attn:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            q_a=QuantArray(*quantize(layer.q_a, 1, scale_dtype)),
            q_b=QuantArray(*quantize(layer.q_b, (1, 2), scale_dtype)),
            kv_a=QuantArray(*quantize(layer.kv_a, 1, scale_dtype)),
            k_pe=QuantArray(*quantize(layer.k_pe, 1, scale_dtype)),
            k_b=QuantArray(*quantize(layer.k_b, (1, 2), scale_dtype)),
            v_b=QuantArray(*quantize(layer.v_b, (1, 2), scale_dtype)),
            o=QuantArray(*quantize(layer.o, (0, 1), scale_dtype), out_scaling=True),
        )


@jax_pytree_struct
class Layer(_Init):
    mlp: MLPLayer | MoELayer
    attn: AttentionLayer
    gamma_pre_attn: jax.Array | ArrayInfo
    gamma_post_attn: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config, use_moe: bool = True) -> "Layer":
        _init = jax.nn.initializers.constant(1.0)
        dtype = cfg.weight_dtype
        return Layer(
            mlp=MoELayer.abstract(cfg) if use_moe else MLPLayer.abstract(cfg),
            attn=AttentionLayer.abstract(cfg),
            gamma_pre_attn=ArrayInfo((cfg.embed,), dtype, ("act_embed",), _init),
            gamma_post_attn=ArrayInfo((cfg.embed,), dtype, ("act_embed",), _init),
        )

    @staticmethod
    def quantize(layer: "Layer", cfg: Config):
        return dataclasses.replace(
            layer, mlp=layer.mlp.quantize(layer.mlp, cfg), attn=layer.attn.quantize(layer.attn, cfg)
        )


@jax_pytree_struct
class Weights(_Init):
    layers: list[Layer]
    embedding: jax.Array | ArrayInfo
    gamma_final: jax.Array | ArrayInfo
    lm_head: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config):
        layers = [Layer.abstract(cfg, use_moe=i >= cfg.first_k_dense) for i in range(cfg.num_layers)]
        return Weights(
            layers=layers,
            embedding=ArrayInfo(
                (cfg.vocab_size, cfg.embed),
                cfg.weight_dtype,
                ("vocab_in", "vocab_out"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            gamma_final=ArrayInfo(
                (cfg.embed,),
                cfg.weight_dtype,
                ("act_embed",),
                jax.nn.initializers.constant(1.0),
            ),
            lm_head=ArrayInfo(
                (cfg.embed, cfg.vocab_size),
                cfg.weight_dtype,
                ("vocab_in", "vocab_out"),
                jax.nn.initializers.he_normal(in_axis=1, out_axis=0),
            ),
        )

    @staticmethod
    def quantize(weights: "Weights", cfg: Config):
        return dataclasses.replace(weights, layers=[layer.quantize(layer, cfg) for layer in weights.layers])


@jax_pytree_struct
class KVCache(_Init):
    k_nope: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
    k_pe: list[jax.Array]  # [batch_size, max_seq_len, qk_rope_head_dim]
    v: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
    length: jax.Array  # []  # sequences are right-aligned for slice udpate performance
    starts: jax.Array  # [batch_size]  # sequences are right-aligned, we need start indices

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int, dtype: int = jnp.bfloat16):
        _init = jax.nn.initializers.zeros
        k_nope_info = ArrayInfo(
            (batch_size, cfg.num_heads, max_seq_len, cfg.qk_nope_head_dim),
            dtype,
            ("batch", "qkv_heads", "sequence", "head_dim"),
            _init,
        )
        k_pe_info = ArrayInfo(
            (batch_size, max_seq_len, cfg.qk_rope_head_dim),
            dtype,
            ("batch", "sequence", "head_dim"),
            _init,
        )
        v_info = ArrayInfo(
            (batch_size, cfg.num_heads, max_seq_len, cfg.v_head_dim),
            dtype,
            ("batch", "qkv_heads", "sequence", "head_dim"),
            _init,
        )
        cache = KVCache(
            k_nope=[k_nope_info for _ in range(cfg.num_layers)],
            k_pe=[k_pe_info for _ in range(cfg.num_layers)],
            v=[v_info for _ in range(cfg.num_layers)],
            length=ArrayInfo((), jnp.int32, (), _init),
            starts=ArrayInfo((batch_size,), jnp.int32, ("batch",), _init),
        )
        if cfg.quantize_cache:
            scale_dtype = cfg.quant_scale_dtype
            cache.k_nope = [
                QuantArray(*quantize(k_nope, -1, scale_dtype), out_scaling=True, scale_expand_dims=-2)
                for k_nope in cache.k_nope
            ]
            cache.k_pe = [
                QuantArray(*quantize(k_pe, -1, scale_dtype), out_scaling=True, scale_expand_dims=(-2, -3))
                for k_pe in cache.k_pe
            ]
            cache.v = [
                QuantArray(*quantize(v, -1, scale_dtype), out_scaling=False, scale_expand_dims=-2)
                for v in cache.v
            ]
        return cache

    @property
    def time_axis(self) -> int:
        return 2


def _count_length_from_left(segment_ids):
    """Count the length of a sequence as len(seq) - len(seq's right padding)."""
    return jnp.sum(jnp.cumsum(jnp.flip(segment_ids != 0, axis=-1), axis=-1) > 0, axis=-1)


def einsum(subscripts: str, lhs: jax.Array, rhs: jax.Array | QuantArray):
    """jnp.einsum wrapper that handles regular arrays and QuantArrays"""
    if is_type(rhs, QuantArray):
        scale = jnp.expand_dims(rhs.scale, rhs.scale_expand_dims)
        if rhs.out_scaling:
            return jnp.einsum(subscripts, lhs, rhs.quant) * scale
        else:
            return jnp.einsum(subscripts, lhs * scale, rhs.quant)
    else:
        return jnp.einsum(subscripts, lhs, rhs)


def update_slice(x: jax.Array | QuantArray, y: jax.Array, pos: int, update_axis: int, quant_axis: int = -1):
    """dynamic_update_slice wrapper that handles regular arrays and QuantArrays"""
    if is_type(x, QuantArray):
        new_quant, new_scale = quantize_update_slice(x, y, pos, update_axis=update_axis, quant_axis=quant_axis)
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        return jax.lax.dynamic_update_slice_in_dim(x, y.astype(x.dtype), pos, axis=update_axis)

def logical_sharding_constraint(x: jax.Array | QuantArray, logical_axes: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules):
    """Generate a sharding constraint for a regular or QuantArray given its logical axes."""
    sharding = logical_to_sharding(logical_axes, mesh, rules)
    if is_type(x, QuantArray):
        return dataclasses.replace(x, quant=jax.lax.with_sharding_constraint(x.quant, sharding))
    else:
        return jax.lax.with_sharding_constraint(x, sharding)


def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim) - min) / (max - min)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


def generate_pos_embeddings(positions, head_dim, cfg: Config):
    fractions = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    freq_extra = 1.0 / (cfg.rope_theta**fractions)
    freq_inter = 1.0 / (cfg.rope_scaling_factor * cfg.rope_theta**fractions)

    low, high = _yarn_find_correction_range(
        cfg.rope_beta_fast, cfg.rope_beta_slow, head_dim, cfg.rope_theta, cfg.rope_original_max_position_embeddings
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, head_dim // 2).astype(jnp.float32)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    freqs = jnp.einsum("...T,k->...Tk", positions, inv_freq, precision=jax.lax.Precision.HIGHEST)
    _yarn_get_mscale = lambda scale, mscale: jnp.where(scale <= 1, 1.0, 0.1 * mscale * jnp.log(scale) + 1.0)
    _mscale = _yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale) / _yarn_get_mscale(
        cfg.rope_scaling_factor, cfg.rope_mscale_all_dim
    )
    sin, cos = jnp.sin(freqs) * _mscale, jnp.cos(freqs) * _mscale
    return sin, cos


def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4
    assert sin.ndim == 3 and cos.ndim == 3
    sin, cos = (
        sin[:, None, :, :],
        cos[:, None, :, :],
    )  # [B, T, head_dim] -> [B, h, T, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal: bool):
    # [B, t, T]
    segment_mask = q_segment_ids[:, :, None] == k_segment_ids[:, None, :]
    # [B, t, T] -> [B, 1, t, T]
    segment_mask = segment_mask[:, None, :, :]

    if causal:
        # [b, h, t, T]
        qk = (1, 1, q_len, k_len)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2)
        k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3)
        q_positions = q_iota + q_offset[:, None, None, None]
        causal_mask = q_positions >= k_iota
        combined_mask = jnp.logical_and(segment_mask, causal_mask)
        return combined_mask
    else:
        return segment_mask


def _get_attn_scale(q_head_dim: int, cfg: Config):
    scale = q_head_dim**-0.5
    if cfg.rope_scaling_factor <= 1.0:
        _yarn_mscale = 1.0
    else:
        _yarn_mscale = 0.1 * cfg.rope_mscale_all_dim * math.log(cfg.rope_scaling_factor) + 1.0
    return scale * _yarn_mscale**2


def attention(
    q_nope: jax.Array,
    q_pe: jax.Array,
    k_nope: jax.Array | tuple[jax.Array, jax.Array],
    k_pe: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    k_segment_ids: jax.Array,
    q_offset: jax.Array,
    cfg: Config,
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
    cfg: Configuration object

    Returns:
    Attention output of shape (batch_size, num_heads, q_len, head_dim)
    """
    scale = _get_attn_scale(q_nope.shape[-1] + q_pe.shape[-1], cfg)

    # grouped-query attention
    b, h, t, d = q_nope.shape
    _, h, T, _ = k_nope.shape

    qk = einsum("bhtd,bhTd->bhtT", q_nope, k_nope)
    qk = qk + einsum("bhtd,bTd->bhtT", q_pe, k_pe)
    qk = qk * scale  # [b, h, t, T]

    mask = make_attention_mask(t, T, q_segment_ids, k_segment_ids, q_offset, cfg.causal)
    qk = jnp.where(mask, qk, -1e30)  # Apply the combined mask
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)

    # grouped-query attention
    attn_ = attn.reshape((b, h, t, T))
    qkv = einsum("bhtT,bhTd->bhtd", attn_, v).astype(cfg.weight_dtype)
    return qkv.reshape((b, h, t, v.shape[-1]))


def attention_kernel(q, k, v, q_segment_ids, kv_segment_ids, q_offset, starts, lengths, cfg: Config):
    """Flash attention kernel!"""
    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)

    # handle grouped query attention
    assert q.shape[-3] % k.shape[-3] == 0
    scale = _get_attn_scale(q.shape[-1], cfg)

    l2p = lambda *xs: logical_to_physical(xs, cfg.rules)
    in_specs = (
        l2p("batch", "act_heads", "sequence", "head_dim"),
        l2p("batch", "act_heads", "sequence", "head_dim"),
        l2p("batch", "act_heads", "sequence", "head_dim"),
        l2p("batch", "sequence"),
        l2p("batch", "sequence"),
        l2p("batch") if starts is not None else None,
        l2p("batch") if lengths is not None else None,
        l2p("batch", "act_heads", "sequence") if k_scale is not None else None,
        l2p("batch", "act_heads", "sequence") if v_scale is not None else None,
    )
    out_specs = l2p("batch", "act_heads", "sequence", "head_dim")

    @partial(shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    def _f(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale):
        q_org_shape = q.shape
        kv_repeats = q.shape[-3] // k.shape[-3]
        q = q.reshape(q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1]))

        if q.shape[-2] != 1:
            mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((q.shape[-2], k.shape[-2])) for _ in range(q.shape[-3])])
            block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 1024)
            block_sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
            attn_fn = splash.make_splash_mqa_single_device(mask=mask, block_sizes=block_sizes)
            attn_fn = jax.vmap(jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))

            segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
            if k_scale is not None:
                k = (k * k_scale[..., None]).astype(jnp.bfloat16)
            if v_scale is not None:
                v = (v * v_scale[..., None]).astype(jnp.bfloat16)
            ret = attn_fn(q * scale, k, v, segment_ids)
        else:
            raise NotImplementedError
            assert q.shape[-2] == 1, "This is a decode kernel, q.shape[-2] must be 1"
            q = q[..., 0, :]
            in_axes = (1, 1, 1, None, None)
            in_axes += ((None if k_scale is None else 1),)
            in_axes += ((None if v_scale is None else 1),)
            hyperparams = dict(scale=scale, block_kv=min(k.shape[-2], 8192))
            ret = jax.vmap(partial(ragged_attention.ragged_decode_fwd, **hyperparams), in_axes=in_axes, out_axes=1)(
                q, k, v, starts, lengths, k_scale, v_scale
            )
        return ret.reshape(q_org_shape[:-1] + (v.shape[-1],))

    lengths = jnp.broadcast_to(lengths, starts.shape)
    return _f(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale).astype(jnp.bfloat16)


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    """Apply RMS normalization."""
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    return jnp.astype(gamma * x / rms, jnp.bfloat16)


def mla_attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    attn_layer: AttentionLayer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    idx: int = 0,
) -> jax.Array:
    dtype = cfg.weight_dtype
    with jax.named_scope("q_embed"):
        q_lora = einsum("btd,dr->btr", x, attn_layer.q_a).astype(dtype)
        q_lora = rms_norm(q_lora, attn_layer.q_gamma).astype(dtype)
        q = einsum("btr,rhq->bhtq", q_lora, attn_layer.q_b).astype(dtype)
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_pe = apply_rotary_embedding(q[..., cfg.qk_nope_head_dim :], sin, cos).astype(dtype)

    with jax.named_scope("kv_compressed_embed"):
        kv_compressed = einsum("btd,dr->btr", x, attn_layer.kv_a).astype(dtype)
        kv_compressed = rms_norm(kv_compressed, attn_layer.kv_gamma).astype(dtype)
        k_pe = einsum("btd,dq->btq", x, attn_layer.k_pe)
        k_pe = apply_rotary_embedding(k_pe[..., None, :, :], sin, cos)[..., 0, :, :].astype(dtype)

    with jax.named_scope("kv_embed"):
        k_nope = einsum("btr,rhq->bhtq", kv_compressed, attn_layer.k_b)
        v = einsum("btr,rhv->bhtv", kv_compressed, attn_layer.v_b)

    with jax.named_scope("full_cache_update"):
        if cache is not None:
            k_nope = update_slice(cache.k_nope[idx], k_nope, cache.length, update_axis=cache.time_axis)
            k_pe = update_slice(cache.k_pe[idx], k_pe, cache.length, update_axis=cache.time_axis - 1)
            v = update_slice(cache.v[idx], v, cache.length, update_axis=cache.time_axis)
            cache_updates = (k_nope, k_pe, v)
        else:
            cache_updates = None

    # constrain the sharding of intermediates for the attention operation
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    spec = ("batch", "act_heads", "sequence", "head_dim")
    q_nope, q_pe = lsc(q_nope, spec), lsc(q_pe, spec)
    k_nope, k_pe, v = lsc(k_nope, spec), lsc(k_pe, ("batch", "sequence", "head_dim")), lsc(v, spec)

    # create position embeddings
    if cache is not None:
        time_indices = jnp.arange(0, v.shape[-2])[None, :]  # [1, T]
        q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
        incremental_position = jnp.max(_count_length_from_left(segment_ids))
        # i.e. valid below where we've written things [B, T]
        k_segment_ids = (
            (time_indices >= cache.starts[:, None]) & (time_indices < (cache.length + incremental_position))
        ).astype(jnp.int32)

        q_offset = cache.length[None]
        starts, lengths = cache.starts, (cache.length + incremental_position)[None]
    else:
        q_segment_ids, k_segment_ids = segment_ids, segment_ids
        q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
        starts = jnp.sum(jnp.cumsum(k_segment_ids != 0, axis=-1) == 0, axis=-1)
        lengths = _count_length_from_left(k_segment_ids)

    # Compute attention
    with jax.named_scope("attention"):
        if (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (cfg.use_decode_attn_kernel and q.shape[-2] == 1):
            raise NotImplementedError
            attn_out = attention_kernel(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                v,
                q_segment_ids,
                k_segment_ids,
                q_offset,
                starts=starts,
                lengths=lengths,
                cfg=cfg,
            )
        else:
            attn_out = attention(q_nope, q_pe, k_nope, k_pe, v, q_segment_ids, k_segment_ids, q_offset, cfg)

    with jax.named_scope("o_proj"):
        attn_out = einsum("bhtv,hvd->btd", attn_out, attn_layer.o)
    attn_out = lsc(attn_out.astype(cfg.weight_dtype), ("batch", "sequence", "act_embed"))
    return attn_out, cache_updates


@partial(jax.jit, static_argnames=("replicated_routing",))
def _route_tokens_to_moe_experts(
    x: jax.Array, weight: jax.Array, bias: jax.Array, replicated_routing: bool, cfg: Config
):
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    x_shape = x.shape
    x = x.reshape((-1, x.shape[-1]))
    if replicated_routing:  # not distributing the routing work avoids communication for small batches
        x = lsc(x, (None, None))
    else:
        x = jax.lax.with_sharding_constraint(x, NamedSharding(cfg.mesh, P(TENSOR_AXIS_NAME, None)))
    weight, bias = lsc(weight, (None, None)), lsc(bias, (None,))

    scores = jax.nn.sigmoid(jnp.einsum("Sk,kj->Sj", x, weight).astype(cfg.moe_gate_dtype))
    scores_with_bias = scores + bias
    group_scores = jnp.sum(
        jax.lax.top_k(scores_with_bias.reshape(scores.shape[:-1] + (cfg.n_group, -1)), 2)[0], axis=-1
    )
    group_idx = jax.lax.top_k(group_scores, cfg.topk_group)[1]
    mask = jnp.any(jnp.arange(cfg.n_group)[:, None] == group_idx[..., None, :], axis=-1)
    mask = jnp.repeat(mask, scores.shape[-1] // mask.shape[-1], -1)
    masked_scores = jnp.where(mask, scores_with_bias, 0.0)
    topk_idx = jax.lax.top_k(masked_scores, cfg.num_experts_per_tok)[1]
    topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1).astype(cfg.moe_gate_dtype)
    topk_weights = cfg.routed_scaling_factor * topk_weights / (jnp.sum(topk_weights, axis=-1)[..., None] + 1e-20)

    topk_weights = lsc(topk_weights, (None, None)).reshape(x_shape[:-1] + (cfg.num_experts_per_tok,))
    topk_idx = lsc(topk_idx, (None, None)).reshape(x_shape[:-1] + (cfg.num_experts_per_tok,))
    return topk_weights, topk_idx


def _moe_gmm(lhs, rhs, group_sizes, topk_idx, cfg: Config):
    assert lhs.ndim == 2 and rhs.ndim == 3, f"{lhs.ndim=} != 2 and {rhs.ndim=} != 3"
    interpret = which_platform(cfg) != "tpu"
    group_sizes = group_sizes.astype(jnp.int32)
    if lhs.shape[0] <= 1024:
        with jax.named_scope("jax.lax.ragged_dot"):
            if is_type(rhs, QuantArray):
                assert rhs.scale.ndim == 2 and rhs.scale.shape == (rhs.quant.shape[0], rhs.quant.shape[2])
                scale = jnp.take_along_axis(rhs.scale, topk_idx[:, None], axis=-2)
                ret = decode_ragged_dot(lhs, rhs.quant, group_sizes, block_g=4, block_n=1024, interpret=interpret)
                ret = ret * scale
            else:
                ret = decode_ragged_dot(lhs, rhs, group_sizes, block_g=4, block_n=1024, interpret=interpret)
    else:
        with jax.named_scope("jax.lax.ragged_dot"):
            if is_type(rhs, QuantArray):
                assert rhs.scale.ndim == 2 and rhs.scale.shape == (rhs.quant.shape[0], rhs.quant.shape[2])
                scale = jnp.take_along_axis(rhs.scale, topk_idx[:, None], axis=-2)
                ret = jax.lax.ragged_dot(lhs, rhs.quant, group_sizes) * scale
            else:
                ret = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    return ret.astype(cfg.weight_dtype)


def moe_block_ep(x: jax.Array, layer: MoELayer, cfg: Config):
    assert x.ndim == 3
    l2p = lambda *axes: logical_to_physical(axes, cfg.rules)
    _psc = lambda z, spec: jax.lax.with_sharding_constraint(z, NamedSharding(cfg.mesh, P(*spec)))
    _qpsc = lambda z, spec: dataclasses.replace(z, quant=_psc(z.quant, spec.quant), scale=_psc(z.scale, spec.scale))
    psc = lambda z, spec: _qpsc(z, spec) if is_type(z, QuantArray) else _psc(z, spec)

    replicated_routing = x.shape[-2] == 1  # we're decoding
    topk_weights, topk_idx = _route_tokens_to_moe_experts(x, layer.w_router, layer.b_router, replicated_routing, cfg)
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

    is_embedding_sharded = not (l2p("act_embed")[0] is None)
    if is_embedding_sharded:  # activations are sharded
        out_spec = P(*(out_spec[:-1] + (tensor_axname,)))  # override last axis name
    if cfg.strategy == "prefill":
        out_spec = P(*(out_spec[:-1] + (tensor_axname,)))  # override last axis name

    expert_count = cfg.mesh.axis_sizes[cfg.mesh.axis_names.index(expert_axname)] if expert_axname is not None else 1
    tensor_count = cfg.mesh.axis_sizes[cfg.mesh.axis_names.index(tensor_axname)] if tensor_axname is not None else 1
    assert cfg.n_routed_experts % expert_count == 0
    expert_size = cfg.n_routed_experts // expert_count

    @partial(shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_spec, check_rep=False)
    def _expert_fn(x, we_gate, we_up, we_down, topk_weights, topk_idx):
        (b, s, d), e = x.shape, cfg.num_experts_per_tok
        expert_idx = jax.lax.axis_index(expert_axname) if expert_axname is not None else 0
        tensor_idx = jax.lax.axis_index(tensor_axname) if tensor_axname is not None else 0
        topk_idx_ = topk_idx.reshape(-1)
        valid_group_mask_ = (topk_idx_ >= expert_size * expert_idx) & (topk_idx_ < expert_size * (expert_idx + 1))
        expert_mapped_topk_idx_ = jnp.where(valid_group_mask_, topk_idx_ - expert_idx * expert_size, 2**30)

        sort_idx_ = jnp.argsort(expert_mapped_topk_idx_, axis=-1)  # [b * s * e]
        isort_idx_ = jnp.argsort(sort_idx_)

        if cfg.strategy == "prefill":
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

        group_sizes = jnp.bincount(topk_idx_sort_, length=cfg.n_routed_experts)
        group_sizes_shard = jax.lax.dynamic_slice_in_dim(group_sizes, expert_idx * expert_size, expert_size, 0)

        with jax.named_scope("we_gate"):
            ff_gate = _moe_gmm(x_repeat_sort_, we_gate, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg)
            ff_gate = jax.nn.silu(ff_gate)
            ff_gate = jnp.where(valid_group_mask_sort_[..., None], ff_gate, 0)
        with jax.named_scope("we_up"):
            ff_up = _moe_gmm(x_repeat_sort_, we_up, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg)
        ff_gate_up = jnp.where(valid_group_mask_sort_[..., None], ff_gate * ff_up, 0)
        with jax.named_scope("we_down"):
            ff_out = _moe_gmm(ff_gate_up, we_down, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg)
            ff_out = jnp.where(valid_group_mask_sort_[..., None], ff_out, 0)  # expensive

        if cfg.strategy == "prefill":
            rs_shape = math.ceil((ff_out.shape[-1] // tensor_count) / 256) * 256 * tensor_count
            pad_size = rs_shape - ff_out.shape[-1]
            ff_out = jnp.pad(ff_out, ((0, 0), (0, pad_size)))
            ff_out = jax.lax.psum_scatter(ff_out, axis_name=tensor_axname, scatter_dimension=1, tiled=True)

        if cfg.strategy == "prefill":
            with jax.named_scope("expert_weighting"):
                ff_out = ff_out * topk_weights.reshape(-1)[sort_idx_][..., None]
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
                ff_out_expert = ff_out_expert.astype(cfg.weight_dtype)
        else:
            with jax.named_scope("unpermute"):
                ff_out = jnp.take_along_axis(ff_out, isort_idx_[..., None], axis=-2)
            with jax.named_scope("expert_weighting"):
                ff_out_expert = jnp.einsum(
                    "Ted,Te->Td", ff_out.reshape((b * s, e, -1)), topk_weights.reshape((b * s, -1))
                )
                ff_out_expert = ff_out_expert.astype(cfg.weight_dtype)

        with jax.named_scope("experts_collective"):
            if cfg.strategy == "prefill":
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
        ff_out_expert = _expert_fn(x_, we_gate, we_up, we_down, topk_weights, topk_idx)[..., :x.shape[-1]]
    with jax.named_scope("moe_shared_expert"):
        ff_out_shared = mlp_block(x, MLPLayer(layer.ws_gate, layer.ws_up, layer.ws_down), cfg)[..., :x.shape[-1]]
    return psc(ff_out_expert + ff_out_shared, l2p("batch", "sequence", "act_embed"))


def mlp_block(x: jax.Array, layer: MLPLayer, cfg: Config):
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    dtype = cfg.weight_dtype
    with jax.named_scope("gate"):
        ff_gate = jax.nn.silu(einsum("btd,df->btf", x, layer.w_gate)).astype(dtype)
    with jax.named_scope("up_proj"):
        ff_up = einsum("btd,df->btf", x, layer.w_up).astype(dtype)
    with jax.named_scope("down_proj"):
        ff_out = einsum("btf,fd->btd", ff_gate * ff_up, layer.w_down).astype(dtype)
    return lsc(ff_out, ("batch", "sequence", "act_embed"))


def forward_layer(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    idx: int,
    cfg: Config,
    cache: KVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = x.astype(cfg.weight_dtype)
    x = jax.lax.with_sharding_constraint(
        x, logical_to_sharding(("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules)
    )

    # Attention block
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.gamma_pre_attn)
    attn_out, cache_updates = mla_attention_block(attn_in, segment_ids, layer.attn, sin, cos, cfg, cache, idx)
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.weight_dtype)

    # FFN block
    with jax.named_scope("attn_post_norm"):
        ff_in = rms_norm(x, layer.gamma_post_attn)
    with jax.named_scope("ffn"):
        ff_out = (mlp_block if is_type(layer.mlp, MLPLayer) else moe_block_ep)(ff_in, layer.mlp, cfg)
    with jax.named_scope("residual"):
        x = x + ff_out.astype(cfg.weight_dtype)

    return x, cache_updates


def forward(x: jax.Array, segment_ids: jax.Array, weights: Weights, cfg: Config, cache: KVCache | None = None):
    with jax.named_scope("vocab_in_proj"):
        # Embed input tokens [B, T] -> [B, T D]
        x = jax.lax.with_sharding_constraint(
            weights.embedding[x, :], logical_to_sharding(("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules)
        )
    batch = x.shape[0]
    positions = segment_ids_to_positions(segment_ids)
    # Apply rotary embeddings: [B, T, head_dim]
    if cache is not None:
        # For inference with cache, we need to index the positional embeddings
        start_indices = jnp.where(cache.length != 0, cache.length - cache.starts, 0)
    else:
        start_indices = jnp.zeros((batch,), dtype=jnp.int32)
    # NOTE: At inference time this only works for UNPACKED sequences.
    positions = start_indices[:, None] + positions
    # [B, T, head_dim]
    sin, cos = generate_pos_embeddings(positions, cfg.qk_rope_head_dim, cfg)
    sin, cos = sin.astype(cfg.weight_dtype), cos.astype(cfg.weight_dtype)

    for idx, layer in enumerate(weights.layers):
        x, cache_updates = forward_layer(x, segment_ids, layer, sin, cos, idx, cfg, cache)
        if cache is not None:
            cache.k_nope[idx], cache.k_pe[idx], cache.v[idx] = cache_updates

    # Final layer norm.
    x = rms_norm(x, weights.gamma_final)
    # Project to vocabulary size
    with jax.named_scope("vocab_out_proj"):
        x = jax.lax.with_sharding_constraint(x, logical_to_sharding(("batch", "sequence", None), cfg.mesh, cfg.rules))
        logits = jnp.einsum("btd,dv->btv", x, weights.lm_head)
    if cache is not None:
        # Increment the sequence length in the cache (i.e. non padding tokens) [B, T] -> [B,]
        cache = dataclasses.replace(cache, length=cache.length + jnp.max(_count_length_from_left(segment_ids)))
        return logits, cache
    return logits


# serialization
def save_pytree(data, path):
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(epath.Path(path), data, ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=1024 * 1024 * 100))


def load_pytree(path, sharding=None):
    import orbax.checkpoint as ocp

    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            epath.Path(path), args=ocp.args.PyTreeRestore(item=item, transforms=transforms, restore_args=restore_args)
        )


# Inference.
@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int):
    # [bs, length] -> [bs, padded]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])])
    segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
    return chunk, segment_ids


def prefill(tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config, pad_id: int = 0, batch_idx: int = 0):
    """Samples from a prompt."""

    # Calculate the next power of 2 for padding, up to cfg.max_seq.
    assert tokens.shape[-1] <= cfg.max_seq_len
    pad_to = 2 ** math.ceil(math.log2((tokens.shape[-1])))
    prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)
    assert prompt.ndim == 2

    if cache is not None:
        cache_shardings = type(cache).shardings(cfg, cfg.mesh, cfg.rules)
        cache = dataclasses.replace(
            cache, length=jnp.zeros_like(cache.length), starts=jnp.sum(jnp.cumsum(tokens != pad_id, axis=-1) == 0, -1)
        )
    else:
        cache_shardings = None
    logits_shardings = logical_to_sharding(("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules)
    logits, cache = jax.jit(forward, donate_argnums=(4,), out_shardings=(logits_shardings, cache_shardings))(
        prompt, prompt_segment_ids, weights, cfg, cache
    )
    max_tokens = jax.jit(partial(jnp.argmax, axis=-1), out_shardings=NamedSharding(cfg.mesh, P()))(logits)
    return max_tokens, logits, cache


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(last_tokens: jax.Array, weights, cache: KVCache, cfg: Config):
    assert last_tokens.ndim == 2
    segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
    next_logits, cache = forward(last_tokens, segment_ids, weights, cfg, cache)
    next_tokens = jnp.argmax(next_logits, -1)
    next_tokens = jax.lax.with_sharding_constraint(next_tokens, NamedSharding(cfg.mesh, P()))
    return next_tokens, cache
