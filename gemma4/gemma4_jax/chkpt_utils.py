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

import os
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import dataclasses
from typing import Callable

import jax
from jax import numpy as jnp
from jax.sharding import AxisType
import torch
from tqdm import tqdm

from . import model as g4jax


def quantize_model(ckpt_path: Path, quant_ckpt_path: Path):
  ckpt_path, quant_ckpt_path = Path(ckpt_path).expanduser(), Path(quant_ckpt_path).expanduser()
  assert ckpt_path.is_dir()
  cfg = g4jax.load_config(ckpt_path / "config.json")
  mesh = jax.make_mesh((1, jax.device_count(), 1), ("x", "y", "z"), axis_types=3 * (AxisType.Explicit,))
  cfg = dataclasses.replace(cfg, mesh=mesh, quant_moe=True, quant_mlp=True, quant_attn=False)

  print("Loading weights...")
  weights = g4jax.load_pytree(
    ckpt_path, g4jax.Weights.shardings(dataclasses.replace(cfg, quant_moe=False, quant_mlp=False, quant_attn=False))
  )

  print("Converting weights...")
  quant_layers = [g4jax.Layer.quantize(layer, cfg) for layer in tqdm(weights.layers, total=len(weights.layers))]
  quant_weights = dataclasses.replace(weights, layers=quant_layers)

  print("Saving weights...")
  if quant_ckpt_path.exists():
    shutil.rmtree(quant_ckpt_path)
  quant_ckpt_path.parent.mkdir(exist_ok=True)
  g4jax.save_pytree(quant_weights, quant_ckpt_path)

  additional_files = [
    f for f in ckpt_path.glob("*")
    if f.is_file() and f.suffix in (".json", ".jinja") and re.search("pytreedef", f.name) is None
  ]
  for additional_file in additional_files:
      shutil.copyfile(additional_file, quant_ckpt_path / additional_file.name)


is_leaf = lambda x: isinstance(x, g4jax.ArrayInfo)
j2t = lambda x: torch.from_dlpack(x)


def t2j(x):
  prev_level, os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", None), "9"
  try:
    return jnp.from_dlpack(x.detach().contiguous())
  finally:
    if prev_level is not None:
      os.environ["TF_CPP_MIN_LOG_LEVEL"] = prev_level


def _index_to_str(x):
  """Convert objects from jax.tree.flatten_with_path to dot separated strings."""
  for field in ["name", "idx", "key"]:
    if hasattr(x, field):
      return str(getattr(x, field))
  raise ValueError


def _get_attn_weight_correct_head_dim_and_kv_heads(value_size: int, key: str, cfg: g4jax.Config):
  can_distinguish_global = (cfg.global_head_dim * cfg.global_kv_heads) != (cfg.local_head_dim * cfg.local_kv_heads)
  if can_distinguish_global:
    is_local = value_size == cfg.local_kv_heads * cfg.local_head_dim
  else:
    if re.search(r"global", key) is None and re.search(r"local", key) is None:
      raise ValueError(f"Cannot distinguish global from local for {key}")
    is_local = re.search(r"local", key) is not None
  return (cfg.local_head_dim, cfg.local_kv_heads) if is_local else (cfg.global_head_dim, cfg.global_kv_heads)


def _get_attn_weight_correct_head_dim_for_qo_proj(value_size: int, key: str, cfg: g4jax.Config):
  can_distinguish_global = (cfg.global_head_dim * cfg.q_heads) != (cfg.local_head_dim * cfg.q_heads)
  if can_distinguish_global:
    is_local = value_size == cfg.q_heads * cfg.local_head_dim
  else:
    if re.search(r"global", key) is None and re.search(r"local", key) is None:
      raise ValueError(f"Cannot distinguish global from local for {key}")
    is_local = re.search(r"local", key) is not None
  return cfg.local_head_dim if is_local else cfg.global_head_dim


def convert_weight(key: str, value: torch.Tensor, cfg: g4jax.Config):
  value = value.detach()
  # attention ################################################################
  if re.search(r"q_proj\.weight", key) is not None:
    head_dim = _get_attn_weight_correct_head_dim_for_qo_proj(value.shape[0], key, cfg)
    assert value.shape == (cfg.q_heads * head_dim, cfg.embed)
    return t2j(value.T.reshape((cfg.embed, cfg.q_heads, head_dim)))
  elif re.search(r"[kv]_proj\.weight", key) is not None:
    head_dim, kv_heads = _get_attn_weight_correct_head_dim_and_kv_heads(value.shape[0], key, cfg)
    assert value.shape == (kv_heads * head_dim, cfg.embed)
    return t2j(value.T.reshape((cfg.embed, kv_heads, head_dim)))
  elif re.search(r"o_proj\.weight", key) is not None:
    head_dim = _get_attn_weight_correct_head_dim_for_qo_proj(value.shape[1], key, cfg)
    assert value.shape == (cfg.embed, cfg.q_heads * head_dim)
    return t2j(value.T.reshape((cfg.q_heads, head_dim, cfg.embed)))
  # Gemma4 MoE router ########################################################
  elif re.search(r"router\.proj\.weight", key) is not None:
    assert value.shape == (cfg.moe_num_experts, cfg.embed)
    return t2j(value.T)
  elif re.search(r"router\.scale", key) is not None:
    assert value.shape == (cfg.embed,)
    return t2j(value)
  # Gemma4 MoE stacked expert nn.Parameter weights ###########################
  elif re.search(r"(moe|experts)\.(gate|up)_proj$", key) is not None:
    # assert value.shape == (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size)
    assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed)
    return t2j(value.mT)
  elif re.search(r"(moe|experts)\.down_proj$", key) is not None:
    # assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed)
    assert value.shape == (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size)
    return t2j(value.mT)
  # elif re.search(r"moe\.per_expert_scale$", key) is not None:
  elif re.search(r"router\.per_expert_scale$", key) is not None:
    assert value.shape == (cfg.moe_num_experts,)
    return t2j(value)
  # MLP ######################################################################
  elif re.search(r"down_proj\.weight", key) is not None:
    assert value.shape[0] == cfg.embed
    return t2j(value.T)
  elif re.search(r"(gate|up)_proj\.weight", key) is not None:
    assert value.shape[1] == cfg.embed
    return t2j(value.T)
  # per-layer input gate/projection ##########################################
  elif re.search(r"per_layer_input_gate\.weight", key) is not None:
    assert value.shape == (cfg.per_layer_input_dim, cfg.embed), f"{value.shape=} != {(cfg.per_layer_input_dim, cfg.embed)}"
    return t2j(value.T)
  elif re.search(r"per_layer_projection\.weight", key) is not None:
    assert value.shape == (cfg.embed, cfg.per_layer_input_dim), f"{value.shape=} != {(cfg.embed, cfg.per_layer_input_dim)}"
    return t2j(value.T)
  # MoE router scales ########################################################
  elif re.search(r"router\.scale$", key) is not None:
    assert value.shape == (cfg.embed,)
    return t2j(value)
  elif re.search(r"router\.proj\.weight$", key) is not None:
    assert value.shape == (cfg.moe_num_experts, cfg.embed)
    return t2j(value.T)
  # per-layer input model-level weights ########################################
  elif re.search(r"embed_tokens_per_layer\.weight", key) is not None:
    pli_total = cfg.num_layers * cfg.per_layer_input_dim
    assert value.shape == (cfg.vocab_size_per_layer_input, pli_total), f"{value.shape=} != {(cfg.vocab_size_per_layer_input, pli_total)}"
    return t2j(value)
  elif re.search(r"per_layer_model_projection\.weight", key) is not None:
    pli_total = cfg.num_layers * cfg.per_layer_input_dim
    assert value.shape == (pli_total, cfg.embed), f"{value.shape=} != {(pli_total, cfg.embed)}"
    return t2j(value.T)
  elif re.search(r"per_layer_projection_norm\.weight", key) is not None:
    assert value.shape == (cfg.per_layer_input_dim,), f"{value.shape=} != {(cfg.per_layer_input_dim,)}"
    return t2j(value)
  # misc #####################################################################
  elif re.search(r"embed_tokens\.weight$", key) is not None:
    assert value.shape == (cfg.vocab_size, cfg.embed)
    return t2j(value)
  elif re.search(r"lm_head", key) is not None:
    assert value.shape == (cfg.vocab_size, cfg.embed)
    return t2j(value.T)
  elif re.search(r"(q|k)_norm\.weight", key) is not None:
    return t2j(value)
  elif re.search(r"per_dim_scale", key) is not None:
    return t2j(value)
  elif re.search(r"layer_scalar", key) is not None:
    return t2j(value.squeeze())
  elif re.search(r"norm\.weight", key) is not None:
    return t2j(value)
  elif re.search(r"layernorm", key) is not None:
    return t2j(value)
  else:
    raise ValueError(f"Unknown weight {key = }")


_HF_KEY_MAPPING = {
  r".*model\.embed_tokens\.weight": "embedding",
  # attention projection weights
  r".*model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"layers.\1.attn.q",
  r".*model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers.\1.attn.k",
  r".*model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"layers.\1.attn.v",
  r".*model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"layers.\1.attn.o",
  # attention norms
  r".*model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"layers.\1.attn.q_gamma",
  r".*model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"layers.\1.attn.k_gamma",
  # layer norms (pre/post attention)
  r".*model\.layers\.([0-9]+)\.input_layernorm\.weight": r"layers.\1.attn_pre_gamma",
  r".*model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"layers.\1.attn_post_gamma",
  # Gemma4-style moe router (maps to moe field = MoELayer)
  r".*model\.layers\.([0-9]+)\.router\.proj\.weight": r"layers.\1.moe.w_router",
  r".*model\.layers\.([0-9]+)\.router\.scale": r"layers.\1.moe.w_router_scale",
  r".*model\.layers\.([0-9]+)\.router\.per_expert_scale": r"layers.\1.moe.per_expert_scale",
  # Gemma4-style moe stacked expert weights (maps to moe field = MoELayer)
  r".*model\.layers\.([0-9]+)\.(moe|experts)\.gate_proj": r"layers.\1.moe.we_gate",
  r".*model\.layers\.([0-9]+)\.(moe|experts)\.up_proj": r"layers.\1.moe.we_up",
  r".*model\.layers\.([0-9]+)\.(moe|experts)\.down_proj": r"layers.\1.moe.we_down",
  # mlp (maps to mlp field = MLPLayer)
  r".*model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"layers.\1.mlp.w_gate",
  r".*model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"layers.\1.mlp.w_up",
  r".*model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"layers.\1.mlp.w_down",
  # pre/post feedforward norms (gemma4)
  r".*model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": r"layers.\1.ffw_pre_gamma_mlp",
  r".*model\.layers\.([0-9]+)\.post_feedforward_layernorm_1\.weight": r"layers.\1.ffw_post_gamma_mlp",
  r".*model\.layers\.([0-9]+)\.pre_feedforward_layernorm_2\.weight": r"layers.\1.ffw_pre_gamma_moe",
  r".*model\.layers\.([0-9]+)\.post_feedforward_layernorm_2\.weight": r"layers.\1.ffw_post_gamma_moe",
  r".*model\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": r"layers.\1.ffw_post_gamma",
  # per-layer input (gemma4) → post_mlp (MLPLayer with w_up=None)
  r".*model\.layers\.([0-9]+)\.per_layer_input_gate\.weight": r"layers.\1.post_mlp.w_gate",
  r".*model\.layers\.([0-9]+)\.per_layer_projection\.weight": r"layers.\1.post_mlp.w_down",
  r".*model\.layers\.([0-9]+)\.post_per_layer_input_norm\.weight": r"layers.\1.post_mlp_post_gamma",
  # layer scalar (gemma4 full attention)
  r".*model\.layers\.([0-9]+)\.layer_scalar": r"layers.\1.layer_scalar",
  # per-layer input embeddings/projection (gemma4 model-level)
  r".*model\.embed_tokens_per_layer\.weight": "per_layer_embed",
  r".*model\.per_layer_model_projection\.weight": "per_layer_proj",
  r".*model\.per_layer_projection_norm\.weight": "per_layer_proj_gamma",
  # final
  r".*model\.norm\.weight": "gamma_final",
  r"lm_head\.weight": "lm_head",
}


def _torch_key_to_jax_key(source_key, custom_key_map: dict[str, str] | None = None):
  key_maps = dict(_HF_KEY_MAPPING, **(dict() if custom_key_map is None else custom_key_map))
  subs = [re.sub(pat, repl, source_key) for pat, repl in key_maps.items() if re.match(pat, source_key)]
  if len(subs) > 1:
    raise ValueError(f"More than 1 key matched: {subs}")
  else:
    return None if len(subs) == 0 else subs[0]


def _remove_extra_layer_scalars(cfg: g4jax.Config, model_params: dict[str, torch.Tensor]):
  return {k: v for k, v in model_params.items()
          if ((m := re.search(r"layers\.([0-9]+)\.layer_scalar", k)) is None
              or cfg.attention_types[int(m[1])] == "global_attention")}


def _remove_redundant_params(cfg: g4jax.Config, model_params: dict[str, torch.Tensor]):
  model_params_keys = list(model_params.keys())
  for k in model_params_keys:
    if re.match(r".*\.embed_scale", k) is not None:
      del model_params[k]
    if re.match(r".*\.rotary_emb.*freq", k) is not None:
      del model_params[k]
  return model_params


def _drop_unused_kv_projections(cfg: g4jax.Config, model_params: dict[str, torch.Tensor]):
  keys = list(model_params.keys())
  for k in keys:
    if (m := re.search(r".*\.([0-9]+)\.self_attn\.(k|v)_proj\.weight", k)) is not None:
      idx = int(m[1])
      if idx >= cfg.num_layers - max(cfg.num_kv_shared_layers, 0):
        del model_params[k]
    elif (m := re.search(r".*\.([0-9]+)\.self_attn\.k_norm\.weight", k)) is not None:
      idx = int(m[1])
      if idx >= cfg.num_layers - max(cfg.num_kv_shared_layers, 0):
        del model_params[k]
  return model_params


def _split_moe_gate_up_proj(cfg: g4jax.Config, model_params: dict[str, torch.Tensor]):
  keys = list(model_params.keys())
  for k in keys:
    value = model_params[k]
    if (m := re.search(r"(.*)\.gate_up_proj(\.weight|)", k)) is not None:
      assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size * 2, cfg.embed)
      gate, up = value[:, :cfg.moe_ffw_size, :], value[:, cfg.moe_ffw_size:, :]
      del model_params[k]
      model_params[f"{m[1]}.gate_proj{m[2]}"] = gate
      model_params[f"{m[1]}.up_proj{m[2]}"] = up
  return model_params


def _map_weight(source_key, value: torch.Tensor, custom_transform_map: dict[str, Callable] | None = None):
  key_maps = dict(dict(), **(dict() if custom_transform_map is None else custom_transform_map))
  fns = {pat: fn for pat, fn in key_maps.items() if re.match(pat, source_key)}
  if len(fns) > 1:
    raise ValueError(f"More than 1 key matched: {fns}")
  else:
    return value if len(fns) == 0 else list(fns.values())[0](value)


def convert_model_or_layer(
  layer: g4jax.Weights | g4jax.Layer,
  ref_layer: torch.nn.Module,
  cfg: g4jax.Config,
  device: jax.Device | None = None,
  sequential: bool = True,
  custom_key_map: dict[str, str] | None = None,
  custom_transform_map: dict[str, Callable] | None = None,
  allow_unconverted_parameters: bool = False,
  prefix: str | None = None,
):
  device = device if device is not None else jax.devices("cpu")[0]
  torch_params = dict(ref_layer.named_parameters() if hasattr(ref_layer, "named_parameters") else ref_layer)
  torch_params |= dict(ref_layer.named_buffers() if hasattr(ref_layer, "named_buffers") else {})
  torch_params = {k: v for (k, v) in torch_params.items() if prefix is None or k.startswith(prefix)}
  # torch_params = _remove_extra_layer_scalars(cfg, torch_params)
  torch_params = _remove_redundant_params(cfg, torch_params)
  torch_params = _split_moe_gate_up_proj(cfg, torch_params)
  torch_params = _drop_unused_kv_projections(cfg, torch_params)

  layer_params = {
    ".".join(map(_index_to_str, k)): v for (k, v) in jax.tree.flatten_with_path(layer, is_leaf=is_leaf)[0]
  }
  new_params = {k: None for k in layer_params.keys()}

  # some checkpoints store both 'embed_tokens' and 'lm_head' even if the embeddings are tied
  # in this case, we check that the weights are identical and delete 'lm_head'
  if cfg.tie_embed and "lm_head.weight" in torch_params:
    torch.testing.assert_close(torch_params["lm_head.weight"], torch_params["model.embed_tokens.weight"])
    del torch_params["lm_head.weight"]

  def convert_weight_thread(tkey, tweight):
    with jax.default_device(device):
      jweight = convert_weight(tkey, _map_weight(tkey, tweight, custom_transform_map=custom_transform_map), cfg)
    jkey = _torch_key_to_jax_key(tkey, custom_key_map=custom_key_map)
    if jkey is None:
      raise ValueError(f"Could not find parameter mapping for torch paramter: `{tkey}`.")
    if jkey not in new_params:
      raise ValueError(f"The JAX model is not expecting `{jkey}`!  Expected keys are {list(new_params.keys())}")
    if new_params[jkey] is not None:
      raise ValueError(f"Parameter `{jkey}` already set!")
    new_params[jkey] = jweight

  if sequential:
    for tkey, tweight in torch_params.items():
      if allow_unconverted_parameters and _torch_key_to_jax_key(tkey, custom_key_map=custom_key_map) is None:
        continue
      convert_weight_thread(tkey, tweight)
  else:
    futures, executor = [], ThreadPoolExecutor(max_workers=16)
    for tkey, tweight in torch_params.items():
      if allow_unconverted_parameters and _torch_key_to_jax_key(tkey, custom_key_map=custom_key_map) is None:
        continue
      futures.append(executor.submit(convert_weight_thread, tkey, tweight))
    for fut in tqdm(futures, desc="Converting weights"):
      fut.result()

  if not allow_unconverted_parameters:
    none_params = {k: v for k, v in new_params.items() if v is None}
    existing_none_params = {k: v for k, v in layer_params.items() if v is None}
    none_params = {k: v for k, v in none_params.items() if k not in existing_none_params}
    if not all(v is not None for v in new_params.values()):
      from pprint import pprint
      pprint(existing_none_params)
      raise ValueError(f"Not all parameters were converted: {none_params}")
  for (key, param), new_param in zip(layer_params.items(), new_params.values()):
    if new_param is None:
      continue
    if param.shape != new_param.shape:
      raise ValueError(f"Shape of {key=} does not match, expected = {param.shape}, got {new_param.shape}")

  if isinstance(layer, g4jax.Weights):
    return jax.tree.unflatten(jax.tree.structure(layer, is_leaf=is_leaf), new_params.values())
  else:
    return jax.tree.unflatten(
      jax.tree.structure(layer, is_leaf=is_leaf),
      [
        new_param if new_param is not None else param
        for (new_param, param) in zip(new_params.values(), layer_params.values())
      ],
    )
