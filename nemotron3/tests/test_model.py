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

import json
import dataclasses
from absl.testing import absltest, parameterized

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import AxisType
from nemotron3_jax import model as n3jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)


NEMOTRON3_30B_JSON = """
{
  "architectures": [
    "NemotronHForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModel": "modeling_nemotron_h.NemotronHForCausalLM",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
  },
  "bos_token_id": 1,
  "chunk_size": 128,
  "conv_kernel": 4,
  "eos_token_id": 2,
  "expand": 2,
  "head_dim": 128,
  "hidden_dropout": 0.0,
  "hidden_size": 2688,
  "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
  "initializer_range": 0.02,
  "intermediate_size": 1856,
  "layer_norm_epsilon": 1e-05,
  "mamba_head_dim": 64,
  "mamba_hidden_act": "silu",
  "mamba_num_heads": 64,
  "mamba_proj_bias": false,
  "mamba_ssm_cache_dtype": "float32",
  "max_position_embeddings": 262144,
  "mlp_bias": false,
  "mlp_hidden_act": "relu2",
  "model_type": "nemotron_h",
  "moe_intermediate_size": 1856,
  "moe_shared_expert_intermediate_size": 3712,
  "n_group": 1,
  "n_groups": 8,
  "n_routed_experts": 128,
  "n_shared_experts": 1,
  "norm_eps": 1e-05,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 52,
  "num_key_value_heads": 2,
  "num_logits_to_keep": 1,
  "pad_token_id": 0,
  "partial_rotary_factor": 1.0,
  "rescale_prenorm_residual": true,
  "residual_in_fp32": false,
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "sliding_window": null,
  "ssm_state_size": 128,
  "tie_word_embeddings": false,
  "time_step_floor": 0.0001,
  "time_step_max": 0.1,
  "time_step_min": 0.001,
  "topk_group": 1,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.55.4",
  "use_bias": false,
  "use_cache": true,
  "use_conv_bias": true,
  "use_mamba_kernels": true,
  "vocab_size": 131072
}
"""

FULL_CFG = n3jax.hf_to_jax_config(json.loads(NEMOTRON3_30B_JSON))


class TestModel(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1), ("x", "y", "z"), axis_types=(AxisType.Explicit,) * 3
        )
        # small config: 2 layers with pattern "ME" (one Mamba, one MoE)
        self.small_cfg = dataclasses.replace(
            FULL_CFG, mesh=self.mesh, num_layers=2, layer_pattern="ME", embed=32, vocab_size=128,
        )

    @parameterized.product(quant=[False, True])
    def test_model_init(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_attn=quant, quant_moe=quant, quant_mamba=quant)
        weights = n3jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.product(quant=[False, True])
    def test_init_hashing(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=n3jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = n3jax.Weights.abstract(cfg)
            abstract2 = n3jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = n3jax.Weights.shardings(cfg)
            shardings2 = n3jax.Weights.shardings(cfg)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = n3jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            abstract2 = n3jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = n3jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            shardings2 = n3jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

    @parameterized.product(quant=[False, True])
    def test_cache_init(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_cache=quant)
        cache = n3jax.KVCache.init(random.key(0), cfg, 2, cfg.max_seq_len)
        del cache

    @parameterized.product(quant_weights=[False, True], quant_cache=[True, False])
    def test_kernel_prefill(self, quant_weights, quant_cache):
        # use a pattern with attention ("*") so the kernel path is exercised
        cfg = dataclasses.replace(
            self.small_cfg, layer_pattern="*E",
            quant_attn=quant_weights, quant_moe=quant_weights, quant_mamba=quant_weights, quant_cache=quant_cache,
            use_prefill_attn_kernel=True,
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = n3jax.Weights.init(random.key(0), cfg)
        cache = n3jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        with jax.sharding.set_mesh(cfg.mesh):
            with self.assertRaisesRegex(NotImplementedError, r"Currently, only TPU supports prefill attention.*"):
                _ = n3jax.prefill(tokens, weights, cache, cfg)

    @parameterized.product(quant_weights=[False, True], quant_cache=[True, False])
    def test_prefill_decode(self, quant_weights, quant_cache):
        cfg = dataclasses.replace(
            self.small_cfg,
            quant_attn=quant_weights, quant_moe=quant_weights, quant_mamba=quant_weights, quant_cache=quant_cache,
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = n3jax.Weights.init(random.key(0), cfg)
        cache = n3jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        with jax.sharding.set_mesh(cfg.mesh):
            max_tokens, _, cache = n3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with jax.sharding.set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = n3jax.decode_step(next_tokens, weights, cache, cfg)


if __name__ == "__main__":
    absltest.main()
