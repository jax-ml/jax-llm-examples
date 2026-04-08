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
from nemotron3_jax import lora as n3lora

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)


# Reuse the same HF config as test_model.py
NEMOTRON3_30B_JSON = """
{
  "architectures": ["NemotronHForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
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


def _small_cfg(mesh):
    # Small 3-layer config covering all three layer kinds: Mamba, MoE, Attention.
    return dataclasses.replace(
        FULL_CFG, mesh=mesh, num_layers=3, layer_pattern="ME*", embed=32, vocab_size=128,
    )


class TestLoRA(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1), ("x", "y", "z"), axis_types=(AxisType.Explicit,) * 3
        )
        self.cfg = _small_cfg(self.mesh)
        # Adapt every layer of every kind in the small config so all code paths are exercised.
        self.lora_cfg = n3lora.LoRAConfig(
            rank=4, alpha=8.0,
            target_attn_layers=tuple(range(self.cfg.num_layers)),
            target_mamba_layers=tuple(range(self.cfg.num_layers)),
            target_moe_shared=True,
        )

    def test_abstract_and_init(self):
        abstract = n3lora.LoRAWeights.abstract(self.cfg, self.lora_cfg)
        self.assertEqual(len(abstract.layers), self.cfg.num_layers)
        # layer pattern: M E *
        self.assertIsNotNone(abstract.layers[0].mamba)
        self.assertIsNone(abstract.layers[0].attn)
        self.assertIsNone(abstract.layers[0].moe_shared)
        self.assertIsNotNone(abstract.layers[1].moe_shared)
        self.assertIsNone(abstract.layers[1].mamba)
        self.assertIsNotNone(abstract.layers[2].attn)

        with jax.sharding.set_mesh(self.cfg.mesh):
            lora = n3lora.LoRAWeights.init(random.key(0), self.cfg, self.lora_cfg)
        # All `b` matrices should be exactly zero on a fresh init (PEFT convention).
        def _check_b_zero(node):
            if isinstance(node, n3lora.LoRAPair):
                self.assertTrue(jnp.all(node.b == 0))
        jax.tree.map(_check_b_zero, lora, is_leaf=lambda x: isinstance(x, n3lora.LoRAPair))

    def test_zero_lora_equivalence(self):
        """Forward with a freshly-initialized (b=0) LoRA must equal forward without LoRA."""
        cfg = self.cfg
        tokens = jnp.ones((1, 16), dtype=jnp.int32)
        with jax.sharding.set_mesh(cfg.mesh):
            weights = n3jax.Weights.init(random.key(1), cfg)
            cache = n3jax.KVCache.init(random.key(2), cfg, tokens.shape[0], cfg.max_seq_len)
            lora = n3lora.LoRAWeights.init(random.key(3), cfg, self.lora_cfg)

            # Baseline (no LoRA)
            base_tokens, _, _ = n3jax.prefill(tokens, weights, cache, cfg)

            # Re-init the cache (prefill donates it).
            cache2 = n3jax.KVCache.init(random.key(2), cfg, tokens.shape[0], cfg.max_seq_len)

            def _prefill_with_lora(toks, w, c, cfg, lora):
                from functools import partial
                import math
                pad_to = 2 ** math.ceil(math.log2((toks.shape[-1])))
                prompt, segs = n3jax.prepare_chunk(toks, pad_to=pad_to, pad_id=n3jax.PAD_ID)
                c = dataclasses.replace(
                    c,
                    starts=n3jax._count_left_padding(prompt, pad_id=n3jax.PAD_ID),
                    iter=-jnp.ones_like(c.iter),
                )
                logits, c = jax.jit(n3jax.forward)(
                    prompt, segs, w, cfg, c, lora, self.lora_cfg.scaling
                )
                return jnp.argmax(logits, -1)

            lora_tokens = _prefill_with_lora(tokens, weights, cache2, cfg, lora)

        self.assertEqual(base_tokens.shape, lora_tokens.shape)
        self.assertTrue(jnp.array_equal(base_tokens, lora_tokens),
                        f"baseline {base_tokens} != lora {lora_tokens}")

    def test_grad_flows_only_to_lora(self):
        """jax.grad with respect to LoRA leaves should produce nonzero grads."""
        cfg = self.cfg
        tokens = jnp.ones((1, 8), dtype=jnp.int32)
        with jax.sharding.set_mesh(cfg.mesh):
            weights = n3jax.Weights.init(random.key(11), cfg)
            lora = n3lora.LoRAWeights.init(random.key(13), cfg, self.lora_cfg)

            # Make `b` nonzero so the LoRA path actually contributes; otherwise grads
            # could be zero by coincidence in some branches.
            def _perturb_b(node):
                if isinstance(node, n3lora.LoRAPair):
                    return n3lora.LoRAPair(a=node.a, b=node.b + 0.01)
                return node
            lora = jax.tree.map(_perturb_b, lora, is_leaf=lambda x: isinstance(x, n3lora.LoRAPair))

            segs = jnp.ones_like(tokens)

            def loss_fn(lora_tree, w):
                logits, _ = n3jax.forward(tokens, segs, w, cfg, None, lora_tree, self.lora_cfg.scaling)
                return jnp.sum(logits.astype(jnp.float32) ** 2)

            grads = jax.jit(jax.grad(loss_fn))(lora, weights)

        # Every LoRA leaf should receive a gradient (not None) and at least one should be nonzero.
        any_nonzero = False
        leaf_count = 0
        for leaf in jax.tree.leaves(grads):
            leaf_count += 1
            if jnp.any(leaf != 0):
                any_nonzero = True
        self.assertGreater(leaf_count, 0)
        self.assertTrue(any_nonzero, "expected at least some nonzero LoRA gradients")


if __name__ == "__main__":
    absltest.main()
