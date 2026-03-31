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
from functools import partial
import itertools

from absl.testing import absltest, parameterized
import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import AxisType, reshard
import numpy as np

from gemma4_jax import hf_configs
from gemma4_jax import model as g4jax


named_product = lambda **kw: [
  dict(testcase_name="_".join(f"{k}_{v}" for k, v in zip(kw, c)), **dict(zip(kw, c)))
  for c in itertools.product(*kw.values())
]

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)

MOE_CFG = g4jax.hf_to_jax_config(json.loads(hf_configs.GEMMA4_26B_A4B_JSON)["text_config"])
DENSE_CFG = g4jax.hf_to_jax_config(json.loads(hf_configs.GEMMA4_31B_JSON)["text_config"])
E2B_CFG = g4jax.hf_to_jax_config(json.loads(hf_configs.GEMMA4_E2B_JSON)["text_config"])

MODELS = ["moe", "dense", "e2b"]


class TestModel(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh((1, 1, len(jax.devices())), ("x", "y", "z"), axis_types=(AxisType.Explicit,) * 3)
        _small = lambda cfg, **kw: dataclasses.replace(
            cfg, mesh=self.mesh,
            num_layers=8, embed=32,
            vocab_size=128, mlp_ffw_size=32,
            per_layer_input_dim=min(cfg.per_layer_input_dim, 16),
            vocab_size_per_layer_input=min(cfg.vocab_size_per_layer_input, 128),
            q_heads=8,
            local_kv_heads=4,
            local_head_dim=16,
            global_head_dim=32,
            attention_types=cfg.attention_types[:8], **kw,
        )
        self.cfgs = dict(
            moe=_small(MOE_CFG, moe_ffw_size=16, moe_num_experts=16),
            dense=_small(DENSE_CFG),
            e2b=_small(E2B_CFG, num_kv_shared_layers=2),
        )

    @staticmethod
    def err_fn(x, y, axis: int | None = None):
        x, y = np.array(x), np.array(y)
        axis_ = axis if axis is not None else (-1 if x.ndim <= 3 else (-1, -2))
        norm = partial(np.linalg.norm, axis=axis_)
        return norm(x - y) / np.maximum(norm(y), 1e-7)

    @parameterized.named_parameters(*named_product(model=MODELS, quant=[False, True]))
    def test_model_init(self, model, quant):
        cfg = dataclasses.replace(self.cfgs[model], quant_attn=quant, quant_moe=quant, quant_mlp=quant)
        weights = g4jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.named_parameters(*named_product(model=MODELS, quant=[False, True]))
    def test_init_hashing(self, model, quant):
        cfg = dataclasses.replace(self.cfgs[model], quant_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=g4jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = g4jax.Weights.abstract(cfg)
            abstract2 = g4jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = g4jax.Weights.shardings(cfg)
            shardings2 = g4jax.Weights.shardings(cfg)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = g4jax.KVCache.abstract(cfg, 2)
            abstract2 = g4jax.KVCache.abstract(cfg, 2)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = g4jax.KVCache.shardings(cfg, 2)
            shardings2 = g4jax.KVCache.shardings(cfg, 2)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

    @parameterized.named_parameters(*named_product(model=MODELS, quant=[False, True]))
    def test_cache_init(self, model, quant):
        cfg = dataclasses.replace(self.cfgs[model], quant_cache=quant)
        cache = g4jax.KVCache.init(random.key(0), cfg, 2)
        del cache

    @parameterized.named_parameters(*named_product(model=MODELS, quant_weights=[False, True], quant_cache=[True, False]))
    def test_prefill_decode(self, model, quant_weights, quant_cache):
        cfg = dataclasses.replace(
            self.cfgs[model], quant_attn=quant_weights, quant_moe=quant_weights, quant_mlp=quant_weights,
            quant_cache=quant_cache,
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = g4jax.Weights.init(random.key(0), cfg)
        cache = g4jax.KVCache.init(random.key(0), cfg, tokens.shape[0])
        with jax.sharding.set_mesh(cfg.mesh):
            max_tokens, _, cache = g4jax.prefill(tokens, weights, cache, cfg)
            next_tokens = max_tokens[:, :-1]
            for _ in range(2):
                next_tokens, cache = g4jax.decode_step(next_tokens, weights, cache, cfg)

    @parameterized.named_parameters(*named_product(model=MODELS, quant_weights=[False, True], quant_cache=[True, False]))
    def test_prefill_logits(self, model, quant_weights, quant_cache):
        cfg: g4jax.Config = dataclasses.replace(
            self.cfgs[model], quant_attn=quant_weights, quant_moe=quant_weights, quant_mlp=quant_weights,
            quant_cache=quant_cache,
        )
        keys = iter(jax.random.split(jax.random.key(0), 2048))
        input = jax.random.randint(next(keys), (4, 25), minval=1, maxval=cfg.vocab_size, dtype=jnp.int32)
        segment_ids = jnp.array([0, 7, 2, 11])[:, None] <= jnp.arange(input.shape[-1])[None, :]
        input = jnp.where(segment_ids, input, g4jax.PAD_ID)
        weights = g4jax.Weights.init(random.key(0), cfg)

        @partial(jax.jit, donate_argnames=("cache",))
        def decode_step(key: jax.typing.ArrayLike, last_tokens: jax.Array, weights, cache):
            assert last_tokens.ndim == 2
            segment_ids = (last_tokens != g4jax.PAD_ID).astype(jnp.int32)
            next_logits, cache = g4jax.forward(last_tokens, segment_ids, weights, cfg, cache)
            next_tokens = g4jax.sample_top(key, next_logits, k=cfg.sample_topk, temp=cfg.sample_temp)
            return next_tokens, next_logits, cache

        assert cfg.mesh is not None
        with jax.sharding.set_mesh(cfg.mesh):
            all_tokens, all_logits = [], []
            cache = g4jax.KVCache.init(random.key(0), cfg, input.shape[0])
            max_tokens, logits, cache = g4jax.prefill(input, weights, cache, cfg)
            next_tokens = max_tokens[:, input.shape[-1]-1:input.shape[-1]]
            all_tokens.append(next_tokens)
            for _ in range(20):
                next_tokens, next_logits, cache = decode_step(next(keys), next_tokens, weights, cache)
                all_logits.append(next_logits)
                all_tokens.append(next_tokens)

            # all_tokens = all_tokens[:15] + all_tokens[16:]  # DEBUG
            # all_tokens = jnp.concat([input, jnp.concat(all_tokens, axis=1)], axis=1)
            all_tokens = jnp.concat([input, jnp.concat(all_tokens, axis=1)], axis=1)[:, :-1]
            all_logits = jnp.concat([
                logits[:, :input.shape[1], :], reshard(jnp.concat(all_logits, axis=1), jax.typeof(logits).sharding.spec)
            ], axis=1)
            cache = g4jax.KVCache.init(random.key(0), cfg, input.shape[0])
            _, comparison_logits, _ = g4jax.prefill(all_tokens, weights, cache, cfg)
            comparison_logits = comparison_logits[:, :all_logits.shape[1], :]
            mask = jnp.pad(segment_ids, ((0, 0), (0, all_tokens.shape[1] - segment_ids.shape[1])), constant_values=True)
            err = jnp.where(mask, self.err_fn(comparison_logits, all_logits), 0)
            with jnp.printoptions(linewidth=200):
                print(f"{model=} {quant_weights=} {quant_cache=}")
                print(err[:, -10:])

if __name__ == "__main__":
    absltest.main()
