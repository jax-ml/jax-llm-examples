# Minimal OpenAI GPT OSS inference

**tl;dr: open-source OpenAI GPT OSS inference using JAX, minimal yet performant**

This model is a work in progress, but it should already work well on both TPU and GPU.

<br/>

This is a pure JAX implementation of OpenAI's GPT OSS for inference, including a
checkpoint converter for the K2 Instruct weights. on TPU.
It should work on GPU.

The entire model is defined in [model.py](gpt_oss_jax/model.py) and invoked
via [main.py](main.py).

## Quickstart

Run:
```
$ python3 main.py
```
