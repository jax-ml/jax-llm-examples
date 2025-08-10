# JAX LLM examples

A collection (in progress) of example high-performance large language model
implementations, written with JAX.

Current contents include:

* [DeepSeek R1](deepseek_r1_jax/)
* [Llama 4](llama4/)
* [Llama 3](llama3/)
* [Qwen 3](qwen3/)
* [Kimi K2](kimi_k2/)
* [OpenAI GPT OSS](gpt_oss/)

## CLI

### `python3 -m jax_llm_examples.cli --help`
```
usage: python3 -m jax_llm_examples [-h] [--version] [-s SEARCH] {ls,run} ...

A collection of JAX implementations for various Large Language Models.

positional arguments:
  {ls,run}
    ls                  List installed models
    run                 Run specified model. Explicitly calls the main.py as
                        `if __name__ == "__main__"`

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -s SEARCH, --search SEARCH
                        Alternative filepath(s) or fully-qualified name (FQN)
                        to use models from.
```

### `python3 -m jax_llm_examples.cli ls --help`
```
usage: python3 -m jax_llm_examples ls [-h]

options:
  -h, --help  show this help message and exit
```

### `python3 -m jax_llm_examples.cli run --help`
```
usage: python3 -m jax_llm_examples run [-h] -n MODEL_NAME

options:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model-name MODEL_NAME
                        Model name
```

---

For multi-host cluster setup and distributed training, see [multi_host_README.md](./multi_host_README.md) and the [tpu_toolkit.sh script](./misc/tpu_toolkit.sh).
