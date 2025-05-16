# Minimal Llama 4 (family) inference

**tl;dr: open-source Llama 4 inference using JAX, minimal yet performant**

*Note: work in progress, Scout supported with defaults, Maverick's defaults
tuning in progress.*

<br/>

This is a pure JAX implementation of Llama 4 inference, including a checkpoint
converter for the weights. It currently runs on TPU. Support for GPU is
in-progress.

The entire model is defined in [model.py](llama4_jax/model.py) and invoked
via [main.py](main.py). Among other things, the model code demonstrates:
* an MLA attention implementation;
* expert and tensor-parallelism via JAX's
  [`shard_map`](https://docs.jax.dev/en/latest/sharded-computation.html#manual-parallelism-with-shard-map)
  for easy multi-device/multi-host computation; and
* simple int8 quantization.

This example aims to be a concise, self-contained, fully open-source codebase,
with performance that is reasonably comparable to other Llama 4 inference
offerings (at cost). We hope that it is easy to understand and offers an
accessible starting point for performant inference with JAX. See the
[performance rundown](#inference-performance-results) below.

In addition, this repo includes an
[overview](#transformer-parallelism-strategies) of how to shard transformers and
a [discussion](#optimizing-llama-4) of the specific optimizations used in
this implementation, as well as a [workflow](#working-with-multi-host-clusters)
for interactive development on multi-host GPU and TPU clusters using
ipyparallel.

## Table of contents
- [Quickstart](#quickstart)
- [Inference performance results](#inference-performance-results)
- [Transformer parallelism strategies](#transformer-parallelism-strategies)
- [Optimizing Llama 4](#optimizing-llama-4)
- [Working with multi-host clusters](#working-with-multi-host-clusters)
- [Next steps](#next-steps)

## Quickstart

Due to the large model size (671B parameters), a multi-host platform is required to run
the full model. We've tested on v5e-64.

Run on all hosts in the TPU cluster:
```
$ python3 main.py
```
e.g. for Cloud TPU:
```
$ gcloud compute tpus tpu-vm ssh {TPU_NAME} --worker=all \
    --command="cd ~/llama4_jax && python3 main.py"
```

Responses:
```
[
  "\n\nI'm Llama, a model designed by Meta. What’s your name, or should I start guessing?<|eot|><|header_start|>assistant<|header_end|>\n\nI'm Llama, a",
  "\n\nHear me, ye knaves! Gather 'round and heed my words, for I shall regale thee with tales of yonder skies and the",
  "\n\nA question that requires precision!\n\nAs a computer program, I don't have personal preferences, taste buds, or a physical body. Therefore, I neither like nor"
]
```

(See [Working with multi-host clusters](#working-with-multi-host-clusters) for full setup.)

## Inference performance results

#### Llama 4 - Scout

| TPU     | batch size | context length | tok/s    | HBM BW util $^*$ | comments        |
| :------ | ---------: | -------------: | -------: | ---------------: | :-------------- |
| v5e-16  |          1 |           4096 |   85.5   | 71.7%            |                 |
| v5e-16  |         16 |           4096 |   76.3   | 64.0%            |                 |
| v5e-16  |         32 |           4096 |   73.0   | 61.2%            |                 |

#### Llama 4 - Maverick

| TPU     | batch size | context length | tok/s    | HBM BW util $^*$ | comments        |
| :------ | ---------: | -------------: | -------: | ---------------: | :-------------- |

TOOD

Results generated using jax 0.5.3, Python 3.10.15. Cost computation based on
https://cloud.google.com/spot-vms/pricing, region `us-central1` as of Feb 28
2025.

$^*$ HBM BW util computed from on-chip network size = 102 GB (Scout) and (TODO)
(Maverick) and 819 GB/s theoretical TPU v5e HBM BW. For small batches, not all
experts are loaded from HBM leading to >100% utilization.

### Optimization Decisions

Llama 4 is a fairly minimal MoE model in that it (i) uses grouped-query
attention, (GQA) and (ii) uses either a simple MLP layer with a gate, up and
down projections or a simple MoE layer with the same linear matrices (but
repeated for each expert).  This presents several opportunities in optimizing
the model for TPUs and GPUs to maximize either compute (in training) or
memory-bandwidth use in inference.

#### Accelerator Agnostic Optimization

- Q: What parameter influences inference speed the most?

  A: HBM bandwidth

- Q: Fully-replicated or sharded activations?

  A: For low-latency decoding, fully-replicated activations are usually faster
  since that strategy relies on a all-reduce communication instead of repeated
  reduce-scatters. Computation (weights shards) is still partitioned, but local
  shared memory is traded for lower-latency communication.

- Q: Why doesn't this match the cost and performance of proprietary inference APIs?

  A: This example aims to balance simplicity with performance, and thus does not
  implement every possible optimization if they would add considerable complexity
  (e.g. heavy use of custom kernels). In addition, this example only uses
  well-known optimization strategies, and does not aim to introduce any new or
  closed-source techniques that inference providers may have independently developed.



#### TPU Optimizations

- Q: Inference or training TPUs (e.g., v5e or v5p)?

  A: Inference (v5e) since the matmul units are not as powerful, but can be lower latency at low utilization.

- Q: How to work with multiple hosts?

  A: (1) Launching the same python script via `python3 script.py` or (2) our `ipyparallel` setup.

- Q: Which TPU image to use?

  A: For v5e: `v2-alpha-tpuv5-lite`, for v6e: `v2-alpha-tpuv6e`. See [runtimes](https://cloud.google.com/tpu/docs/runtimes).

#### Custom Kernels

TODO

## Transformer parallelism strategies

This section overviews different sharding strategies and their performance considerations for Transformer architectures in general.
For a very in-depth guide on this topic, check out [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).
The next section goes over Llama-specific optimizations.

A typical decoder-only transformer consists of

1. An input embedding
    - a single weight $V \times D$
2. Repeated Decoder Layers (Attention + a Feed-forward layer)
    * Attention Layer
        - project input $BSD$ to $BSNH$ for queries, $BSNH$ for keys and $BSNH$ values, typically $D \approx N \cdot H$
        - compute the attention operation on $BSNH$, $BSNH$, $BSNH$ giving $BSNH$
        - project the output $BSNH$ back to $BSD$ using a projection matrix
    * Feed-forward Layer - a Multilayer Perceptron (MLP) or a Mixture-of-Experts (MoE)
        - always (i) up-projection -> (ii) nonlinearity -> (iii) down-projection
        - MLP
            - up-projection: $BSD \times DF \rightarrow BSF$
            - down-projection: $BSF \times DF \rightarrow BSD$
        - MoE
            - each token in $BS$ can be routed to a matrix slice $EDF[\text{idx}, :, :]$
            - up-projection: $BSD \times EDF \rightarrow BSF$
            - down-projection: $BSF \times EDF \rightarrow BSD$
3. An output projection
    - a single weight $D \times V$

<p align="center">

| Abbreviation | Dimension                           |
| :----------: | ----------------------------------- |
|      V       | vocabulary size                     |
|      B       | batch                               |
|      S       | sequence                            |
|      D       | model dimension                     |
|      F       | up-projection dimension             |
|      N       | number of query, key or value heads |
|      H       | head dimension                      |
|      E       | expert dimension                    |

</p>

### Sharding strategies for a transformer

The simplest sharding strategy, naive eager pipeline parallelism, is putting the
first couple of layers on the first device, the next couple of layers on the
second, and so on, and it requires simple communication of passing activations
between devices every couple of layers. Unfortunately, for fast inference, this
implies that latter devices wait for the earlier ones to complete - decoding at
a speed of a single device. Strategies that favor parallel work among devices,
tensor-parallelism and fully-sharded data-parallel, are a better fit. We find
tensor-parallelism results in fastest inference.

<p align="center">

| Strategy      | Input                |                                      | QKV                     |                                           | Output                    |                                          | Up                      |                                            | Down                      |                   |
| ------------- | -------------------- | ------------------------------------ | ----------------------- | ----------------------------------------- | ------------------------- | ---------------------------------------- | ----------------------- | ------------------------------------------ | ------------------------- | ----------------- |
| $\text{TP}_1$ | $BD$                 | $\overset{W_{qkv}}{\longrightarrow}$ | $BH_{\color{red} x}$    | $\overset{W_\text{out}}{\longrightarrow}$ | $BD$                      | $\overset{W_\text{up}}{\longrightarrow}$ | $BF_{\color{red} x}$    | $\overset{W_\text{down}}{\longrightarrow}$ | $BD$                      | $\longrightarrow$ |
|               |                      |                                      |                         |                                           | ${\color{red} \text{AR}}$ |                                          |                         |                                            | ${\color{red} \text{AR}}$ |                   |
| $\text{TP}_2$ | $BD_{\color{red} x}$ | $\overset{W_{qkv}}{\longrightarrow}$ | $BH_{\color{red} x}$    | $\overset{W_\text{out}}{\longrightarrow}$ | $BD_{\color{red} x}$      | $\overset{W_\text{up}}{\longrightarrow}$ | $BF_{\color{red} x}$    | $\overset{W_\text{down}}{\longrightarrow}$ | $BD_{\color{red} x}$      | $\longrightarrow$ |
|               |                      |                                      | $\color{red} \text{RS}$ |                                           | $\color{red} \text{RS}$   |                                          | $\color{red} \text{RS}$ |                                            | $\color{red} \text{RS}$   |                   |
| $\text{FSDP}$ | $BD$                 | $\overset{W_{qkv}}{\longrightarrow}$ | $BH$                    | $\overset{W_\text{out}}{\longrightarrow}$ | $BD$                      | $\overset{W_\text{up}}{\longrightarrow}$ | $BF$                    | $\overset{W_\text{down}}{\longrightarrow}$ | $BD$                      | $\longrightarrow$ |
|               |                      | $\color{red} \text{AG}$              |                         | $\color{red} \text{AG}$                   |                           | $\color{red} \text{AG}$                  |                         | $\color{red} \text{AG}$                    |                           |                   |

</p>

where:
- ${\color{red} \text{AR}}$ - all-reduce [`jax.lax.psum`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum.html)
- ${\color{red} \text{RS}}$ - reduce-scatter [`jax.lax.psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html)
- ${\color{red} \text{AG}}$ - all-gather [`jax.lax.all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html)

The key to designing a sharding strategy is minimizing communication overhead.
There are typically several alternatives and the compiler will overlap
communication with computation as much as possible. Given this, it's usually
worth trying several alternatives and picking the one minimizing the total
runtime. The best strategy depends on the hardware configuration. The following
are general rules of thumb in different contexts

For low latency with 1D/2D sharding, the primary sharded matrix multiplication strategies are:
  - $BD_x \times D_xF \underset{\text{scatter}}{\longrightarrow} B F_x$ -
  contracting dimension with scatter (1 unit of comms)
  - $BD \times DF_x \underset{}{\longrightarrow} B F_x$ - replicated activations (no comms)
  - $BD_x \times D_xF \underset{\text{all-reduce}}{\longrightarrow} B F$ -
    contracting dimension with reduce comms after (2 units of comms)

- for attention activations should be sharded over heads (effectively the feature dimension)
- do not all-gather weights (no FSDP)

#### FSDP vs tensor-parallelism trade-off

Total FSDP comms is:

$$ 4 \times \text{cost}(\text{all-gather}) = 2 DH + 2 DF \approx 4 D^2 $$

Total tensor-parallelism comms is:

$$ 4 \times \text{cost}(\text{scatter}) = 2 \times \text{cost}(\text{all-reduce}) = 2 B H + 2 B F \approx 4 * B D $$

This, very roughly implies the trade-off (in favor of FDSP):

$$ \mathcal{O} \left( D^2 \right ) \leq \mathcal{O}\left(B D \right) \rightarrow \mathcal{O}\left(D\right) \leq \mathcal{O}\left(B\right) $$

FSDP can be more efficient if the the batch size is on the order of the model dimension. For fast latency Llama 4 Scout

$$
B = 16 ~~~ D = 5120
$$

strongly implying a preference for tensor-parallelism (in the context of low-latency decoding).

## Optimizing Llama 4

### MLP Layers (and shared expert) in Inference

The MLP in the network is a fairly standard up-down MLP linear layer. Llama 4
uses either an MLP or an MoE layer (which contains an MLP layer referred to as
shared expert) after every attention layer.

$$
\left(\text{silu}(x W_\text{gate}) \cdot (x W_\text{up}) \right) W_\text{down}
$$

so we have to choose a two-step matrix multiplication sharding strategy
($W_\text{gate}$ or $W_\text{up}$ then $W_\text{down}$). For low-latency
settings pure tensor-parallelism works well

$$
B D \overset{DF_{\color{red}x}}\rightarrow B F_{\color{red}x} \overset{F_{\color{red}x}D}{\rightarrow} {\color{red}{AR}} \rightarrow BD
$$

<p align="center">

TODO

<!--![Decode Profile](./images/ds_r1_decode_profile_bs8_ctx2048.png)-->

<p align="center">
Fig: Decode profile TODO
<p>
</p>

## Working with multi-host clusters

When working with many accelerators, JAX offers
[Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
with a global view of the computation, but the program needs to be run on many hosts
each of which controls a subset of the actual accelerators.

The simplest way to run a JAX program on multiple hosts is to [run the same
Python file from all the hosts at the same
time](https://docs.jax.dev/en/latest/multi_process.html) - for example by
launching an ssh command on all hosts in the cluster.

However, for development it's often easier to (1.) efficiently share code
changes to all hosts, (2.) have a way of easily launching computation on all
hosts and (3.) have the ability to debug interactively.

This section shows how you can do that:
1. Shared disk setup - [NFS](https://ubuntu.com/server/docs/network-file-system-nfs) & [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse)
2. [Batch SSH commands](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/ssh)
3. Interactive cluster setup with [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/)

This guide has specific instructions for setting up a TPU Pod with GCS, but a
similar setup can be applied to any Linux multi-host platform, including GPU.

#### Setup McJax
```
bash mcjax_setup.sh
```

#### Clone the model
```
source mcjax_defs.sh

MODEL_INSTALL_COMMAND=$(cat << EOM
  cd ~/
  if [ ! -d jax-llm-examples/llama4 ]; then
    git clone https://github.com/jax-ml/jax-llm-examples.git
  fi
  cd jax-llm-examples/llama4
  uv pip install -e .
EOM
)
tpu_exec 0 15 MODEL_INSTALL_COMMAND
```

#### Setup [sshfs](https://github.com/libfuse/sshfs) (optionally)
For a quick preview from a local machine.
```
sshfs ~/$LOCAL_DATA_FOLDER $WORKER0_IP$:~/$LOCAL_DATA_FOLDER
```

#### Start the jupyter server
```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''
```

#### Setup port forward on local machine
```
source mcjax_defs.sh
gcloud alpha compute tpus tpu-vm ssh --zone $TPU_ZONE --project=$TPU_PROJECT $TPU_NAME -L 8888:localhost:8888
```

#### Jupyter Notebook
Open http://127.0.0.1:8888/
> Cell 0:
```python
import ipyparallel as ip
from pathlib import Path
connection_file = Path("~/nfs/security/ipcontroller-client.json").expanduser()
client = ip.Client(connection_info=connection_file)
print(sorted(list(client._engines.keys())))  # one less than worker num
# this process is the final worker
```

> Cell 1:
```python
%%px --local
import socket
import jax
jax.distributed.initialize()  # no arguments, TPUs automatically detect peers
print(f"Hello from {socket.gethostname()}")
```

> Note: "--local" argument means "also run on this process", it's necessary to
> get easy access to the output of computations on worker 0


## Next steps

- [ ] GPU suport
- [ ] ragged attention auto-tuning
- [ ] ragged decode kernel
