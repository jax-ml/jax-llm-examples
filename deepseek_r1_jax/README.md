# Minimal Deepseek R1 inference

**tl;dr: open-source Deepseek R1 inference using JAX, minimal yet performant**

<br/>

This is a pure JAX implementation of Deepseek V3 inference, including a
checkpoint converter for the R1 weights. It currently runs on TPU. Support for
GPU is in-progress.

The entire model is defined in [model.py](deepseek_r1_jax/model.py) and invoked
via [main.py](main.py). Among other things, the model code demonstrates:
* an MLA attention implementation;
* expert and tensor-parallelism via JAX's
  [`shard_map`](https://docs.jax.dev/en/latest/sharded-computation.html#manual-parallelism-with-shard-map)
  for easy multi-device/multi-host computation; and
* simple int8 quantization.

This example aims to be a concise, self-contained, fully open-source codebase,
with performance that is reasonably comparable to other R1 inference offerings (at
cost). We hope that it is easy to understand and offers an accessible starting
point for performant inference with JAX. See the
[performance rundown](#inference-performance-results) below.

In addition, this repo includes an
[overview](#transformer-parallelism-strategies) of how to shard transformers and
a [discussion](#optimizing-deepseek-v3) of the specific optimizations used in
this implementation, as well as a [workflow](#working-with-multi-host-clusters)
for interactive development on multi-host GPU and TPU clusters using
ipyparallel.

## Table of contents
- [Quickstart](#quickstart)
- [Inference performance results](#inference-performance-results)
- [Transformer parallelism strategies](#transformer-parallelism-strategies)
- [Optimizing Deepseek V3](#optimizing-deepseek-v3)
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
    --command="cd ~/deepseek-r1-jax && python3 main.py"
```

```
Responses:
['\n'
 "Okay, the user asked me to tell my name. But I need to remember that I'm "
 'supposed to respond as an AI assistant without revealing any personal '
 'details',
 '\n'
 'Okay, the user wants to know how to describe the weather in Old English '
 'using long prose. Let me start by recalling what Old English is like. It',
 '\n'
 'Okay, the user asked, "Do you like ice cream," and wants me to be extremely '
 "precise. Let me start by understanding what they're looking for"]
```
(See [Working with multi-host clusters](#working-with-multi-host-clusters) for full setup.)

## Inference performance results

| TPU     | batch size | context length | tok/s    | HBM BW util $^*$ | comments        |
| :------ | ---------: | -------------: | -------: | ---------------: | :-------------- |
| v5e-64  |          1 |             32 |   75.8   | 113%             |                 |
| v5e-64  |          1 |            512 | **75.9** | 113%             | max tok/s       |
| v5e-64  |          1 |           4096 |   73.8   | 110%             |                 |
| v5e-64  |          1 |           8192 |   71.0   | 106%             |                 |
| v5e-64  |          8 |             32 |   50.5   | 75.2%            |                 |
| v5e-64  |          8 |            512 |   48.0   | 71.4%            |                 |
| v5e-64  |          8 |           4096 |   42.1   | 62.6%            |                 |
| v5e-64  |          8 |           8192 |   35.6   | 52.9%            |                 |
| v5e-64  |        128 |             32 |   19.6   | 29.1%            | cost optimal    |
| v5e-64  |        128 |            512 |   17.4   | 25.8%            |                 |

Results generated using jax 0.5.2, Python 3.10.15. Cost computation based on
https://cloud.google.com/spot-vms/pricing, region `us-central1` as of Feb 28
2025.

$^*$ HBM BW util computed from on-chip network size = 780 GB and 819 GB/s
theoretical TPU v5e HBM BW. For small batches, not all experts are loaded from
HBM leading to >100% utilization.

### Optimization Decisions

Deepseek is a unique model in that it (i) uses a unique form of attention,
Multi-head Latent Attention (MLA) and (ii) uses an MoE layer with a large number of
small experts. This presents some challenges in optimizing the model for
TPUs and GPUs to maximize either compute (in training) or memory-bandwidth use
in inference.

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

- Q: How to efficiently compute MLA attention on TPUs (which has 128 aligned-registers) for embeddings `nope` (d=128) vs `pe` embedding (d=64)

  A: Instead of concatenating the embeddings, we compute the inner product `qk`,`qk_nope` and `qk_pe`, separately summing them.

- Q: Inference or training TPUs (e.g., v5e or v5p)?

  A: Inference (v5e) since the matmul units are not as powerful, but can be lower latency at low utilization.

- Q: How to work with multiple hosts?

  A: (1) Launching the same python script via `python3 script.py` or (2) our `ipyparallel` setup.

- Q: Which TPU image to use?

  A: For v5e: `v2-alpha-tpuv5-lite`, for v6e: `v2-alpha-tpuv6e`. See [runtimes](https://cloud.google.com/tpu/docs/runtimes).
  
#### Custom Kernels

1. [ragged dot](deepseek_r1_jax/decode_ragged_dot.py) - a grouped matmul operation
    
    *Needed for good decode inference performance with small batches.*

    XLA underlying JAX is very good at merging and fusing operations which means
    we often don't need custom kernels for optimal hardware performance.
    However, Deepseek R1 uses uncommonly many, but small experts.
    
    For anything but small batch sizes in decode, we can use
    [jax.lax.ragged_dot](https://github.com/jax-ml/jax/blob/5179642eb5572b8fbec01dca5e03e2a636e513c2/jax/_src/lax/lax.py#L2209)
    for full performance, but where `jax.lax.ragged_dot` is suboptimal, we write
    a custom TPU kernel which more aggressively prefetches the right-hand side
    into [TPU's VMEM](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

## Transformer parallelism strategies

This section overviews different sharding strategies and their performance considerations for Transformer architectures in general.
For a very in-depth guide on this topic, check out [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). 
The next section goes over Deepseek-specific optimizations.

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

The simplest sharding strategy, pipeline parallelism, is putting the first
couple of layers on the first device, the next couple of layers on the second,
and so on, since it requires simples communication of passing activations
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

FSDP can be more efficient if the the batch size is on the order of the model dimension. For fast latency Llama 3.1 70B

$$
B = 16 ~~~ D = 8192
$$

strongly implying a preference for tensor-parallelism (in the context of low-latency decoding).

## Optimizing Deepseek V3

### MLA Attention

The attention layer computes the equivalent of $x_q = x W_q$, $x_{k,v} = x W_{k,q}$ via

$$
x_{q,k,v} = \text{rms-norm}\left( x W_{A_{q,k,v}}\right) W_{B_{q,k,v}}
$$

In our low-latency setting, we have fully-replicated activations and need
outputs sharded over attention heads.

In regular attention we don't have to communicate because $x W_{q,k,v}$ is
$BD \overset{DQ_xH}{\rightarrow} BQ_xH$, (i.e., the weight sharding implies
output sharding), but in MLA we have two matmuls potentially requiring
communication which adds to latency. To side-step this problem, we **fully
replicated either matrix.** Since the rms-norm operates on the full tensor axis,
it's often easier to replicate $W_{A_{q,k,v}}$ since that results in the
intermediate $x W_{A_{q,k,v}}$ being fully replicated as well.

### MLP Layers (and shared expert) in Inference

The MLP in both the first 3 layers of the network (Deepseek R1 uses 3 standard
MLPs followed by 58 MoE layers) is a fairly standard Llama-like operation

$$
\left(\text{silu}(x W_\text{gate}) \cdot (x W_\text{up}) \right) W_\text{down}
$$

so we have to choose a two-step matrix multiplication sharding strategy
($W_\text{gate}$ or $W_\text{up}$ then $W_\text{down}$). For low-latency
settings pure tensor-parallelism works well

$$
B D \overset{DF_{\color{red}x}}\rightarrow B F_{\color{red}x} \overset{F_{\color{red}x}D}{\rightarrow} {\color{red}{AR}} \rightarrow BD
$$

### MoE Layer in Inference

Since the expert layers in Deepseek V3 are small, $E_{=256}D_{=7168}F_{=2048}$
and TPUs work best when the minor-most dimensions is $\geq 128$, we can only shard
the expert matrices among $16$ devices mesh axis ($16 \cdot 128 = 2048$). For the other
axis in the mesh the only other sharde-able dimension is the expert dimension.

For an all-reduce matmul sharding strategy we end up with the following sharding:

$$
B D \overset{E_{\color{red}y} DF_{\color{red}z}}\rightarrow B F_{\color{red}z} \overset{E_{\color{red}y}F_{\color{red}z}D}{\rightarrow} {\color{red}{AR}_z} \rightarrow {\color{red}{AR}_y} \rightarrow BD
$$

<p align="center">

![Decode Profile](./images/ds_r1_decode_profile_bs8_ctx2048.png)

<p align="center">
Fig: Decode profile for an MoE Layer with batch size = 8 and context length of 2048 (not context limit).
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

### Creating a multi-host TPU VM

```bash
TPU_ZONE="zone, e.g. us-central1-a"
PROJECT="your-project"
IMAGE="v2-alpha-tpuv5-lite"
ACCELERATOR="v5litepod-64"
TPU_NAME="my_tpu"
TPU_NAME="$NAME_PREFIX"-"$ACCELERATOR"

gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$TPU_ZONE" \
  --project="$PROJECT" --accelerator-type="$ACCELERATOR" --version="$IMAGE"
```

### Setting up code & data

#### 1. [gcsfuse](https://cloud.google.com/storage/docs/cloud-storage-fuse/install#install-source-code)

For datasets and checkpoints.

```
gcsfuse --implicit-dirs {bucket_name_no_gs://} {local_folder}
```
#### 2. NFS

For code consistency between hosts in the TPU Pod / Cluster.

```bash
# on worker 0
WORKER0_IP="..."
sudo apt install -y nfs-server nfs-common net-tools tmux
mkdir -p ~/nfs; sudo umount ~/nfs
echo "$HOME/nfs $WORKER0_IP/24(rw,sync,no_subtree_check)" | sudo tee /etc/exports
sudo exportfs -a
sudo systemctl enable nfs-server; sudo systemctl restart nfs-server
sudo chown $USER:$USER -R ~/nfs
```

```bash
# on all other workers (!= 0)
SERVER_IP="..."
mkdir -p ~/nfs
sudo umount ~/nfs; sudo mount -t nfs $SERVER_IP:/home/$USER/nfs ~/nfs
```

#### (Optionally) 3. [sshfs](https://github.com/libfuse/sshfs)

For a quick preview from a local machine.

```bash
sshfs ~/local_folder TPU_WORKER_0_IP:~/remote_folder
```

### Utilities

```bash
TPU_NAME="..."
TPU_ZONE="..."
TPU_PROJECT="..."

tpu_exec() {
    local workers=$(seq $1 $2 | tr '\n' ',')
    gcloud alpha compute tpus tpu-vm ssh --zone="$TPU_ZONE" --project="$TPU_PROJECT" \
      "$TPU_NAME" --worker="$workers" --command="$2"
}
tpu_exec all 'pip install -U "jax[tpu]"'
```

### Starting the `ipyparallel` cluster

Start $N - 1$ workers (ipyparallel calls them `engines`) because we want worker 0 to execute interactively.

```bash
SERVER_IP="..."
CONTROLLER_SETUP=$(cat << EOM
tmux kill-session -t controller; pkill -9 python
tmux new -d -s controller '\
  . ~/venv/bin/activate && ipcontroller --profile-dir=~/nfs --ip=$SERVER_IP'
EOM
)

ENGINE_SETUP=$(cat << EOM
tmux kill-session -t engine; pkill -9 ipengine
tmux new -d -s engine '. ~/venv/bin/activate && ipengine --profile-dir=~/nfs'
EOM
)

tpu_exec 0 0  "$CONTROLLER_CMD"  # only worker 0
tpu_exec 1 15 "$ENGINE_CMD" # all workers except worker 0
```

#### Jupyter Notebook
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
- [ ] ragged decode MLA kernel
- [ ] further prefill throughput optimizations
- [ ] distilled models
