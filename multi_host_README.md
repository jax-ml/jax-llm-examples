# Multi-Host Cluster Setup for JAX LLM Examples

This document explains the concepts and workflow for running JAX-based LLMs on multi-host clusters, such as Google Cloud TPU Pods or multi-node GPU clusters.

## Overview

JAX offers [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) with a global view of computation, but the program must be run on multiple hosts, each controlling a subset of the actual accelerators.

The simplest way to run a JAX program on multiple hosts is to [run the same Python file from all the hosts at the same time](https://docs.jax.dev/en/latest/multi_process.html) (e.g., via ssh on all hosts). For development, it's often easier to:
1. Efficiently share code changes to all hosts
2. Easily launch computation on all hosts
3. Debug interactively

## Source of Truth: [misc/tpu_toolkit.sh](./misc/tpu_toolkit.sh)

All setup and management commands for multi-host clusters are provided in the [`misc/tpu_toolkit.sh`](./misc/tpu_toolkit.sh) script. **Please refer to this script for the most up-to-date and authoritative setup instructions.**

- The script includes functions for:
  - Creating and managing NFS shared folders
  - Setting up Python virtual environments
  - Launching and managing IPyParallel clusters
  - Utility functions for SSH, SCP, and cluster restarts

**Usage examples and configuration are documented in the comments at the top of the script.**

## Typical Workflow

1. **Edit and source `misc/tpu_toolkit.sh`**
2. **Set required environment variables** (`TPU_NAME`, `TPU_NUM_NODES`, `TPU_ZONE`, `TPU_PROJECT`)
3. **Run the setup functions** (see script for details)
4. **Use the provided functions to manage your cluster**

## Interactive and Notebook Usage

- For interactive development, you can use IPyParallel to connect a Jupyter notebook to the cluster. See the script and the [ipyparallel documentation](https://ipyparallel.readthedocs.io/en/latest/) for details.
- Example Python code for connecting to the cluster is included in the script comments.

## Troubleshooting

- **Import/package errors on engines:**
  - Ensure your virtualenv is activated before running Jupyter and on all engines.
  - Example check:
    ```python
    %%px --local
    import sys
    print(sys.executable)
    ```
- **Notebook hangs on `jax.distributed.initialize()`:**
  - You may have pre-existing Jupyter sessions or uncleared engines. Run `client[:].abort()` in your notebook, or restart the engines using the toolkit script.
- **Out-of-memory (OOM) errors:**
  - Pre-existing sessions may still have weights loaded. Clear sessions and try again.

---

For all setup and management commands, see [`misc/tpu_toolkit.sh`](./misc/tpu_toolkit.sh). 