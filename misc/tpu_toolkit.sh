#!/usr/bin/env zsh

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

# Some explicit assumptions in this script:
# - a debian-based system:
#   - apt
#   - nfs-server nfs-common net-tools packages
# - `tpu_ssh_exec` method (in this script) needs updating to a worker ssh cmd
# - local network is assumed to be the only interface on nodes matching '^e.*'
# - pip packages installation instructions are hard-coded
# - nfs folder is hardcoded to be `~/nfs`

# words of warning:
# - it's very easy to "hang" a VM by issuing an interactive `sudo apt`

################################################################################
# configuration ################################################################
################################################################################

# example:
#TPU_NAME="rdyro-v5e-16"
#TPU_NUM_NODES="4"  # number of hosts, not chips
#TPU_ZONE="..."  # e.g., "us-central1-a" or "europe-west4-b"
#TPU_PROJECT="..."  # your Google Cloud project

# extra args to ssh, MUST START WITH "--" or be empty, e.g., "-- -J my_proxy_jump_server"
SSH_EXTRA_ARGS=""
SSH_EXTRA_ARGS=(${(z)SSH_ARGS})

# Example usage:

# 1. Run common setup on the cluster setting up:
#     - an NFS shared folder under ~/nfs
#     - a standalone Python virtual environment for the project at ~/venv
#     - the IPyParallel cluster with a `bash ~/restart_cluster.sh`
#       with a `bash ~/restart_cluster.sh` script on node #0 for restarting it
#     - Google utilities: gcloud command and gcsfuse for weights and datasets
# $ tpu_setup_all

# 2. Connect to node #0 and start the jupyter server on it
#    - either by connecting with VSCode
#    - or by starting a jupyter server manually

# 3. In a notebook cell, connect to the IPyParallel cluster
# ```python
# from pathlib import Path
# import ipyparallel as ip
# client = ip.Client(str(Path("~/nfs/security/ipcontroller-engine.json").expanduser()))
# client.wait_for_engines(NODE_NUMBER - 1)  # this node is one of the engines
# ```

# 4. Alternatively, you can use `tpu_ssh_exec` function from this toolkit to
#    launch python scripts on all hosts
# $ tpu_ssh_exec ". ~/venv/bin/activate && cd ~/nfs && python3 main.py"


################################################################################
# public toolkit functions #####################################################
################################################################################

: "${TPU_NAME:?"TPU_NAME env variable cannot be empty"}"
: "${TPU_NUM_NODES:?"TPU_NUM_NODES env variable cannot be empty"}"
: "${TPU_ZONE:?"TPU_ZONE env variable cannot be empty"}"
: "${TPU_PROJECT:?"TPU_PROJECT env variable cannot be empty"}"

tpu_describe() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$TPU_ZONE" --project="$TPU_PROJECT" | rg externalIp | tee .tpu_ips.log
}

tpu_ssh_exec() {
    # execute a command on a worker or a set of workers
    # $ tpu_ssh_exec "$cmd_str" $start_worker $end_worker_inclusive
    local cmd="$1" worker_start_idx="${2:-0}" worker_end_idx="${3:-$((TPU_NUM_NODES - 1))}"
    local workers=$(echo $(seq $worker_start_idx $worker_end_idx) | sed 's/ /,/g')
    [[ $workers == "" ]] && return 0
    gcloud alpha compute tpus tpu-vm ssh --zone="$TPU_ZONE" --project="$TPU_PROJECT" \
      "$TPU_NAME" --worker="$workers" --command="$cmd" "${SSH_EXTRA_ARGS[@]}" | tee .ssh_output.log
    local _status=$?
    [[ $_status -ne 0 ]] && { echo "cmd=\"$cmd\" failed"; return $_status; } || return 0
}

tpu_ssh() {
    # ssh into a worker
    # $ tpu_ssh $worker_id
    local id="${1:-0}"
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --project="$TPU_PROJECT" --zone="$TPU_ZONE" --worker="$id" "${SSH_EXTRA_ARGS[@]}"
}

tpu_scp() {
    # copy files from or to a worker (automatically recursive, i.e. works for directories)
    # $ tpu_scp $worker_id $from $to_dest
    local id="${1:-0}"
    local from="${2:?"Provide a from location"}"
    local to="${3:?"Provide a to destination"}"
    gcloud alpha compute tpus tpu-vm scp "$from" "$to" \
      --project="$TPU_PROJECT" --zone="$TPU_ZONE" --worker="$id" --recurse "${SSH_EXTRA_ARGS[@]}"
}

tpu_restart_cluster() {
  tpu_ssh_exec "bash restart_cluster.sh" 0 0
}

tpu_setup_all() {  # combined setup function
  _tpu_setup_nfs || { echo "tpu_setup_nfs failed"; return 1; }
  _tpu_setup_python || { echo "_tpu_setup_python failed"; return 1; }
  _tpu_setup_cluster || { echo "_tpu_setup_cluster failed"; return 1; }
  _tpu_setup_gcloud_and_gcsfuse || { echo "_tpu_setup_gcloud_and_gcsfuse failed"; return 1; }
}

tpu_create() {
  # for image selection see: https://cloud.google.com/tpu/docs/v5e and https://cloud.google.com/tpu/docs/v6e
  # $ tpu_create v5litepod-64
  local tpu_accel="${1:?"Provide the TPU accelerator type to create, e.g. v5litepod-64, v6e-8"}"
  local tpu_image="${2:-}"
  if [[ "$tpu_accel" == v5litepod* ]]; then
    tpu_image="v2-alpha-tpuv5-lite" && echo "Guessing the TPU VM image to be '$tpu_image'"
  elif [[ "$tpu_accel" == v6e* ]]; then
    tpu_image="v2-alpha-tpuv6e" && echo "Guessing the TPU VM image to be '$tpu_image'"
  fi
  [[ "$tpu_image" == "" ]] && echo "You must specify the VM image explicitly for accelerator-type=$tpu_accel" && return 1

  gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$TPU_ZONE" --project="$TPU_PROJECT" --accelerator-type="$tpu_accel" \
    --version="$tpu_image" --spot
  tpu_describe #  run describe to get external ips
}

tpu_delete() {
  echo "Deleting $TPU_NAME"
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$TPU_ZONE" --project="$TPU_PROJECT" --quiet
}

################################################################################
# helper functions follow, no need to call them manually #######################
################################################################################

# specific setup routines ######################################################
_tpu_setup_nfs() {
  # update uid and gid for the user to match across tpu nodes (for NFS) ########
  local USERNAME=$(tpu_ssh_exec "whoami" 0 0)
#sudo screen -d -m bash -c "pkill -u $USERNAME; usermod -u 31003 $USERNAME;
  local CHANGE_USER_UID_GID=$(cat << EOM
sudo tmux new -d bash -c "pkill -u $USERNAME; usermod -u 31005 $USERNAME; 
  groupmod -g 31005 $USERNAME; chown $USERNAME:$USERNAME -R /home/$USERNAME"
EOM
)

  # change user UID and GID
  echo "\nUpdating user UID and GID -------------------------------------------"
  tpu_ssh_exec "$CHANGE_USER_UID_GID"
  tpu_ssh_exec 'echo "$(hostname) user=$(whoami) uid=$(id -u) gid=$(id -g)"'

  # setup NFS
  echo "\nSetting up NFS under ~/nfs ------------------------------------------"

  local NFS_SETUP_ALL=$(cat << 'EOM'
set -x
sudo pkill -9 unattended-upgr || true
sudo dpkg --configure -a
sudo DEBIAN_FRONTEND=noninteractive apt update -y
sudo pkill -9 unattended-upgr || true
#sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt install -y nfs-server nfs-common \
  net-tools tmux
echo '#!/bin/bash' > ~/print_ip.sh
echo "sudo ifconfig | grep '^e' -A 1 | awk '{ if (\$1 == \"inet\") print \$2}'" \
  >> ~/print_ip.sh
EOM
)

  local NFS_SERVER_SETUP=$(cat << 'EOM'
set -xe
IP=$(sh ~/print_ip.sh); mkdir -p ~/nfs || true; sudo umount ~/nfs || true
echo "$HOME/nfs $IP/16(rw,sync,no_subtree_check)" | sudo tee /etc/exports
sudo exportfs -a; sudo systemctl enable nfs-server; sudo systemctl restart nfs-server
sudo chown $USER:$USER -R ~/nfs
EOM
)

  tpu_ssh_exec "$NFS_SETUP_ALL"
  tpu_ssh_exec "$NFS_SERVER_SETUP" 0 0
  tpu_ssh_exec "sh ~/print_ip.sh" 0 0 | tee .node0_ip.txt  # get the worker 0 IP
  local NODE0_IP=$(cat .node0_ip.txt)

  local NFS_CLIENT_SETUP=$(cat << EOM
set -xe
mkdir -p ~/nfs; sudo umount ~/nfs || true; echo $NODE0_IP > node0_ip.txt
sudo mount -t nfs -vvvv $NODE0_IP:/home/\$USER/nfs ~/nfs
EOM
)
  tpu_ssh_exec "$NFS_CLIENT_SETUP" 1
}

# setup matching python 3.12 version ###########################################
_tpu_setup_python() {
  echo "\nInstalling Python 3.12 ----------------------------------------------"

  local INSTALL_PYTHON=$(cat << 'EOM'
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
python3 -m pip install -U uv; python3 -m uv venv --seed --python 3.12 "$HOME/venv"
. "$HOME/venv/bin/activate"
echo ". ~/venv/bin/activate" >> ~/.bashrc; echo ". ~/venv/bin/activate" >> ~/.zshrc
pip install -U uv pip && uv pip install -U "jax[tpu]" ipyparallel ipykernel ipywidgets
EOM
)
  tpu_ssh_exec "$INSTALL_PYTHON"
}

# setup ssh cluster control from node 0 to other nodes #########################
_RESTART_CLUSTER_SCRIPT=$(cat << 'EOM'
#!/usr/bin/env bash

num_hosts=REPLACE_TPU_NUM_NODES
hostprefix=$(hostname | sed 's/[0-9]\+$//')  # hosts end with a -{number} suffix

# stop the cluster #############################################################
tmux kill-session -t controller || true
pkill -9 'python3|python|ipcontroller' || true

tmux kill-session -t controller || true
pkill -9 ipcontroller || true
for i in $(seq 1 $(( num_hosts - 1)) ); do
  ssh -o "StrictHostKeyChecking=no" "$hostprefix""$i" \
    "tmux kill-session -t engine || true; pkill -9 ipengine || true" &
done
wait

# start the cluster again ######################################################
tmux new -d -s controller '. ~/venv/bin/activate \
  && ipcontroller --profile-dir=~/nfs --ip=$(sh ~/print_ip.sh) 2>&1 \
  | tee ipcontroller_log.txt'

sleep 2  # give the controller a moment to stand up

for i in $(seq 1 $(( num_hosts - 1 )) ); do
  ssh -o "StrictHostKeyChecking=no" "$hostprefix""$i" \
    "tmux new -d -s engine '. ~/venv/bin/activate && \
    ipengine --profile-dir=~/nfs 2>&1 | tee ipengine_log.txt'" &
done
wait
EOM
)

_tpu_setup_cluster() {
  echo "$_RESTART_CLUSTER_SCRIPT" > /tmp/restart_cluster.sh
  sed -i "s/REPLACE_TPU_NUM_NODES/$TPU_NUM_NODES/g" /tmp/restart_cluster.sh
  tpu_scp 0 /tmp/restart_cluster.sh $TPU_NAME:~/restart_cluster.sh

  GENERATE_KEY_CMD=$(cat << 'EOM'
[[ ! -f ~/.ssh/id_ed25519.pub ]] && mkdir -p ~/.ssh && ssh-keygen -q -N '' -f ~/.ssh/id_ed25519 -t ed25519
cat ~/.ssh/id_ed25519.pub
EOM
)
  ead_key=$(tpu_ssh_exec "$GENERATE_KEY_CMD" 0 0)
  SET_SSH_KEY=$(cat << EOM
  mkdir -p ~/.ssh
  echo "$ead_key" >> ~/.ssh/authorized_keys
EOM
)
  tpu_ssh_exec "$SET_SSH_KEY" 0
  tpu_ssh_exec "bash restart_cluster.sh" 0 0  # start the cluster
}

# setup gcsfuse for remote datasets and weights ################################
_tpu_setup_gcloud_and_gcsfuse() {
  local GCLOUD_INSTALL=$(cat << 'EOM'
cd $HOME && mkdir -p googlestuff && cd googlestuff
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x87_64.tar.gz
yes | ./google-cloud-sdk/install.sh
wget https://go.dev/dl/go1.24.3.linux-amd64.tar.gz && tar -xvf go1.24.3.linux-amd64.tar.gz
sudo rm -f /usr/bin/go && sudo ln -sf $(realpath go/bin/go) /usr/bin/go  # latest go
go install github.com/googlecloudplatform/gcsfuse/v2@master
sudo ln -sf $(realpath ~/go/bin/gcsfuse) /usr/bin/gcsfuse
cd $HOME
EOM
)
  tpu_ssh_exec "$GCLOUD_INSTALL"
}
