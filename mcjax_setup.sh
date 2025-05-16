# Load the TPU configuration
source mcjax_defs.sh

# Creat TPU VM
gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$TPU_ZONE" \
  --project="$PROJECT" --accelerator-type="$ACCELERATOR" --version="$IMAGE" \
  --metadata=enable-oslogin=TRUE

# Install required packages and virtualenv for all the nodes.
INSTALL_COMMAND=$(cat << EOM
  rm ./venv -rf
  sudo pkill -9 apt
  sudo dpkg --configure -a
  sudo apt update
  sudo apt install -y nfs-common nfs-kernel-server nfs-server net-tools tmux python3-ipyparallel
  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
  export PATH="\$HOME/.local/bin:\$PATH"
  uv python install 3.10 && uv venv --python 3.10
  source .venv/bin/activate
  uv pip install -U "jax[tpu]" ipyparallel
EOM
)

tpu_exec 0 15 "$INSTALL_COMMAND"

# Install jupyter, ipykernel and NFS for node 0
INSTALL_JUPYTER_COMAND=$(cat << EOM
  source .venv/bin/activate
  pip uninstall -y jupyter ipykernel
  pip install jupyter ipykernel
  # Create a new kernel for Jupyter to include all the dependencies
  python -m ipykernel install --user --name uv310 --display-name "Python (uv310)"

  # Setup NFS on node 0
  mkdir -p ~/nfs; sudo umount ~/nfs
  echo "\$HOME/nfs $WORKER0_IP/24(rw,sync,no_subtree_check)" | sudo tee /etc/exports
  sudo exportfs -a
  sudo systemctl enable nfs-server; sudo systemctl restart nfs-server
  sudo chown \$USER:\$USER -R ~/nfs

  # Setup gcsfuse for checkpointing
  mkdir $LOCLA_FOLDER
  gcsfuse --implicit-dirs $REMOTE_DATA_BUCKET $LOCLA_FOLDER
EOM
)

tpu_exec 0 0 "$INSTALL_JUPYTER_COMAND" # only worker 0

MOUNT_COMMAND=$(cat << EOM
  mkdir -p ~/nfs
  sudo umount ~/nfs
  # VM_USER should be username in your TPU VM and should be the same across all VM workers.
  sudo mount -t nfs $WORKER0_IP:/home/\$VM_USER/nfs ~/nfs
EOM
)
tpu_exec 1 15 "$MOUNT_COMMAND"

CONTROLLER_SETUP=$(cat << EOM
tmux kill-session -t controller; pkill -9 python;
sleep 5
tmux new -d -s controller '\
  . ~/.venv/bin/activate && ipcontroller --profile-dir=~/nfs --ip=$WORKER0_IP'
tmux ls
EOM
)

ENGINE_SETUP=$(cat << EOM
tmux kill-session -t engine; pkill -9 ipengine
sleep 5
tmux new -d -s engine 'source ~/.venv/bin/activate && ipengine --profile-dir=~/nfs'
tmux ls
EOM
)

tpu_exec 0 0  "$CONTROLLER_SETUP"  # only worker 0
sleep 10
tpu_exec 1 15 "$ENGINE_SETUP" # all workers except worker 0
