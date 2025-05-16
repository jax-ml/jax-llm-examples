export TPU_ZONE="us-central1-a"
export TPU_PROJECT={YOUR_PROJECT_ID}
export IMAGE={YOUR_HARDWARE_TYPE eg v2-alpha-tpuv5-lite}
export ACCELERATOR="v5litepod-64"
export TPU_NAME="$USER-v5e-64" # TPU name, change to your own
export NUM_WORKERS=16
export VM_USER={$USER_google_com if oslogin is enabled, else $USER}
export WORKER0_IP={TPU_VM_INTERNAL_IP, e.g. '10.128.0.76'}

export LOCAL_DATA_FOLDER='data'
export REMOTE_DATA_BUCKET='lancewang-llm-data' # GCS bucket for storing checkpoints, name only, no gs:// prefix

# TPU execution utility function
tpu_exec() {
    local workers=$(seq $1 $2 | tr '\n' ',')
    gcloud alpha compute tpus tpu-vm ssh --zone="$TPU_ZONE" --project="$TPU_PROJECT" \
      "$TPU_NAME" --worker="$workers" --command="$3" -- -o ProxyCommand='corp-ssh-helper -relay=mtv5.r.ext.google.com %h %p'
}
