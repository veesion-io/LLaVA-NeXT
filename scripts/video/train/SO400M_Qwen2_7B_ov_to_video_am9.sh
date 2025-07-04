#!/bin/bash
set -x

source env/bin/activate

export CPATH="/usr/local/cuda/include:$CPATH"

# Set up the data folder
# All video and track files should be downloaded from the scalable-training-dataset-us-east-1 S3 bucket
# No local paths should be used for video/track files
# JSON dataset should be downloaded from the original bucket
#IMAGE_FOLDER="/ssd2/kchoi/experiments/DLT-138/Simone_28_one_video_one_label/videos/"
#VIDEO_FOLDER="/ssd2/kchoi/experiments/DLT-138/Simone_28_one_video_one_label/videos/"
#IMAGE_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments_resampled/"
#VIDEO_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments_resampled/"
#IMAGE_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments/"
#VIDEO_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments/"
# DATA_YAML="scripts/video/train/exp.yaml" # e.g exp.yaml
mkdir -p data
aws s3 cp s3://scalable-training-dataset/gemini_fine_tuning/32k/gemini_finetuning_subset_cheating_description.json data/gemini_finetuning_subset_cheating_description.json
DATA_YAML="data/gemini_finetuning_subset_cheating_description.json"

############### Prepare Envs #################
# Install CUDA development headers and ninja for DeepSpeed JIT compilation
sudo yum install -y cuda-devel-12-6
python3 -m pip install ninja
python3 -m pip install flash-attn --no-build-isolation
alias python=python3

# H100-optimized NCCL settings for better performance
export CPATH="/usr/local/cuda/include:$CPATH"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="$NCCL_INTERFACE"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2  # Enhanced GPU Direct RDMA for H100
export NCCL_P2P_LEVEL=NVL    # NVLink for P5en instances
export NCCL_NVLS_ENABLE=1    # Enable NVLink SHARP for H100
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Torch Inductor optimization settings for H100
export TORCH_INDUCTOR_CACHE_DIR="/tmp/inductor_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache"
export TORCH_COMPILE_DEBUG=0
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
############### Show Envs ####################

nvidia-smi

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
#

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9_vision_focused"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Set default values for multi-node training if not provided
export NODE_RANK=${NODE_RANK:-0}
export GPU_INSTANCES_NUMBER=${GPU_INSTANCES_NUMBER:-8}
export GPU_COUNT=${GPU_COUNT:-1}
export MASTER_PRIVATE_IP=${MASTER_PRIVATE_IP:-127.0.0.1}

echo "Using NODE_RANK: ${NODE_RANK}"
echo "Using GPU_INSTANCES_NUMBER: ${GPU_INSTANCES_NUMBER}"
echo "Using GPU_COUNT: ${GPU_COUNT}"
echo "Using MASTER_PRIVATE_IP: ${MASTER_PRIVATE_IP}"

# Pass SSH options to deepspeed's underlying pdsh/ssh
export PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=no"

# Ensure .ssh directory exists
mkdir -p ~/.ssh

# deepspeed \
#     --master_port 1234 \
#     --num_nodes ${GPU_INSTANCES_NUMBER} \
#     --node_rank ${NODE_RANK} \
#     --num_gpus ${GPU_COUNT} \
#     --master_addr ${MASTER_PRIVATE_IP} \
#     --hostfile ~/hostfile \
ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nnodes="${GPU_INSTANCES_NUMBER}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPU_COUNT}" \
    --master_addr="${MASTER_PRIVATE_IP}" \
    --master_port=1234 \
  llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --mm_tunable_parts="mm_vision_tower,mm_language_model" \
    --mm_vision_tower_lr=1e-5 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "'(1x1),...,(6x6)'" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --eval_dataset_size 128 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --optim "adamw_torch_fused" \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --dataloader_drop_last True \
    --frames_upbound 40 \
    --video_fps 5 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --mm_spatial_pool_stride 2 \
    --verbose_logging \
    --report_to tensorboard \
    --attn_implementation "flash_attention_2"
#    --force_sample False \
exit 0;

