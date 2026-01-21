# Run with $ CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_VIP5.sh 4 [toys, clothing, beauty, sports] 13579 vitb32 2 8 20
# Memory-optimized version with fp16 and gradient accumulation
#
# Environment variables for customization:
#   BATCH_SIZE=4      - per-GPU batch size (default: 4, reduce if OOM)
#   GRAD_ACCUM=8      - gradient accumulation steps (default: 8)
#   MAX_TEXT_LEN=256  - max sequence length (default: 256)
#   USE_FP16=1        - enable fp16 (default: 1, set to 0 to disable)
#   LR=5e-4           - learning rate (default: 5e-4 for fp16, 1e-3 for fp32)
#
# Example without fp16 (if NaN issues persist):
#   USE_FP16=0 LR=1e-3 BATCH_SIZE=2 bash scripts/train_VIP5.sh ...
#
# If still OOM, try: BATCH_SIZE=2 GRAD_ACCUM=16 MAX_TEXT_LEN=128

#!/bin/bash

split=$2
img_feat_type=$4
img_feat_size_ratio=$5
name=$split-$img_feat_type-$img_feat_size_ratio-$6-$7
output=snap/$name

# Memory optimization: Use smaller batch size with gradient accumulation
# Effective batch size = batch_size * gradient_accumulation_steps * num_gpus
# With batch_size=4, grad_accum=8, 4 GPUs: effective = 4 * 8 * 4 = 128
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_TEXT_LEN=${MAX_TEXT_LEN:-256}
USE_FP16=${USE_FP16:-1}

# Lower learning rate for fp16 to avoid numerical instability
if [ "$USE_FP16" = "1" ]; then
    LR=${LR:-5e-4}
    FP16_FLAG="--fp16"
else
    LR=${LR:-1e-3}
    FP16_FLAG=""
fi

echo "Training config: BATCH_SIZE=$BATCH_SIZE, GRAD_ACCUM=$GRAD_ACCUM, LR=$LR, FP16=$USE_FP16"

# Validate GPU setup
NUM_GPUS=$1
VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Requested GPUs: $NUM_GPUS, Visible GPUs: $VISIBLE_GPUS"

if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "ERROR: Requested $NUM_GPUS GPUs but only $VISIBLE_GPUS are visible!"
    echo "Check your CUDA_VISIBLE_DEVICES setting."
    echo "Current CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    exit 1
fi

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port $3 \
    src/train.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train $split \
        --valid $split \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --optim adamw \
        --warmup_ratio 0.1 \
        --lr $LR \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'sequential,direct,explanation' \
        --backbone '/home/mlsnrs/.cache/huggingface/hub/t5-base' \
        --output $output \
        --epoch $7 \
        --use_adapter \
        --unfreeze_layer_norms \
        --reduction_factor $6 \
        --use_single_adapter \
        --max_text_length $MAX_TEXT_LEN \
        --gen_max_length 64 \
        --image_feature_type $img_feat_type \
        --image_feature_size_ratio $img_feat_size_ratio \
        --whole_word_embed \
        --category_embed \
        $FP16_FLAG 2>&1 | tee log/$name.log
