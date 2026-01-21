# Run with $ CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_VIP5.sh 4 [toys, clothing, beauty, sports] 13579 vitb32 2 8 20
# Memory-optimized version with fp16 and gradient accumulation

#!/bin/bash

split=$2
img_feat_type=$4
img_feat_size_ratio=$5
name=$split-$img_feat_type-$img_feat_size_ratio-$6-$7
output=snap/$name

# Memory optimization: Use smaller batch size with gradient accumulation
# Effective batch size = batch_size * gradient_accumulation_steps * num_gpus
# With batch_size=8, grad_accum=4, 7 GPUs: effective = 8 * 4 * 7 = 224 (similar to 36 * 7 = 252)
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_TEXT_LEN=${MAX_TEXT_LEN:-512}

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
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 5.0 \
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
        --fp16 2>&1 | tee log/$name.log
