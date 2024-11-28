# create finetune.sh
cat > finetune.sh << EOL
#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3  # adjust based on your GPU setup

# Model configuration
PROMPT_VERSION=v1
MODEL_VERSION="qwen-7b"

# Launch training
deepspeed train_navigation.py \
    --deepspeed ./zero2.json \
    --model_name_or_path lmms-lab/llava-next-interleave-qwen-7b \
    --version $PROMPT_VERSION \
    --data_path ./data/navigation_dataset \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type linear \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/navigation \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
EOL

chmod +x finetune.sh