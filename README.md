deepspeed train_navigation.py \
    --deepspeed_config ds_config.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4