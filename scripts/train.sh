accelerate launch train.py \
    --seed=123 \
    --experience_name="DiffSkelet_training" \
    --data_root="" \
    --output_dir="" \
    --report_to="tensorboard" \
    --resolution=128 \
    --channel_attn=True \
    --train_batch_size=16 \
    --perceptual_coefficient=0.01 \
    --skelet_coefficient=0.5 \
    --max_train_steps=200000 \
    --ckpt_interval=20000 \
    --gradient_accumulation_steps=1 \
    --log_interval=100 \
    --learning_rate=1e-5 \
    --lr_scheduler="linear" \
    --lr_warmup_steps=10000 \
    --mixed_precision="no"
    