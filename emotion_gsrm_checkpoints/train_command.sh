#!/bin/bash
# Emotion-GSRM Training Command
# Generated: 2026-03-09 14:02:47

swift sft \
  --model Qwen/Qwen2.5-Omni-7B-Instruct \
  --model_revision main \
  --torch_dtype bfloat16 \
  --dataset emotion_gsrm_checkpoints/swift_train.jsonl \
  --max_length 4096 \
  --learning_rate 2e-05 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --seed 42 \
  --output_dir ./emotion_gsrm_checkpoints \
  --logging_dir ./emotion_gsrm_logs \
  --save_strategy epoch \
  --save_total_limit 3 \
  --logging_steps 10 \
  --bf16 true \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --target_modules all
