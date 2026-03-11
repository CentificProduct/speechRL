#!/bin/bash
# Emotion-GSRM Training Command
# Generated: 2026-03-11 15:42:33

swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model_revision main \
  --torch_dtype bfloat16 \
  --dataset /home/azureuser/atik/speechRL_backup/emotion_gsrm_checkpoints/swift_train.jsonl \
  --max_length 4096 \
  --learning_rate 2e-05 \
  --num_train_epochs 100 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --seed 42 \
  --output_dir /home/azureuser/atik/speechRL_backup/emotion_gsrm_checkpoints \
  --logging_dir /home/azureuser/atik/speechRL_backup/emotion_gsrm_logs \
  --save_strategy epoch \
  --save_total_limit 3 \
  --logging_steps 10 \
  --bf16 true \
  --train_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --val_dataset /home/azureuser/atik/speechRL_backup/emotion_gsrm_checkpoints/val_swift_train.jsonl \
  --eval_strategy epoch
