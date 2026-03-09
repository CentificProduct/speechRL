"""
Emotion-GSRM: Training Configuration
======================================
All hyperparameters from the proposal:
  - Model: Qwen2.5-Omni-7B
  - Data: 5K-7K CoT synthesis samples
  - LR: 2x10^-5
  - Batch size: 32
  - Epochs: 10
  - Framework: SWIFT
  - Inference: K=16 samples averaged at temperature 1.0, top-p 0.6
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TrainingFramework(str, Enum):
    SWIFT = "swift"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class LoRATarget(str, Enum):
    ALL_LINEAR = "all_linear"
    QKV_PROJ = "qkv_proj"
    ATTENTION_ONLY = "attention_only"
    CUSTOM = "custom"


@dataclass
class TrainingConfig:
    """
    Complete training configuration for Emotion-GSRM SFT.
    Default values match the proposal specifications exactly.
    """

    # ---- Model ----
    model_name: str = "Qwen/Qwen2.5-Omni-7B"
    model_revision: str = "main"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"

    # ---- Data ----
    train_data_path: str = "./cot_synthesis_output/sft_training_data.jsonl"
    val_data_path: str = ""
    val_split_ratio: float = 0.05
    max_length: int = 4096
    audio_max_duration: float = 30.0

    # ---- Training hyperparameters (from proposal) ----
    learning_rate: float = 2e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    per_device_batch_size: int = 4
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42

    # ---- LoRA / PEFT ----
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target: LoRATarget = LoRATarget.ALL_LINEAR
    lora_target_modules: Optional[list[str]] = None

    # ---- Framework ----
    framework: TrainingFramework = TrainingFramework.SWIFT
    output_dir: str = "./emotion_gsrm_checkpoints"
    logging_dir: str = "./emotion_gsrm_logs"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    logging_steps: int = 10

    # ---- Distributed training ----
    num_gpus: int = 4
    deepspeed_config: str = ""
    bf16: bool = True
    fp16: bool = False

    # ---- Inference (from proposal) ----
    inference_k: int = 16
    inference_temperature: float = 1.0
    inference_top_p: float = 0.6
    inference_max_new_tokens: int = 2048

    # ---- Callbacks & monitoring ----
    use_wandb: bool = True
    wandb_project: str = "emotion-gsrm"
    wandb_run_name: str = ""
    early_stopping_patience: int = 3
    early_stopping_metric: str = "eval_loss"

    def to_swift_args(self) -> dict:
        """Convert to SWIFT sft command arguments."""
        args = {
            "model": self.model_name,
            "model_revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "dataset": self.train_data_path,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "save_strategy": self.save_strategy,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "bf16": self.bf16,
        }

        if self.use_lora:
            args.update({
                "sft_type": "lora",
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            })
            if self.lora_target == LoRATarget.ALL_LINEAR:
                args["lora_target_modules"] = "ALL"
            elif self.lora_target_modules:
                args["lora_target_modules"] = self.lora_target_modules
        else:
            args["sft_type"] = "full"

        if self.val_data_path:
            args["val_dataset"] = self.val_data_path
            args["eval_strategy"] = self.eval_strategy

        if self.deepspeed_config:
            args["deepspeed"] = self.deepspeed_config

        return args

    def to_swift_cli(self) -> str:
        """Generate the SWIFT CLI command string for training."""
        args = self.to_swift_args()
        parts = ["swift sft"]
        for key, value in args.items():
            if isinstance(value, bool):
                # SWIFT CLI uses --flag or --flag=false format, not --flag true
                if value:
                    parts.append(f"  --{key}")
                else:
                    parts.append(f"  --{key}=false")
            elif isinstance(value, list):
                parts.append(f"  --{key} {' '.join(str(v) for v in value)}")
            else:
                parts.append(f"  --{key} {value}")
        return " \\\n".join(parts)

    def to_hf_training_args(self) -> dict:
        """Convert to HuggingFace TrainingArguments dict."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_batch_size,
            "per_device_eval_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "evaluation_strategy": self.eval_strategy,
            "save_strategy": self.save_strategy,
            "save_total_limit": self.save_total_limit,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "seed": self.seed,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "report_to": "wandb" if self.use_wandb else "none",
            "run_name": self.wandb_run_name or "emotion-gsrm-sft",
            "load_best_model_at_end": True,
            "metric_for_best_model": self.early_stopping_metric,
        }

    def validate(self) -> list[str]:
        """Validate config and return list of warnings."""
        warnings = []
        effective_batch = (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * self.num_gpus
        )
        if effective_batch != self.batch_size:
            warnings.append(
                f"Effective batch size ({effective_batch}) differs from "
                f"target ({self.batch_size}). Adjust per_device_batch_size, "
                f"gradient_accumulation_steps, or num_gpus."
            )
        if self.learning_rate > 1e-4:
            warnings.append(f"LR {self.learning_rate} is high for SFT.")
        if self.max_length < 2048:
            warnings.append(f"max_length={self.max_length} may be too short.")
        if not self.use_lora and self.num_gpus < 4:
            warnings.append("Full fine-tuning of 7B needs 4+ GPUs. Use LoRA.")
        return warnings