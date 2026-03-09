"""
Emotion-GSRM: SFT Trainer
===========================
Orchestrates fine-tuning of Qwen2.5-Omni-7B using the SWIFT framework
or HuggingFace Trainer as fallback.

Two training paths:
  1. SWIFT CLI (recommended): Generates and executes swift sft command
  2. HuggingFace Trainer: Programmatic training with custom callbacks
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from sft.config import TrainingConfig, TrainingFramework
from sft.dataset import SFTDataset, SFTCollator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------------

class EmotionGSRMCallback:
    """
    Custom callback for monitoring emotion-specific training metrics.

    Tracks per-dimension score prediction accuracy during evaluation
    and logs to wandb if enabled.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.epoch_metrics: list[dict] = []

    def on_evaluate(self, eval_results: dict, epoch: int) -> None:
        """Called after each evaluation step."""
        metrics = {
            "epoch": epoch,
            "eval_loss": eval_results.get("eval_loss", 0.0),
            "timestamp": time.time(),
        }
        self.epoch_metrics.append(metrics)

        logger.info(
            f"Epoch {epoch}: eval_loss={metrics['eval_loss']:.4f}"
        )

    def on_train_end(self, final_metrics: dict) -> None:
        """Called at end of training."""
        output_path = Path(self.config.output_dir) / "training_metrics.json"
        with open(output_path, "w") as f:
            json.dump({
                "config": {
                    "model": self.config.model_name,
                    "lr": self.config.learning_rate,
                    "epochs": self.config.num_epochs,
                    "batch_size": self.config.batch_size,
                },
                "epoch_metrics": self.epoch_metrics,
                "final_metrics": final_metrics,
            }, f, indent=2)

        logger.info(f"Training metrics saved to {output_path}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EmotionGSRMTrainer:
    """
    Main trainer for Emotion-GSRM SFT.

    Supports two training backends:
      1. SWIFT (default): CLI-based, handles Qwen multimodal natively
      2. HuggingFace: Programmatic, more customizable

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    dataset : SFTDataset, optional
        Pre-loaded dataset. If None, loads from config.train_data_path.
    """

    def __init__(
        self,
        config: TrainingConfig,
        dataset: Optional[SFTDataset] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.callback = EmotionGSRMCallback(config)

        # Validate config
        warnings = config.validate()
        for w in warnings:
            logger.warning(f"Config warning: {w}")

    def train(self) -> dict:
        """
        Run training using the configured framework.

        Returns training metrics dict.
        """
        if self.config.framework == TrainingFramework.SWIFT:
            return self._train_swift()
        elif self.config.framework == TrainingFramework.HUGGINGFACE:
            return self._train_hf()
        else:
            return self._train_custom()

    # ---- SWIFT Training Path ----

    def _train_swift(self) -> dict:
        """
        Train using SWIFT CLI.

        Steps:
          1. Prepare dataset in SWIFT format
          2. Generate CLI command
          3. Execute training
          4. Collect metrics
        """
        logger.info("=== SWIFT Training Pipeline ===")

        # Step 1: Prepare data
        if self.dataset is None:
            self.dataset = SFTDataset(
                data_path=self.config.train_data_path,
                max_length=self.config.max_length,
                val_split=self.config.val_split_ratio,
            ).load()

        swift_data_path = Path(self.config.output_dir) / "swift_train.jsonl"
        swift_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_swift_jsonl(swift_data_path)

        # Update config with actual data path
        self.config.train_data_path = str(swift_data_path)

        # Val data path
        if self.dataset.val_samples:
            val_path = swift_data_path.parent / f"val_{swift_data_path.name}"
            self.config.val_data_path = str(val_path)

        # Step 2: Generate CLI command
        cli_command = self.config.to_swift_cli()
        logger.info(f"SWIFT command:\n{cli_command}")

        # Save command for reproducibility
        cmd_path = Path(self.config.output_dir) / "train_command.sh"
        with open(cmd_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Emotion-GSRM Training Command\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            if self.config.num_gpus > 1:
                f.write(
                    f"CUDA_VISIBLE_DEVICES="
                    f"{','.join(str(i) for i in range(self.config.num_gpus))} "
                )
            f.write(cli_command + "\n")

        logger.info(f"Training command saved to {cmd_path}")

        # Step 3: Execute
        return self._execute_swift(cli_command)

    def _execute_swift(self, command: str) -> dict:
        """Execute SWIFT training command."""
        # Build actual shell command
        shell_cmd = command.replace(" \\\n", " ")

        if self.config.num_gpus > 1:
            gpu_list = ",".join(str(i) for i in range(self.config.num_gpus))
            shell_cmd = f"CUDA_VISIBLE_DEVICES={gpu_list} {shell_cmd}"

        logger.info("Starting SWIFT training...")
        start_time = time.time()

        try:
            process = subprocess.Popen(
                shell_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            output_lines = []
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                if not line:
                    continue
                output_lines.append(line)
                logger.info(f"[SWIFT] {line}")

            process.wait()
            elapsed = time.time() - start_time

            if process.returncode == 0:
                logger.info(
                    f"SWIFT training completed in {elapsed / 3600:.1f} hours"
                )
                return {
                    "status": "success",
                    "elapsed_hours": elapsed / 3600,
                    "output_dir": self.config.output_dir,
                }
            else:
                logger.error(
                    f"SWIFT training failed (return code {process.returncode})"
                )
                return {
                    "status": "failed",
                    "return_code": process.returncode,
                    "last_output": output_lines[-20:] if output_lines else [],
                }

        except FileNotFoundError:
            logger.error(
                "SWIFT not found. Install with: pip install ms-swift"
            )
            return {"status": "error", "message": "SWIFT not installed"}

    # ---- HuggingFace Training Path ----

    def _train_hf(self) -> dict:
        """
        Train using HuggingFace Trainer.

        Programmatic alternative to SWIFT for more control.
        """
        logger.info("=== HuggingFace Training Pipeline ===")

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                AutoProcessor,
                Trainer,
                TrainingArguments,
                EarlyStoppingCallback,
            )
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return {"status": "error", "message": str(e)}

        # Load dataset
        if self.dataset is None:
            self.dataset = SFTDataset(
                data_path=self.config.train_data_path,
                max_length=self.config.max_length,
                val_split=self.config.val_split_ratio,
            ).load()

        # Load model and tokenizer
        logger.info(f"Loading model: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.torch_dtype, torch.bfloat16),
            device_map="auto",
        )

        # Apply LoRA if configured
        if self.config.use_lora:
            logger.info(
                f"Applying LoRA: rank={self.config.lora_rank}, "
                f"alpha={self.config.lora_alpha}"
            )
            target_modules = None
            if self.config.lora_target_modules:
                target_modules = self.config.lora_target_modules

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Setup processor/collator
        try:
            processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
        except Exception:
            processor = None

        collator = SFTCollator(
            tokenizer=tokenizer,
            processor=processor,
            max_length=self.config.max_length,
        )

        # Convert dataset to HF format
        train_dataset = self.dataset.to_hf_dataset()
        val_dataset = None
        if self.dataset.val_samples:
            # Quick conversion of val samples
            val_records = []
            for s in self.dataset.val_samples:
                msgs = s["messages"]
                val_records.append({
                    "system": msgs[0]["content"],
                    "input": msgs[1]["content"],
                    "output": msgs[2]["content"] if len(msgs) > 2 else "",
                    "audio_path": s.get("audios", [""])[0],
                    "utterance_id": s.get("utterance_id", ""),
                })
            from datasets import Dataset
            val_dataset = Dataset.from_list(val_records)

        # Training arguments
        training_args = TrainingArguments(**self.config.to_hf_training_args())

        # Callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0 and val_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting HuggingFace training...")
        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

        # Save
        trainer.save_model()
        trainer.save_state()

        metrics = train_result.metrics
        metrics["elapsed_hours"] = elapsed / 3600

        self.callback.on_train_end(metrics)

        logger.info(
            f"Training complete in {elapsed / 3600:.1f} hours. "
            f"Final loss: {metrics.get('train_loss', 'N/A')}"
        )

        return {
            "status": "success",
            "metrics": metrics,
            "output_dir": self.config.output_dir,
        }

    # ---- Custom Training Path ----

    def _train_custom(self) -> dict:
        """Placeholder for custom training loops."""
        logger.warning("Custom training not yet implemented")
        return {"status": "not_implemented"}

    # ---- Utilities ----

    def prepare_data_only(self) -> Path:
        """
        Only prepare the training data without running training.

        Useful for inspecting formatted data before committing to
        a full training run.
        """
        if self.dataset is None:
            self.dataset = SFTDataset(
                data_path=self.config.train_data_path,
                max_length=self.config.max_length,
                val_split=self.config.val_split_ratio,
            ).load()

        output_path = Path(self.config.output_dir) / "swift_train.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_swift_jsonl(output_path)

        # Also save a summary
        summary_path = Path(self.config.output_dir) / "data_summary.txt"
        with open(summary_path, "w") as f:
            f.write(self.dataset.summary())

        # Save training command
        cmd_path = Path(self.config.output_dir) / "train_command.sh"
        self.config.train_data_path = str(output_path)
        with open(cmd_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(self.config.to_swift_cli() + "\n")

        logger.info(f"Data prepared at {output_path}")
        return output_path

    def estimate_resources(self) -> dict:
        """
        Estimate compute resources needed for training.

        Based on model size, data size, and training config.
        """
        # Qwen2.5-Omni-7B parameters
        model_params_b = 7.0
        bytes_per_param = 2 if self.config.bf16 else 4

        # Memory estimates
        if self.config.use_lora:
            # LoRA: only train ~2% of params
            trainable_ratio = (
                2 * self.config.lora_rank / 4096  # Rough estimate
            )
            trainable_params_b = model_params_b * trainable_ratio
            # Model weights + optimizer states + gradients
            model_memory_gb = model_params_b * bytes_per_param / 1e9
            optimizer_memory_gb = trainable_params_b * 8 / 1e9  # AdamW: 8 bytes
            gradient_memory_gb = trainable_params_b * bytes_per_param / 1e9
        else:
            model_memory_gb = model_params_b * bytes_per_param / 1e9
            optimizer_memory_gb = model_params_b * 8 / 1e9
            gradient_memory_gb = model_params_b * bytes_per_param / 1e9

        total_memory_gb = (
            model_memory_gb + optimizer_memory_gb + gradient_memory_gb
        )
        # Activation memory (rough estimate)
        activation_gb = (
            self.config.per_device_batch_size
            * self.config.max_length
            * 4096  # Hidden dim
            * 4  # bytes
            / 1e9
        )
        total_memory_gb += activation_gb

        # Time estimate (rough)
        if self.dataset:
            n_samples = len(self.dataset)
        else:
            n_samples = 6000  # Default estimate

        steps_per_epoch = n_samples / self.config.batch_size
        total_steps = steps_per_epoch * self.config.num_epochs
        # Estimate ~0.5s per step on A100
        estimated_hours = total_steps * 0.5 / 3600

        return {
            "model_memory_gb": round(model_memory_gb, 1),
            "optimizer_memory_gb": round(optimizer_memory_gb, 1),
            "gradient_memory_gb": round(gradient_memory_gb, 1),
            "activation_memory_gb": round(activation_gb, 1),
            "total_memory_gb": round(total_memory_gb, 1),
            "per_gpu_memory_gb": round(total_memory_gb / self.config.num_gpus, 1),
            "recommended_gpu": (
                "A100-80GB" if total_memory_gb / self.config.num_gpus > 40
                else "A100-40GB" if total_memory_gb / self.config.num_gpus > 20
                else "A6000-48GB"
            ),
            "estimated_steps": int(total_steps),
            "estimated_hours": round(estimated_hours, 1),
            "n_training_samples": n_samples,
            "effective_batch_size": self.config.batch_size,
            "is_lora": self.config.use_lora,
        }