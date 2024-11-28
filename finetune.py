import os
import torch
import deepspeed
from pathlib import Path
import wandb
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def create_deepspeed_config():
    """DeepSpeed configuration optimized for LLaVA fine-tuning"""
    return {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "wall_clock_breakdown": False
    }

class NavigationTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_wandb()
        self.setup_model()
        self.setup_data()
        self.setup_deepspeed()

    def setup_wandb(self):
        """Initialize WandB logging"""
        wandb.init(
            project="llava-navigation",
            config=self.config,
            name=f"finetune-stage-{self.config.stage}"
        )

    def setup_model(self):
        """Load and prepare LLaVA model"""
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        print("Loading model...")
        model_name = get_model_name_from_path(self.config.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.config.model_path,
            device_map=None  # Let DeepSpeed handle device placement
        )

        # Freeze parameters except for projections
        for name, param in self.model.named_parameters():
            if "mm_projector" not in name:
                param.requires_grad = False

    def setup_data(self):
        """Prepare dataset and dataloader"""
        from navigation_dataset import NavigationDataset

        print("Loading dataset...")
        self.dataset = NavigationDataset(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def setup_deepspeed(self):
        """Initialize DeepSpeed engine"""
        print("Setting up DeepSpeed...")
        ds_config = create_deepspeed_config()
        
        # Update config with training parameters
        ds_config["train_micro_batch_size_per_gpu"] = self.config.batch_size
        ds_config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps
        ds_config["optimizer"]["params"]["lr"] = self.config.learning_rate
        ds_config["scheduler"]["params"]["warmup_num_steps"] = int(
            len(self.dataloader) * self.config.num_epochs * self.config.warmup_ratio
        )
        ds_config["scheduler"]["params"]["total_num_steps"] = len(self.dataloader) * self.config.num_epochs

        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
            model_parameters=self.model.parameters()
        )

    def train(self):
        """Main training loop"""
        print("Starting training...")
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            self.model_engine.train()
            epoch_iterator = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}",
                disable=not torch.distributed.get_rank() == 0
            )

            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(self.model_engine.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model_engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=batch["images"],
                    labels=batch["labels"]
                )

                loss = outputs.loss

                # Backward pass
                self.model_engine.backward(loss)
                self.model_engine.step()

                # Logging
                if global_step % self.config.logging_steps == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step
                    })

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint-{global_step}")

                global_step += 1
                epoch_iterator.set_postfix(loss=loss.item())

            # Save epoch checkpoint
            self.save_model(f"epoch-{epoch}")

        # Save final model
        self.save_model("final")

    def save_model(self, tag):
        """Save model checkpoint"""
        output_dir = Path(self.config.output_dir) / tag
        self.model_engine.save_checkpoint(output_dir)
        self.tokenizer.save_pretrained(output_dir)

def main():
    from dataclasses import dataclass

    @dataclass
    class TrainingConfig:
        model_path: str = "lmms-lab/llava-next-interleave-qwen-7b"
        data_path: str = "data/navigation_dataset"
        output_dir: str = "checkpoints/navigation"
        stage: int = 1
        
        # Training params
        batch_size: int = 1
        gradient_accumulation_steps: int = 4
        num_epochs: int = 3
        learning_rate: float = 2e-5
        warmup_ratio: float = 0.03
        
        # System
        num_workers: int = 4
        
        # Logging
        logging_steps: int = 10
        save_steps: int = 500

    config = TrainingConfig()
    
    trainer = NavigationTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()