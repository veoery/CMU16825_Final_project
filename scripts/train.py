"""Training script for CAD-MLLM."""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cad_mllm import (
    CADMLLMConfig,
    TrainingConfig,
    CADMLLMModel,
    set_seed,
    print_model_info,
    verify_lora_training,
    get_optimizer,
    get_cosine_schedule_with_warmup,
    AverageMeter,
    save_checkpoint,
    CADDataset,
    CADCollator,
    DummyCADDataset,
    create_dummy_dataset,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CAD-MLLM")

    # Model arguments
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-4B", help="Name of the LLM model from HuggingFace")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")

    # Data arguments
    parser.add_argument("--create_dummy_data", action="store_true", help="Create dummy dataset")
    parser.add_argument("--num_dummy_samples", type=int, default=100, help="Number of dummy samples")
    parser.add_argument("--omnicad_txt_path", type=str, default="data/Omni-CAD/txt/0000.json", help="Path to Omni-CAD text descriptions")
    parser.add_argument("--omnicad_json_root", type=str, default="data/Omni-CAD/json", help="Root directory for Omni-CAD JSON files")

    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch,
    config,
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    global_step = 0

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

        # Update metrics
        loss_meter.update(loss.item() * config.gradient_accumulation_steps)

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # Logging
        if global_step % config.logging_steps == 0:
            print(f"\nStep {global_step} | Loss: {loss_meter.avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if global_step % config.save_steps == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint-epoch{epoch}-step{global_step}")
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)

    return loss_meter.avg


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create dummy data if requested
    if args.create_dummy_data or args.train_data_path is None:
        print("Creating dummy dataset...")
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)

        train_path = data_dir / "train_dummy.json"
        val_path = data_dir / "val_dummy.json"

        create_dummy_dataset(str(train_path), num_samples=args.num_dummy_samples)
        create_dummy_dataset(str(val_path), num_samples=20)

        args.train_data_path = str(train_path)
        args.val_data_path = str(val_path)

    # Create configurations
    model_config = CADMLLMConfig(
        llm_model_name=args.llm_model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        device=args.device,
        dtype=args.dtype,
    )

    train_config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        seed=args.seed,
        device=args.device,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
    )

    # Initialize model
    print("\nInitializing CAD-MLLM model...")
    model = CADMLLMModel(model_config)
    print_model_info(model)

    # Verify LoRA training setup
    verify_lora_training(model)

    # Create datasets
    print("\nLoading datasets...")
    if args.create_dummy_data:
        # Use DummyCADDataset for dummy data
        train_dataset = DummyCADDataset(
            data_path=train_config.train_data_path,
            tokenizer=model.tokenizer,
            max_seq_length=model_config.max_seq_length,
            modalities=["text"],
        )
    else:
        # Use full CADDataset for real Omni-CAD data
        train_dataset = CADDataset(
            data_path=args.omnicad_txt_path,
            json_root=args.omnicad_json_root,
            tokenizer=model.tokenizer,
            max_seq_length=model_config.max_seq_length,
            modalities=["text"],
        )

    # Create dataloaders
    collator = CADCollator(model.tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Set to 0 for debugging
    )

    # Setup optimizer and scheduler
    optimizer = get_optimizer(
        model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    num_training_steps = len(train_dataloader) * train_config.num_epochs // train_config.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    print("\nStarting training...")
    print(f"Total epochs: {train_config.num_epochs}")
    print(f"Total training steps: {num_training_steps}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Gradient accumulation steps: {train_config.gradient_accumulation_steps}")
    print(f"Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}\n")

    for epoch in range(train_config.num_epochs):
        avg_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            train_config,
        )

        print(f"\nEpoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        # Save checkpoint at end of epoch
        checkpoint_path = os.path.join(train_config.output_dir, f"checkpoint-epoch{epoch}")
        save_checkpoint(model, optimizer, scheduler, epoch, 0, checkpoint_path)

    # Save final model
    final_path = os.path.join(train_config.output_dir, "final_model")
    model.save_pretrained(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
