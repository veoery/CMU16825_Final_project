"""Utility functions for training CAD-MLLM."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_info(model: torch.nn.Module):
    """Print model information including parameter counts.

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    print("=" * 60 + "\n")


def verify_lora_training(model: torch.nn.Module):
    """Verify that only LoRA parameters are trainable.

    This function checks that:
    1. Only LoRA adapter parameters are trainable
    2. Base LLM parameters are frozen
    3. Prints detailed information about trainable parameters

    Args:
        model: CAD-MLLM model instance

    Returns:
        bool: True if only LoRA parameters are trainable, False otherwise
    """
    print("\n" + "="*60)
    print("LoRA Training Verification")
    print("="*60)

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    # Check if LoRA is being used
    lora_params = [name for name, _ in trainable_params if 'lora' in name.lower()]
    non_lora_trainable = [name for name, _ in trainable_params if 'lora' not in name.lower()]

    print(f"\nTotal trainable parameters: {len(trainable_params)}")
    print(f"LoRA parameters (trainable): {len(lora_params)}")
    print(f"Non-LoRA trainable parameters: {len(non_lora_trainable)}")

    if lora_params:
        print("\n✓ LoRA parameters are trainable:")
        for name in lora_params[:5]:  # Show first 5
            print(f"  - {name}")
        if len(lora_params) > 5:
            print(f"  ... and {len(lora_params) - 5} more")

    if non_lora_trainable:
        print("\n⚠ WARNING: Non-LoRA parameters are also trainable:")
        for name in non_lora_trainable[:10]:  # Show first 10
            print(f"  - {name}")
        if len(non_lora_trainable) > 10:
            print(f"  ... and {len(non_lora_trainable) - 10} more")

    # Check base LLM is frozen
    base_llm_trainable = [name for name, _ in trainable_params
                          if 'llm' in name.lower() and 'lora' not in name.lower()]

    if not base_llm_trainable:
        print("\n✓ Base LLM parameters are frozen (as expected)")
    else:
        print("\n⚠ WARNING: Base LLM parameters are trainable:")
        for name in base_llm_trainable[:5]:
            print(f"  - {name}")

    # Summary
    total_trainable = sum(count for _, count in trainable_params)
    lora_count = sum(count for name, count in trainable_params if 'lora' in name.lower())

    print(f"\nTrainable parameter count:")
    print(f"  LoRA: {lora_count:,} ({100 * lora_count / total_trainable:.2f}%)")
    print(f"  Other: {total_trainable - lora_count:,} ({100 * (total_trainable - lora_count) / total_trainable:.2f}%)")

    print("="*60 + "\n")

    # Return True only if ONLY LoRA parameters are trainable
    # Allow projectors to be trainable as they need to be trained
    allowed_trainable = ['lora', 'projector', 'projection']
    only_allowed = all(
        any(allowed in name.lower() for allowed in allowed_trainable)
        for name, _ in trainable_params
    )

    return only_allowed


def get_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    """Create AdamW optimizer with proper weight decay.

    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient

    Returns:
        Optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias and layer norm
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Create learning rate scheduler with cosine annealing and warmup.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch (for resuming)

    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    save_path: str,
    is_best: bool = False,
):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        save_path: Path to save checkpoint directory
        is_best: Whether this is the best model
    """
    import os

    # Ensure the checkpoint directory exists
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    # Save checkpoint info (optimizer, scheduler state)
    checkpoint_info_path = os.path.join(save_path, "trainer_state.pt")
    torch.save(checkpoint, checkpoint_info_path)

    # Save model (save_pretrained will handle directory creation internally)
    model.save_pretrained(save_path)

    if is_best:
        best_path = save_path.replace("checkpoint", "best_model")
        os.makedirs(best_path, exist_ok=True)
        model.save_pretrained(best_path)
        # Also save optimizer/scheduler state for best model
        best_checkpoint_info_path = os.path.join(best_path, "trainer_state.pt")
        torch.save(checkpoint, best_checkpoint_info_path)
        print(f"Saved best model to {best_path}")

    print(f"Saved checkpoint to {save_path}")
