"""Configuration classes for CAD-MLLM."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CADMLLMConfig:
    """Main configuration for CAD-MLLM model.

    Args:
        llm_model_name: Name of the LLM model from HuggingFace
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA adaptation

        # Text encoder config
        freeze_text_encoder: Whether to freeze text encoder

        # Image encoder config (for future use)
        image_encoder_name: Name of the image encoder
        freeze_image_encoder: Whether to freeze image encoder
        image_size: Input image size

        # Point cloud encoder config (for future use)
        point_encoder_hidden_dim: Hidden dimension for point cloud encoder
        freeze_point_encoder: Whether to freeze point cloud encoder
        num_points: Number of points in point cloud

        # Projection layer config
        projector_hidden_dim: Hidden dimension for projection layers
        projector_num_layers: Number of layers in projection MLP

        # Training config
        max_seq_length: Maximum sequence length for CAD commands
        vocab_size: Vocabulary size for CAD tokenizer
        pad_token_id: Padding token ID
    """

    # LLM configuration
    llm_model_name: str = "Qwen/Qwen2.5-7B"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # Text encoder (using LLM's embeddings)
    freeze_text_encoder: bool = False

    # Image encoder configuration (extensible)
    image_encoder_name: str = "facebook/dinov2-large"
    freeze_image_encoder: bool = True
    image_size: int = 224

    # Point cloud encoder configuration (extensible)
    point_encoder_hidden_dim: int = 512
    freeze_point_encoder: bool = True
    num_points: int = 2048

    # Projection layer configuration
    projector_hidden_dim: int = 2048
    projector_num_layers: int = 2

    # Training configuration
    max_seq_length: int = 512
    vocab_size: int = 32000  # Will be set by tokenizer
    pad_token_id: Optional[int] = None

    point_encoder_type: str = "mlp"
    michelangelo_config_path: Optional[str] = None
    michelangelo_ckpt_path: Optional[str] = None

    # Device configuration
    device: str = "cuda"
    dtype: str = "bfloat16"  # "float32", "float16", or "bfloat16"


@dataclass
class TrainingConfig:
    """Training configuration for CAD-MLLM.

    Args:
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate for optimizer
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        use_wandb: Whether to use Weights & Biases for logging
        seed: Random seed for reproducibility
    """

    output_dir: str = "./outputs"
    num_epochs: int = 20
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    use_wandb: bool = False
    seed: int = 42
    device: str = "cuda"

    # Data configuration
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    num_workers: int = 4
