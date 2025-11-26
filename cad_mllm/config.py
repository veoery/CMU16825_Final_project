"""Configuration classes for CAD-MLLM."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


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
    miche_encoder_cfg_path = "configs/michelangelo_point_encoder_cfg.yaml"
    miche_encoder_sd_path = "checkpoints/michelangelo_point_encoder_state_dict.pt"
    freeze_miche_encoder = True
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


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum training stage.

    Args:
        name: Stage name (e.g., "text", "text+pc", "text+pc+img")
        modalities: List of modalities to use in this stage
        num_epochs: Number of epochs for this stage
        learning_rate: Learning rate for this stage
        train_projectors: Whether to train projectors in this stage
        train_encoders: Whether to train encoders (usually False for frozen encoders)
        modality_sample_probs: Probability distribution for sampling modality combinations
    """
    name: str
    modalities: List[str]
    num_epochs: int
    learning_rate: float = 2e-5
    train_projectors: bool = True
    train_encoders: bool = False
    # Modality sampling probabilities: e.g., {"text": 0.3, "text+pc": 0.4, "text+pc+img": 0.3}
    modality_sample_probs: Optional[Dict[str, float]] = None


@dataclass
class CurriculumTrainingConfig(TrainingConfig):
    """Curriculum-based training configuration for CAD-MLLM.

    Implements progressive training strategy:
    - Stage 1: Text only (train LLM with LoRA)
    - Stage 2: Text + Point Cloud (train PC encoder/projector + LLM)
    - Stage 3: Text + Point Cloud + Image (train all modalities + LLM)

    Args:
        curriculum_stages: List of curriculum stages to execute
        enable_curriculum: Whether to use curriculum training
        projector_lr_multiplier: Learning rate multiplier for projectors
    """

    # Curriculum configuration
    enable_curriculum: bool = True
    curriculum_stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(
            name="stage1_text",
            modalities=["text"],
            num_epochs=5,
            learning_rate=2e-5,
            train_projectors=False,  # No projectors for text-only
            modality_sample_probs={"text": 1.0},
        ),
        CurriculumStage(
            name="stage2_text_pc",
            modalities=["text", "point_cloud"],
            num_epochs=5,
            learning_rate=2e-5,
            train_projectors=True,
            modality_sample_probs={
                "text": 0.3,
                "text+point_cloud": 0.7,
            },
        ),
        CurriculumStage(
            name="stage3_all",
            modalities=["text", "point_cloud", "image"],
            num_epochs=10,
            learning_rate=1e-5,  # Lower LR for final stage
            train_projectors=True,
            modality_sample_probs={
                "text": 0.2,
                "text+point_cloud": 0.3,
                "text+image": 0.2,
                "text+point_cloud+image": 0.3,
            },
        ),
    ])

    # Projector-specific learning rate (can be different from LLM)
    projector_lr_multiplier: float = 5.0  # Projectors learn faster than LLM
