"""CAD-MLLM: Multimodal Large Language Model for CAD Generation."""

from .config import CADMLLMConfig, TrainingConfig, CurriculumStage, CurriculumTrainingConfig
from .model import CADMLLMModel
from .inference import CADAutocomplete, autocomplete_cad
from .utils.training import (
    set_seed,
    count_parameters,
    print_model_info,
    verify_lora_training,
    get_optimizer,
    get_cosine_schedule_with_warmup,
    AverageMeter,
    save_checkpoint,
)
from .data import (
    CADDataset,
    CADCollator,
    DummyCADDataset,
    create_dummy_dataset,
    MultimodalCADDataset,
    MultimodalCADCollator,
)

__all__ = [
    # Config
    "CADMLLMConfig",
    "TrainingConfig",
    "CurriculumStage",
    "CurriculumTrainingConfig",
    # Model
    "CADMLLMModel",
    # Inference
    "CADAutocomplete",
    "autocomplete_cad",
    # Training utilities
    "set_seed",
    "count_parameters",
    "print_model_info",
    "verify_lora_training",
    "get_optimizer",
    "get_cosine_schedule_with_warmup",
    "AverageMeter",
    "save_checkpoint",
    # Data
    "CADDataset",
    "CADCollator",
    "DummyCADDataset",
    "create_dummy_dataset",
    "MultimodalCADDataset",
    "MultimodalCADCollator",
]
