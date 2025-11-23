"""Data loading utilities for CAD-MLLM."""

try:
    from .dataset import (
        CADDataset,
        CADCollator,
        DummyCADDataset,
        create_dummy_dataset,
        MultimodalCADDataset,
        MultimodalCADCollator,
    )
except ImportError:
    pass

try:
    from .dataloader import get_autocomplete_dataloader, AutocompleteDataset, AutocompleteCollator
except ImportError:
    pass

__all__ = [
    "CADDataset",
    "CADCollator",
    "DummyCADDataset",
    "create_dummy_dataset",
    "MultimodalCADDataset",
    "MultimodalCADCollator",
    "get_autocomplete_dataloader",
    "AutocompleteDataset",
    "AutocompleteCollator",
]
