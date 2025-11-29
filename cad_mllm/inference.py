"""
CAD Sequence Autocomplete Inference

Simple wrapper for generating complete CAD sequences from partial inputs.
Designed for easy integration with evaluation pipelines.
"""

import json
import torch
import re
from pathlib import Path
from typing import Optional, Dict, List, Union
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor

from .model import CADMLLMModel
from .config import CADMLLMConfig


class CADAutocomplete:
    """Inference wrapper for CAD sequence autocompletion.

    Example usage:
        >>> autocomplete = CADAutocomplete(checkpoint_path="path/to/checkpoint-best")
        >>> result = autocomplete.complete(
        ...     truncated_json="path/to/partial.json",
        ...     caption="A modern chair",
        ...     image="path/to/image.png",
        ...     point_cloud="path/to/pc.npy"
        ... )
        >>> # result["sequence"] is ready for CAD engine!
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_seq_length: int = 13000,
    ):
        """Initialize autocomplete model.

        Args:
            checkpoint_path: Path to trained checkpoint directory
            device: Device to run on ("cuda", "cpu", "mps")
            dtype: Model dtype ("bfloat16", "float16", "float32")
            max_seq_length: Maximum sequence length (should match training)
        """
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_seq_length = max_seq_length

        print(f"Loading CAD Autocomplete from {checkpoint_path}...")

        # Load tokenizer and processors
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        # Load model
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.eval()

        print("✅ Model ready for inference!")

    def _load_checkpoint(self, checkpoint_path: str) -> CADMLLMModel:
        """Load model from checkpoint (supports both legacy and LoRA format)."""
        ckpt_dir = Path(checkpoint_path)

        # Initialize model with same config
        config = CADMLLMConfig(
            llm_model_name="Qwen/Qwen3-8B",
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
        )

        print(f"  Initializing base model...")
        model = CADMLLMModel(config)

        # Check for SafeTensors LoRA format (new format)
        adapter_file = ckpt_dir / "adapter_model.safetensors"
        image_proj_file = ckpt_dir / "image_projector.pt"
        point_proj_file = ckpt_dir / "point_projector.pt"

        if adapter_file.exists():
            print(f"  Loading LoRA adapters from {ckpt_dir}...")
            # Load LoRA adapters using PEFT
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(
                model.llm.get_base_model(),
                str(ckpt_dir),
                is_trainable=False
            )

            if image_proj_file.exists():
                print(f"  Loading image projector...")
                image_proj_state = torch.load(image_proj_file, map_location=self.device)
                model.image_projector.load_state_dict(image_proj_state)

            if point_proj_file.exists():
                print(f"  Loading point cloud projector...")
                point_proj_state = torch.load(point_proj_file, map_location=self.device)
                model.point_projector.load_state_dict(point_proj_state)

        # Fallback: try legacy pytorch_model.bin format
        else:
            legacy_ckpt = ckpt_dir / "pytorch_model.bin"
            if legacy_ckpt.exists():
                print(f"  Loading from legacy pytorch_model.bin...")
                checkpoint = torch.load(legacy_ckpt, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {ckpt_dir}\n"
                    f"Expected: adapter_model.safetensors OR pytorch_model.bin"
                )

        model = model.to(self.device).to(self.dtype)
        return model

    def complete(
        self,
        truncated_json: Union[str, Path, dict],
        caption: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        point_cloud: Optional[Union[str, Path, np.ndarray]] = None,
        max_new_tokens: int = 3000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict:
        """Complete a partial CAD sequence.

        Args:
            truncated_json: Path to truncated JSON file or dict with partial sequence
            caption: Text description of the CAD model
            image: Optional image (path or PIL Image)
            point_cloud: Optional point cloud (path to .npy or numpy array)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling parameter
            do_sample: If False, use greedy decoding

        Returns:
            Dictionary with:
                - sequence: Complete list of CAD operations (ready for CAD engine!)
                - metadata: Generation info (partial_ops, generated_ops, total_ops)
        """
        # 1. Load partial sequence
        if isinstance(truncated_json, (str, Path)):
            with open(truncated_json, 'r') as f:
                partial_data = json.load(f)
        else:
            partial_data = truncated_json

        partial_ops = partial_data.get("sequence", [])

        # 2. Format input prompt
        partial_json_str = json.dumps(partial_data, separators=(',', ':'))
        prompt = f"Complete this CAD sequence: {caption}\n{partial_json_str}"

        # 3. Tokenize
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )["input_ids"].to(self.device)

        # 4. Process multimodal inputs
        pixel_values = self._process_image(image) if image else None
        point_cloud_tensor = self._process_point_cloud(point_cloud) if point_cloud else None

        # 5. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                point_cloud=point_cloud_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 6. Decode generated tokens
        generated_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 7. Parse generated operations
        generated_ops = self._parse_operations(generated_text)

        # 8. Merge with partial sequence to create complete CAD sequence
        full_sequence = partial_ops + generated_ops

        return {
            "sequence": full_sequence,  # Complete, executable CAD sequence!
            "metadata": {
                "caption": caption,
                "partial_operations": len(partial_ops),
                "generated_operations": len(generated_ops),
                "total_operations": len(full_sequence),
            }
        }

    def _process_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Process image input."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device)

    def _process_point_cloud(self, pc: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """Process point cloud input."""
        if isinstance(pc, (str, Path)):
            pc = np.load(pc)

        # Normalize (same as training)
        pc_normalized = (pc - pc.mean(axis=0)) / (pc.std(axis=0) + 1e-8)
        pc_tensor = torch.from_numpy(pc_normalized).unsqueeze(0).to(self.device).to(self.dtype)

        return pc_tensor

    def _parse_operations(self, generated_text: str) -> List[Dict]:
        """Parse generated text into list of CAD operations."""
        try:
            # Try parsing as JSON array
            try:
                return json.loads(f"[{generated_text}]")
            except json.JSONDecodeError:
                # Extract operation objects using regex
                op_pattern = r'\{[^}]+\}'
                matches = re.findall(op_pattern, generated_text)
                return [json.loads(match) for match in matches]
        except Exception as e:
            print(f"Warning: Failed to parse operations: {e}")
            print(f"Generated text: {generated_text[:500]}...")
            return []

    def batch_complete(
        self,
        samples: List[Dict],
        **generation_kwargs
    ) -> List[Dict]:
        """Complete multiple samples (useful for evaluation).

        Args:
            samples: List of dicts, each with keys:
                - truncated_json: Path or dict
                - caption: str
                - image: Optional path/PIL Image
                - point_cloud: Optional path/numpy array
            **generation_kwargs: Parameters for generation (temperature, top_p, etc.)

        Returns:
            List of results (same format as complete())
        """
        results = []
        for sample in samples:
            result = self.complete(
                truncated_json=sample["truncated_json"],
                caption=sample["caption"],
                image=sample.get("image"),
                point_cloud=sample.get("point_cloud"),
                **generation_kwargs
            )
            results.append(result)

        return results

    def save_result(self, result: Dict, output_path: Union[str, Path]):
        """Save complete CAD sequence to JSON file.

        Args:
            result: Output from complete()
            output_path: Where to save the JSON
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)


# Convenience function for quick inference
def autocomplete_cad(
    checkpoint_path: str,
    truncated_json: Union[str, Path, dict],
    caption: str,
    image: Optional[Union[str, Path, Image.Image]] = None,
    point_cloud: Optional[Union[str, Path, np.ndarray]] = None,
    output_path: Optional[Union[str, Path]] = None,
    **generation_kwargs
) -> Dict:
    """One-line function for CAD sequence completion.

    Example:
        >>> result = autocomplete_cad(
        ...     checkpoint_path="outputs/checkpoint-best",
        ...     truncated_json="partial_chair.json",
        ...     caption="Modern minimalist chair",
        ...     image="chair.png",
        ...     point_cloud="chair.npy",
        ...     output_path="complete_chair.json"
        ... )
        >>> # result["sequence"] is ready for CAD engine!

    Args:
        checkpoint_path: Path to trained model checkpoint
        truncated_json: Partial CAD sequence (path or dict)
        caption: Text description
        image: Optional image input
        point_cloud: Optional point cloud input
        output_path: Optional path to save complete JSON
        **generation_kwargs: temperature, top_p, max_new_tokens, etc.

    Returns:
        Dict with "sequence" (complete CAD operations) and "metadata"
    """
    model = CADAutocomplete(checkpoint_path)
    result = model.complete(
        truncated_json=truncated_json,
        caption=caption,
        image=image,
        point_cloud=point_cloud,
        **generation_kwargs
    )

    if output_path:
        model.save_result(result, output_path)

    return result
