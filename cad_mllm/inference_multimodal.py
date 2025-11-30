"""
Multimodal CAD Autocomplete Inference (with LoRA weight merging)

This version merges LoRA weights into the base model to enable multimodal inference
with inputs_embeds, which PEFT doesn't support in its generate() method.
"""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, Union
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor

from .model import CADMLLMModel
from .config import CADMLLMConfig


class CADAutocompleteMultimodal:
    """Multimodal inference wrapper that merges LoRA weights for full multimodal support."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_seq_length: int = 13000,
    ):
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_seq_length = max_seq_length

        print(f"Loading CAD Autocomplete (Multimodal) from {checkpoint_path}...")
        print("⚠️  This will merge LoRA weights into base model for multimodal inference")

        # Load tokenizer and processors
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        # Load model with LoRA weights merged
        self.model = self._load_and_merge_lora(checkpoint_path)
        self.model.eval()

        print("✅ Model ready for MULTIMODAL inference!")

    def _load_and_merge_lora(self, checkpoint_path: str) -> CADMLLMModel:
        """Load checkpoint and merge LoRA weights into base model."""
        from peft import PeftModel

        ckpt_dir = Path(checkpoint_path)

        # Check for LoRA checkpoint
        adapter_file = ckpt_dir / "adapter_model.safetensors"

        if not adapter_file.exists():
            raise FileNotFoundError(
                f"No LoRA checkpoint found at {ckpt_dir}\n"
                f"Expected: adapter_model.safetensors"
            )

        # STEP 1: Initialize base model WITHOUT LoRA
        print(f"  Initializing base model (without LoRA)...")
        config = CADMLLMConfig(
            llm_model_name="Qwen/Qwen3-8B",
            use_lora=False,  # Don't wrap in LoRA
        )
        model = CADMLLMModel(config)

        # STEP 2: Load LoRA adapters from checkpoint
        print(f"  Loading LoRA adapters from {ckpt_dir}...")
        model.llm = PeftModel.from_pretrained(
            model.llm,
            str(ckpt_dir),
            is_trainable=False
        )

        # STEP 3: Merge LoRA weights into base model
        print(f"  Merging LoRA weights into base model...")
        model.llm = model.llm.merge_and_unload()
        print(f"  ✓ LoRA weights merged - model is now a regular transformer")

        # STEP 4: Load projectors
        image_proj_file = ckpt_dir / "image_projector.pt"
        point_proj_file = ckpt_dir / "point_projector.pt"

        if image_proj_file.exists():
            print(f"  Enabling image encoder and projector...")
            model.enable_image_encoder()
            model.enable_image_projector()
            image_proj_state = torch.load(image_proj_file, map_location=self.device)
            model.image_projector.load_state_dict(image_proj_state)

        if point_proj_file.exists():
            print(f"  Enabling point cloud encoder and projector...")
            model.enable_point_encoder()
            model.enable_point_projector()
            point_proj_state = torch.load(point_proj_file, map_location=self.device)
            model.point_projector.load_state_dict(point_proj_state)

        model = model.to(self.device).to(self.dtype)
        return model

    def complete(
        self,
        truncated_json: Union[str, Path, dict],
        caption: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        point_cloud: Optional[Union[str, Path, np.ndarray]] = None,
        output_path: Optional[str] = None,
        max_new_tokens: int = 3000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict:
        """Complete a partial CAD sequence with full multimodal support."""

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

        # 5. MULTIMODAL GENERATION (works because LoRA is merged!)
        print(f"\n{'='*80}")
        print("Generation Setup (MULTIMODAL):")
        print(f"{'='*80}")
        print(f"Text tokens: {input_ids.shape[1]}")
        print(f"Image: {'✓' if pixel_values is not None else '✗'}")
        print(f"Point Cloud: {'✓' if point_cloud_tensor is not None else '✗'}")
        print(f"{'='*80}\n")

        # Build combined embeddings
        embeddings_list = []

        # Text embeddings
        text_embeds = self.model.text_encoder(input_ids)
        embeddings_list.append(text_embeds)

        # Image embeddings
        if pixel_values is not None and self.model.has_image_encoder:
            image_feats = self.model.image_encoder(pixel_values)
            image_embeds = self.model.image_projector(image_feats)
            embeddings_list.append(image_embeds)

        # Point cloud embeddings
        if point_cloud_tensor is not None and self.model.has_point_encoder:
            pc_feats = self.model.point_encoder(point_cloud_tensor)
            pc_embeds = self.model.point_projector(pc_feats)
            embeddings_list.append(pc_embeds)

        # Concatenate all embeddings
        inputs_embeds = torch.cat(embeddings_list, dim=1)

        print(f"Combined embeddings shape: {inputs_embeds.shape}")

        # 6. Generate using inputs_embeds (NOW WORKS!)
        with torch.no_grad():
            generated_ids = self.model.llm.generate(
                inputs_embeds=inputs_embeds,  # ✓ Works now that LoRA is merged!
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        print(f"Generated {generated_ids.shape[1]} total tokens ({generated_ids.shape[1] - inputs_embeds.shape[1]} new)")

        # 7. Decode
        # IMPORTANT: When using inputs_embeds, generated_ids starts from token 0
        # We need to skip the number of tokens corresponding to our input embeddings
        prompt_length = inputs_embeds.shape[1]

        if generated_ids.shape[1] > prompt_length:
            new_tokens = generated_ids[0, prompt_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            generated_text = ""

        print(f"\n{'='*80}")
        print("RAW MODEL OUTPUT (first 1000 chars):")
        print(f"{'='*80}")
        print(generated_text[:1000])
        print(f"{'='*80}\n")

        # Save raw output
        if output_path:
            raw_output_path = str(output_path).replace('.json', '_raw.txt')
            with open(raw_output_path, 'w') as f:
                f.write(generated_text)
            print(f"💾 Full raw output saved to: {raw_output_path}")

        # 8. Parse operations
        generated_ops = self._parse_operations(generated_text)

        # 9. Merge with partial sequence
        full_sequence = partial_ops + generated_ops

        return {
            "sequence": full_sequence,
            "metadata": {
                "caption": caption,
                "partial_operations": len(partial_ops),
                "generated_operations": len(generated_ops),
                "total_operations": len(full_sequence),
                "used_image": pixel_values is not None,
                "used_point_cloud": point_cloud_tensor is not None,
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

        # Normalize
        pc_normalized = (pc - pc.mean(axis=0)) / (pc.std(axis=0) + 1e-8)
        pc_tensor = torch.from_numpy(pc_normalized).unsqueeze(0).to(self.device).to(self.dtype)

        return pc_tensor

    def _parse_operations(self, generated_text: str) -> list:
        """Parse generated text into CAD operations."""
        import re
        try:
            try:
                return json.loads(f"[{generated_text}]")
            except json.JSONDecodeError:
                op_pattern = r'\{[^}]+\}'
                matches = re.findall(op_pattern, generated_text)
                return [json.loads(match) for match in matches]
        except Exception as e:
            print(f"Warning: Failed to parse operations: {e}")
            return []
