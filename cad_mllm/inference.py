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

        # Check for SafeTensors LoRA format (new format)
        adapter_file = ckpt_dir / "adapter_model.safetensors"
        image_proj_file = ckpt_dir / "image_projector.pt"
        point_proj_file = ckpt_dir / "point_projector.pt"

        if adapter_file.exists():
            # Initialize model WITHOUT LoRA first
            print(f"  Initializing base model (without LoRA)...")
            config = CADMLLMConfig(
                llm_model_name="Qwen/Qwen3-8B",
                use_lora=False,  # Don't wrap in LoRA yet
                lora_r=8,
                lora_alpha=16,
            )
            model = CADMLLMModel(config)

            # Now load LoRA adapters from checkpoint
            print(f"  Loading LoRA adapters from {ckpt_dir}...")
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(
                model.llm,  # Base LLM (not wrapped yet)
                str(ckpt_dir),
                is_trainable=False
            )

            # Load projectors - MUST enable encoders first!
            if image_proj_file.exists():
                print(f"  Enabling image encoder and projector...")
                model.enable_image_encoder()
                model.enable_image_projector()
                print(f"  Loading image projector state...")
                image_proj_state = torch.load(image_proj_file, map_location=self.device)
                model.image_projector.load_state_dict(image_proj_state)

            if point_proj_file.exists():
                print(f"  Enabling point cloud encoder and projector...")
                model.enable_point_encoder()
                model.enable_point_projector()
                print(f"  Loading point cloud projector state...")
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
        output_path: Optional[str] = None,
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

        # 2. Format input prompt - CRITICAL: Match training format!
        # During training, the model sees FULL JSON but labels are masked up to partial boundary
        # During inference, we give PARTIAL JSON and ask model to generate remaining ops
        # Key: The JSON must be INCOMPLETE (no closing braces) so model continues the sequence!
        
        # Build incomplete JSON that model will complete
        # Format: {"entities":{...},"sequence":[{op1},{op2},
        # Model will generate: {op3},{op4},...]}
        
        incomplete_json = {
            "entities": partial_data.get("entities", {}),
            "sequence": partial_ops
        }
        
        # Convert to string and remove closing braces to make it incomplete
        json_str = json.dumps(incomplete_json, separators=(',', ':'))
        
        # Remove the final "]}}" to make it incomplete
        # The model will generate more operations and close it properly
        if json_str.endswith(']}'): 
            if len(partial_ops) > 0:
                # If there are operations, remove final ]}} and leave trailing comma
                # {"entities":{...},"sequence":[{op1},{op2}]}} -> {"entities":{...},"sequence":[{op1},{op2},
                json_str = json_str[:-2] + ','  # Remove ]}, add comma for continuation
            else:
                # If no operations yet, remove final ]} to leave open array
                # {"entities":{},"sequence":[]}} -> {"entities":{},"sequence":[
                json_str = json_str[:-2]  # Remove ]}
        
        prompt = f"Complete this CAD sequence: {caption}\n{json_str}"
        
        print(f"\n{'='*80}")
        print("PROMPT (last 200 chars):")
        print(f"{'='*80}")
        print(prompt[-200:])
        print(f"{'='*80}\n")

        # 3. Tokenize
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # 4. Process multimodal inputs
        pixel_values = self._process_image(image) if image else None
        point_cloud_tensor = self._process_point_cloud(point_cloud) if point_cloud else None

        # 5. MULTIMODAL GENERATION using custom autoregressive loop
        # We build embeddings manually and use the LLM's forward pass iteratively
        print(f"\n{'='*80}")
        print("Generation Setup (MULTIMODAL):")
        print(f"{'='*80}")
        print(f"Text tokens: {input_ids.shape[1]}")
        print(f"Image: {'✓' if pixel_values is not None else '✗'}")
        print(f"Point Cloud: {'✓' if point_cloud_tensor is not None else '✗'}")
        print(f"Requested max_new_tokens: {max_new_tokens}")
        print(f"{'='*80}\n")

        # 6. Prepare initial embeddings (same order as training: image, PC, text)
        embeddings_list = []
        attention_list = []
        
        # Image embeddings
        if pixel_values is not None and self.model.has_image_encoder:
            pixel_values = pixel_values.to(self.dtype)
            image_features = self.model.image_encoder(pixel_values)
            if self.model.image_projector is not None:
                image_embeds = self.model.image_projector(image_features)
                embeddings_list.append(image_embeds)
                batch_size, seq_len = image_embeds.shape[:2]
                image_mask = torch.ones(batch_size, seq_len, device=image_embeds.device)
                attention_list.append(image_mask)
                print(f"Image embeddings: {image_embeds.shape}")
        
        # Point cloud embeddings
        if point_cloud_tensor is not None and self.model.has_point_encoder:
            point_cloud_tensor = point_cloud_tensor.to(self.dtype)
            point_features = self.model.point_encoder(point_cloud_tensor)
            if self.model.point_projector is not None:
                point_embeds = self.model.point_projector(point_features)
                embeddings_list.append(point_embeds)
                batch_size, seq_len = point_embeds.shape[:2]
                point_mask = torch.ones(batch_size, seq_len, device=point_embeds.device)
                attention_list.append(point_mask)
                print(f"Point cloud embeddings: {point_embeds.shape}")
        
        # Text embeddings
        text_embeds = self.model.text_encoder(input_ids)
        text_embeds = self.model.text_projector(text_embeds)
        embeddings_list.append(text_embeds)
        attention_list.append(attention_mask)
        print(f"Text embeddings: {text_embeds.shape}")
        
        # Concatenate all modalities
        inputs_embeds = torch.cat(embeddings_list, dim=1)
        full_attention_mask = torch.cat(attention_list, dim=1)
        
        print(f"Combined embeddings: {inputs_embeds.shape}")
        
        # 7. Autoregressive generation loop
        generated_ids = []
        current_embeds = inputs_embeds
        current_mask = full_attention_mask
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.model.llm(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    use_cache=False,  # Don't use KV cache with inputs_embeds
                )
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token.item())
                
                # Get embedding for next token
                next_token_embed = self.model.text_encoder(next_token)
                next_token_embed = self.model.text_projector(next_token_embed)
                
                # Append to sequence
                current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
                current_mask = torch.cat([current_mask, torch.ones(1, 1, device=current_mask.device)], dim=1)
        
        print(f"Generated {len(generated_ids)} new tokens")

        # 8. Decode generated tokens
        if generated_ids:
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            generated_text = ""
        
        print(f"\n{'='*80}")
        print("RAW MODEL OUTPUT (first 1000 chars):")
        print(f"{'='*80}")
        print(generated_text[:1000])
        print(f"{'='*80}\n")
        
        # Save full raw output for debugging
        if output_path:
            raw_output_path = str(output_path).replace('.json', '_raw.txt')
            with open(raw_output_path, 'w') as f:
                f.write(generated_text)
            print(f"💾 Full raw output saved to: {raw_output_path}")

        # 9. Parse generated operations
        generated_ops = self._parse_operations(generated_text)

        # 10. Merge with partial sequence to create complete CAD sequence
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
            loaded = np.load(pc)
            # Handle .npz format (compressed numpy archive)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                # Extract the first array (usually 'arr_0' or 'points')
                key = list(loaded.keys())[0]
                pc = loaded[key]
            else:
                pc = loaded

        # Normalize (same as training)
        pc_normalized = (pc - pc.mean(axis=0)) / (pc.std(axis=0) + 1e-8)
        pc_tensor = torch.from_numpy(pc_normalized).unsqueeze(0).to(self.device).to(self.dtype)

        return pc_tensor

    def _parse_operations(self, generated_text: str) -> List[Dict]:
        """Parse generated text into list of CAD operations.
        
        The model generates continuation of the sequence array, which may include:
        - Operation objects: {"index":3,"type":"Sketch","entity":"..."}
        - Closing brackets and braces: ]}}
        - Other JSON metadata
        
        We need to extract ONLY the operation objects.
        """
        try:
            # Strategy 1: Try to extract sequence array from complete JSON
            # If model generated: {"index":3,...},{...}]}} 
            # We want to extract the operation objects
            
            # First, try to complete the JSON and parse it
            # Wrap in array brackets if needed
            test_json = generated_text.strip()
            
            # Remove trailing closing braces that would break array parsing
            if test_json.endswith(']}'): 
                # Extract content before the closing sequence array bracket
                # Find the sequence of operations before ]}}
                pass
            
            # Strategy 2: Use regex to find all operation objects
            # Match complete JSON objects with index, type, entity fields
            op_pattern = r'\{[^{}]*"type"\s*:\s*"[^"]*"[^{}]*\}'
            matches = re.findall(op_pattern, generated_text)
            
            operations = []
            for match in matches:
                try:
                    op = json.loads(match)
                    # Validate it's a CAD operation (has index and type)
                    if "type" in op:
                        operations.append(op)
                except json.JSONDecodeError:
                    continue
            
            if operations:
                return operations
            
            # Strategy 3: Fallback to naive array parsing
            try:
                return json.loads(f"[{generated_text}]")
            except json.JSONDecodeError:
                pass
            
            return []
            
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
