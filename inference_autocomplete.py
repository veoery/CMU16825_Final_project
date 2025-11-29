"""
Inference wrapper for CAD sequence autocomplete.

This script shows how to:
1. Load a trained autocomplete model
2. Provide partial CAD sequence + multimodal inputs
3. Generate completion
4. Post-process to get complete, executable JSON
"""

import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor

from cad_mllm import CADMLLMModel, CADMLLMConfig


class CADAutocompleteInference:
    """Wrapper for CAD sequence autocomplete inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize inference engine.

        Args:
            checkpoint_path: Path to trained checkpoint (e.g., "outputs/stage3_all/checkpoint-best")
            device: Device to run on ("cuda", "cpu", "mps")
            dtype: Model dtype (torch.bfloat16 or torch.float16)
        """
        self.device = device
        self.dtype = dtype

        # Load tokenizer and image processor
        print("Loading tokenizer and image processor...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        # Load model checkpoint
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        print("✅ Inference engine ready!")

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(Path(checkpoint_path) / "pytorch_model.bin", map_location=self.device)

        # Initialize model with same config as training
        config = CADMLLMConfig(
            llm_model_name="Qwen/Qwen3-8B",
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
        )
        model = CADMLLMModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device).to(self.dtype)

        return model

    def complete_sequence(
        self,
        truncated_json_path: str,
        caption: str,
        image_path: str = None,
        point_cloud_path: str = None,
        max_new_tokens: int = 3000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> dict:
        """Complete a partial CAD sequence.

        Args:
            truncated_json_path: Path to truncated JSON file
            caption: Text description of the CAD model
            image_path: Optional path to rendering image
            point_cloud_path: Optional path to point cloud (.npy)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (False = greedy decoding)

        Returns:
            Dictionary containing:
                - full_sequence: Complete list of CAD operations (executable!)
                - partial_sequence: Original partial sequence (for reference)
                - generated_operations: Newly generated operations
                - metadata: Generation metadata
        """
        # 1. Load truncated JSON
        with open(truncated_json_path, 'r') as f:
            truncated_json = json.load(f)

        partial_ops = truncated_json.get("sequence", [])
        kept_operations = len(partial_ops)

        print(f"📋 Partial sequence has {kept_operations} operations")

        # 2. Format prompt (using FULL truncated JSON as context)
        # The model will continue from here
        prompt = f"Complete this CAD sequence: {caption}\n{json.dumps(truncated_json, separators=(',', ':'))}"

        # 3. Tokenize
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=13000,  # Should match training max_seq_length
        )["input_ids"].to(self.device)

        print(f"📊 Input tokens: {input_ids.shape[1]}")

        # 4. Process multimodal inputs
        pixel_values = None
        point_cloud = None

        if image_path:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"].to(self.device)
            print("🖼️  Image loaded")

        if point_cloud_path:
            pc_data = np.load(point_cloud_path)
            # Normalize point cloud (same as training)
            pc_normalized = (pc_data - pc_data.mean(axis=0)) / (pc_data.std(axis=0) + 1e-8)
            point_cloud = torch.from_numpy(pc_normalized).unsqueeze(0).to(self.device).to(self.dtype)
            print(f"☁️  Point cloud loaded ({pc_data.shape[0]} points)")

        # 5. Generate completion
        print(f"🤖 Generating completion (max {max_new_tokens} tokens)...")

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                point_cloud=point_cloud,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 6. Decode generated tokens (only the NEW tokens, not the input)
        generated_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"✅ Generated {len(generated_tokens)} tokens")
        print(f"📝 Generated text preview: {generated_text[:200]}...")

        # 7. Parse generated operations
        try:
            # The generated text should be continuation of the JSON
            # It might be: ", op41, op42, ..., op100]" or similar
            # Try to parse it as a JSON array or extract operations

            # Attempt 1: Parse as JSON array
            try:
                generated_ops = json.loads(f"[{generated_text}]")
            except json.JSONDecodeError:
                # Attempt 2: Try to extract just the operations part
                # Look for operation objects in the generated text
                import re
                op_pattern = r'\{[^}]+\}'
                matches = re.findall(op_pattern, generated_text)
                generated_ops = [json.loads(match) for match in matches]

            print(f"✨ Parsed {len(generated_ops)} new operations")

        except Exception as e:
            print(f"⚠️  Warning: Could not parse generated JSON: {e}")
            print(f"Generated text: {generated_text}")
            generated_ops = []

        # 8. CRITICAL: Post-processing to create complete, executable JSON
        full_sequence = partial_ops + generated_ops

        print(f"🎯 Complete sequence: {kept_operations} (partial) + {len(generated_ops)} (generated) = {len(full_sequence)} total operations")

        # 9. Create final output (ready for CAD engine!)
        result = {
            "sequence": full_sequence,  # ← This is the complete, executable CAD sequence!
            "metadata": {
                "caption": caption,
                "total_operations": len(full_sequence),
                "partial_operations": kept_operations,
                "generated_operations": len(generated_ops),
                "generation_config": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                },
            },
        }

        return result

    def save_output(self, result: dict, output_path: str):
        """Save complete CAD sequence to file (ready for CAD engine).

        Args:
            result: Output from complete_sequence()
            output_path: Where to save the JSON
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"💾 Saved complete CAD sequence to: {output_path}")
        print(f"   This JSON can be directly loaded into your CAD engine!")


def example_usage():
    """Example of how to use the inference wrapper."""

    # 1. Initialize inference engine
    engine = CADAutocompleteInference(
        checkpoint_path="/path/to/checkpoint-best",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # 2. Generate completion for a partial CAD sequence
    result = engine.complete_sequence(
        truncated_json_path="data/json_truncated/0000/00000071_00005_tr_02.json",
        caption="Modern minimalist chair with wooden legs",
        image_path="data/img/0000/00000071_00005.png",
        point_cloud_path="data/pointcloud/0000/00000071_00005.npy",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=3000,
    )

    # 3. Save complete, executable JSON
    engine.save_output(result, "output_complete_chair.json")

    # 4. Use the complete sequence
    print("\n" + "="*80)
    print("COMPLETE CAD SEQUENCE (ready for CAD engine)")
    print("="*80)
    print(f"Total operations: {result['metadata']['total_operations']}")
    print(f"First 3 operations:")
    for i, op in enumerate(result['sequence'][:3]):
        print(f"  {i+1}. {op}")
    print("...")

    # Load into CAD engine (pseudo-code)
    # cad_engine.load_sequence(result["sequence"])
    # cad_engine.render()


if __name__ == "__main__":
    example_usage()
