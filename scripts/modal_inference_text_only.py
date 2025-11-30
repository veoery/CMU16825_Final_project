"""
Modal Inference Script for CAD-MLLM

Deploy on Modal:
    modal deploy modal_inference.py

Run:
    modal run modal_inference.py --repo "omnicad-lab-L3d/stage3-epoch0-step100-20251128_220651" --prompt "Create a cylinder"
"""

import modal
import os
from pathlib import Path

# Define image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("torch>=2.4.0", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install(
        "transformers>=4.40.0",
        "peft>=0.7.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "huggingface_hub>=0.20.0",
        "omegaconf>=2.3.0",
        "scipy>=1.10.0",
        "einops>=0.7.0",
    )
)

app = modal.App(name="cad-mllm-inference", image=image)

@app.function(gpu="A100", cpu=4, memory=32768, timeout=1800)
# @app.function(gpu="A100", timeout=1800)  # Change gpu="A100" to other options below
def run_inference(
    repo: str,
    prompt: str,
    image_path: str = None,
    pc_path: str = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Run CAD-MLLM inference on Modal GPU."""
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForCausalLM, Dinov2Model, AutoImageProcessor
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    device = "cuda"
    dtype = torch.bfloat16

    print("\n" + "="*70)
    print("CAD-MLLM Inference on Modal")
    print("="*70)

    # ========== LOAD LLM ==========
    print(f"\n📦 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # ========== LOAD LORA ==========
    print(f"📦 Loading LoRA from {repo}...")
    try:
        model = PeftModel.from_pretrained(llm, repo)
        model.eval()
        print("   ✓ LoRA loaded")
    except Exception as e:
        print(f"   ⚠️  LoRA load failed: {e}")
        print(f"   Using base model only")
        model = llm
        model.eval()

    # ========== LOAD PROJECTORS ==========
    print(f"📦 Loading projectors...")
    img_proj_state = torch.load(
        hf_hub_download(repo, "image_projector.pt"),
        map_location=device,
        weights_only=False,
    )
    pc_proj_state = torch.load(
        hf_hub_download(repo, "point_projector.pt"),
        map_location=device,
        weights_only=False,
    )

    # Extract dimensions from state dict (keys like "projector.0.weight")
    weight_keys = [k for k in img_proj_state.keys() if 'weight' in k]
    img_in_dim = img_proj_state[weight_keys[0]].shape[1]
    out_dim = img_proj_state[weight_keys[-1]].shape[0]

    weight_keys_pc = [k for k in pc_proj_state.keys() if 'weight' in k]
    pc_in_dim = pc_proj_state[weight_keys_pc[0]].shape[1]

    # Create wrapper class to match the saved state dict structure
    # Saved structure has: 0=Linear, 1-2=non-trainable layers, 3=Linear
    class ProjectorWrapper(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),      # 0: trainable
                torch.nn.GELU(),                           # 1: non-trainable
                torch.nn.LayerNorm(hidden_dim),           # 2: non-trainable (estimated)
                torch.nn.Linear(hidden_dim, out_dim),     # 3: trainable
            )

        def forward(self, x):
            return self.projector(x)

    # Build projectors
    image_projector = ProjectorWrapper(img_in_dim, 2048, out_dim).to(device).to(dtype)
    image_projector.load_state_dict(img_proj_state, strict=False)
    image_projector.eval()

    point_projector = ProjectorWrapper(pc_in_dim, 2048, out_dim).to(device).to(dtype)
    point_projector.load_state_dict(pc_proj_state, strict=False)
    point_projector.eval()

    print(f"   ✓ Image projector: {img_in_dim} → {out_dim}")
    print(f"   ✓ Point projector: {pc_in_dim} → {out_dim}")

    # ========== LOAD ENCODERS ==========
    print("📦 Loading encoders...")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-large", torch_dtype=dtype)
    image_encoder = image_encoder.to(device)
    image_encoder.eval()

    try:
        from cad_mllm.encoders import MichelangeloPointEncoder
        point_encoder = MichelangeloPointEncoder(
            encoder_cfg_path=hf_hub_download(repo, "michelangelo_point_encoder_cfg.yaml"),
            encoder_sd_path=hf_hub_download(repo, "michelangelo_point_encoder_state_dict.pt"),
            num_points=2048,
            dtype=dtype,
            freeze=True,
            device=device,
        )
        print("   ✓ Image encoder: DINOv2")
        print("   ✓ Point encoder: Michelangelo")
    except Exception as e:
        print(f"   ⚠️  Point encoder not available: {e}")
        point_encoder = None

    # ========== GENERATE ==========
    print("\n" + "="*70)
    print("GENERATING")
    print("="*70)
    print(f"Prompt: {prompt}")

    embeddings = []
    masks = []

    # Image
    if image_path:
        print(f"\n🖼️  Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            img_feats = image_encoder(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
            img_embeds = image_projector(img_feats.to(dtype))
        embeddings.append(img_embeds)
        masks.append(torch.ones(img_embeds.shape[:2], device=device))
        print(f"   ✓ Shape: {img_embeds.shape}")

    # Point Cloud
    if pc_path and point_encoder:
        print(f"\n☁️  Processing point cloud: {pc_path}")
        data = np.load(pc_path)
        for key in ['points', 'xyz', 'point_cloud', 'data']:
            if key in data.files:
                points = data[key].astype(np.float32)
                break
        else:
            raise KeyError(f"No points in {pc_path}")

        points = torch.from_numpy(points).unsqueeze(0).to(device).to(dtype)
        with torch.no_grad():
            pc_feats = point_encoder(points)
            pc_embeds = point_projector(pc_feats.to(dtype))
        embeddings.append(pc_embeds)
        masks.append(torch.ones(pc_embeds.shape[:2], device=device))
        print(f"   ✓ Shape: {pc_embeds.shape}")

    # Text
    print(f"\n📝 Processing text prompt")
    text_inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=512)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embeds = llm.model.embed_tokens(text_inputs["input_ids"])
    embeddings.append(text_embeds)
    masks.append(text_inputs["attention_mask"])
    print(f"   ✓ Shape: {text_embeds.shape}")

    # Concatenate
    inputs_embeds = torch.cat(embeddings, dim=1)
    attention_mask = torch.cat(masks, dim=1)
    print(f"\n✅ Final input: {inputs_embeds.shape}")

    # Generate
    print(f"\n🚀 Generating {max_tokens} tokens...")
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(result)
    print("="*70)

    return result


@app.local_entrypoint()
def main(
    repo: str = "omnicad-lab-L3d/stage3-epoch0-step100-20251128_220651",
    prompt: str = "Create a cylindrical shape with hollow center",
    image_path: str = None,
    pc_path: str = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Local entrypoint for running inference on Modal."""
    result = run_inference.remote(
        repo=repo,
        prompt=prompt,
        image_path=image_path,
        pc_path=pc_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result
