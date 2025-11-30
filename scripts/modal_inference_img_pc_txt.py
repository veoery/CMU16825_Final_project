"""
Modal Inference Script for CAD-MLLM with Modal Volume Support

Deploy on Modal:
    modal deploy modal_inference.py

Run Examples:

Text only:
    modal run modal_inference.py \
        --repo "omnicad-lab-L3d/stage3-epoch0-step100-20251128_220651" \
        --prompt "Create a cylindrical shape with hollow center" \
        --max-tokens 256

With image + point cloud from Modal Volume:
    modal run modal_inference.py \
        --repo "omnicad-lab-L3d/stage3-epoch0-step100-20251128_220651" \
        --image "gt/img/0090/00902594_00001_000.png" \
        --pc "gt/pointcloud/0090/00902594_00001.npz" \
        --prompt "Generate a CAD model matching this image and point cloud" \
        --max-tokens 512
"""

import modal
import os
from pathlib import Path

# Define image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Install git for pip to clone from GitHub
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
        "git+https://github.com/veoery/CMU16825_Final_project.git@AWS",
    )
)

app = modal.App(name="cad-mllm-inference", image=image)

# Reference your Modal Volume named "l43d"
volume = modal.Volume.from_name("l43d", create_if_missing=False)

@app.function(gpu="A100", cpu=4, memory=32768, timeout=1800, volumes={"/mnt/data": volume})
# @app.function(gpu="A100", timeout=1800, volumes={"/mnt/data": volume})  # Change gpu="A100" to other options below
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
    import json
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForCausalLM, Dinov2Model, AutoImageProcessor
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    device = "cuda"
    dtype = torch.bfloat16

    # ========== JSON EXTRACTION HELPER ==========
    def extract_first_json(text):
        """
        Extract and repair truncated JSON from model output.
        If JSON is truncated, auto-close with remaining brackets.
        """
        try:
            start_index = text.find('{')
            if start_index == -1:
                return None, None

            stack = []
            in_string = False
            escape = False

            for i, char in enumerate(text[start_index:]):
                current_idx = start_index + i

                # Handle escape sequences and strings
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                # Handle brackets
                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}' or char == ']':
                    if stack and stack[-1] == char:
                        stack.pop()
                        # Perfect closure
                        if len(stack) == 0:
                            valid_json_str = text[start_index : current_idx + 1]
                            try:
                                json_obj = json.loads(valid_json_str)
                                return valid_json_str, json_obj
                            except json.JSONDecodeError:
                                return None, None
                    else:
                        return None, None

            # Auto-repair truncated JSON
            print("⚠️ JSON truncated, attempting auto-repair...")

            truncated_str = text[start_index:]

            if in_string:
                truncated_str += '\"'

            closing_suffix = "".join(reversed(stack))
            repaired_text = truncated_str + closing_suffix

            try:
                json_obj = json.loads(repaired_text)
                print(f"✓ Auto-repair success. Added suffix: {closing_suffix}")
                return repaired_text, json_obj
            except json.JSONDecodeError:
                print(f"⚠️ Repair validation failed")
                return None, None

        except Exception as e:
            print(f"⚠️ Extraction error: {e}")
            return None, None

    # Add project root and michelangelo to path
    import sys
    try:
        print(f"   [DEBUG] CWD: {os.getcwd()}")

        # Try multiple paths to find the project root (where cad_mllm is)
        possible_project_roots = [
            "CMU16825_Final_project",
            "/mnt/data/CMU16825_Final_project",  # In Modal Volume
            "/root/cmu/16825_l43d/CMU16825_Final_project",  # Local fallback
        ]

        project_root = None
        for path in possible_project_roots:
            exists = os.path.exists(path)
            has_cad_mllm = os.path.exists(os.path.join(path, "cad_mllm"))
            print(f"   [DEBUG] {path} (cad_mllm: {has_cad_mllm})")
            if exists and has_cad_mllm:
                project_root = path
                print(f"   ✓ Found project root at: {project_root}")
                break

        if project_root:
            # Add project root so cad_mllm can be imported
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                print(f"   ✓ Added project root to sys.path: {project_root}")

            # Add Michelangelo directory so "import michelangelo" works
            # The structure is: Michelangelo/michelangelo/, so adding Michelangelo parent
            # allows the encoder to find the michelangelo module
            michelangelo_parent = os.path.join(project_root, "Michelangelo")
            if os.path.exists(michelangelo_parent) and michelangelo_parent not in sys.path:
                sys.path.insert(0, michelangelo_parent)
                print(f"   ✓ Added Michelangelo to sys.path: {michelangelo_parent}")
        else:
            print(f"   ⚠️ Project root not found in any expected location")
            for path in possible_project_roots:
                print(f"      - Checked: {path}")
    except Exception as e:
        print(f"   ❌ Could not add paths: {e}")
        import traceback
        traceback.print_exc()

    # Convert volume paths to mounted paths
    if image_path:
        image_path = f"/mnt/data/{image_path}"
    if pc_path:
        pc_path = f"/mnt/data/{pc_path}"

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

        # Use local config file from project
        cfg_path = os.path.join(project_root, "configs", "michelangelo_point_encoder_cfg.yaml")

        # Try to find point encoder state dict locally or from Volume
        sd_path = None
        possible_sd_paths = [
            os.path.join(project_root, "checkpoints", "michelangelo_point_encoder_state_dict.pt"),
            "/mnt/data/michelangelo_pt/michelangelo_point_encoder_state_dict.pt",  # Modal Volume
            os.path.join(project_root, "Michelangelo", "checkpoints", "michelangelo_point_encoder_state_dict.pt"),
        ]

        for path in possible_sd_paths:
            if os.path.exists(path):
                sd_path = path
                print(f"   Found point encoder checkpoint at: {sd_path}")
                break

        if not sd_path:
            print(f"   ⚠️ Point encoder state dict not found locally, trying HuggingFace...")
            sd_path = hf_hub_download(repo, "michelangelo_point_encoder_state_dict.pt")

        point_encoder = MichelangeloPointEncoder(
            encoder_cfg_path=cfg_path,
            encoder_sd_path=sd_path,
            num_points=2048,
            dtype=dtype,
            freeze=True,
            device=device,
        )
        print("   ✓ Image encoder: DINOv2")
        print("   ✓ Point encoder: Michelangelo")
    except Exception as e:
        print(f"   ⚠️  Point encoder not available: {e}")
        import traceback
        print("   Import traceback:")
        traceback.print_exc()
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
    print("RESULT (RAW)")
    print("="*70)
    print(result)
    print("="*70)

    # Save raw output for debugging to Modal Volume
    print("\n💾 Saving raw output...")
    try:
        import time
        from pathlib import Path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw_dir = "/mnt/data/output_ckpt_4/raw"
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        raw_file = f"{raw_dir}/output_{timestamp}.txt"
        with open(raw_file, "w") as f:
            f.write(result)
        print(f"   ✓ Raw output saved: {raw_file}")
    except Exception as e:
        print(f"   ⚠️  Could not save raw output: {e}")

    # Extract and repair JSON
    print("\n" + "="*70)
    print("EXTRACTING JSON")
    print("="*70)
    clean_json_str, json_obj = extract_first_json(result)

    if clean_json_str:
        print("✓ JSON extracted and repaired successfully")
        print("\n" + "="*70)
        print("VALID JSON OUTPUT")
        print("="*70)
        print(clean_json_str)
        print("="*70)

        # Save extracted JSON to Modal Volume
        print("\n💾 Saving extracted JSON...")
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            json_dir = "/mnt/data/output_ckpt_4/json"
            Path(json_dir).mkdir(parents=True, exist_ok=True)
            json_file = f"{json_dir}/output_{timestamp}.json"
            with open(json_file, "w") as f:
                f.write(clean_json_str)
            print(f"   ✓ JSON output saved: {json_file}")
        except Exception as e:
            print(f"   ⚠️  Could not save JSON output: {e}")

        return clean_json_str
    else:
        print("⚠️ Could not extract valid JSON, returning raw output")
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
