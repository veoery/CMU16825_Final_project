#!/usr/bin/env python3
"""
Batch Modal Inference Script for CAD-MLLM

This script prepares sample data paths from Modal Volume and runs inference.

Usage:
    modal run batch_modal_inference.py --num-samples 2 --folder 0090
"""

import modal
import os
import glob
import json
import shutil
from pathlib import Path
import argparse

# Define Modal image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
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

app = modal.App(name="cad-mllm-batch-inference", image=image)

# Reference your Modal Volume
volume = modal.Volume.from_name("l43d", create_if_missing=False)


@app.function(gpu="A100", cpu=4, memory=32768, timeout=3600, volumes={"/mnt/data": volume})
def run_single_inference(
    repo: str,
    prompt: str,
    image_paths: list,  # List of image paths for multi-view support
    pc_path: str,
    max_tokens: int = 10240,
    temperature: float = 0.5,
):
    """Run inference for a single sample."""
    import torch
    import numpy as np
    import json
    import sys
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForCausalLM, Dinov2Model, AutoImageProcessor
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    device = "cuda"
    dtype = torch.bfloat16

    # ========== JSON EXTRACTION HELPER ==========
    def extract_first_json(text):
        try:
            start_index = text.find('{')
            if start_index == -1:
                return None, None

            stack = []
            in_string = False
            escape = False

            for i, char in enumerate(text[start_index:]):
                current_idx = start_index + i
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

                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}' or char == ']':
                    if stack and stack[-1] == char:
                        stack.pop()
                        if len(stack) == 0:
                            valid_json_str = text[start_index : current_idx + 1]
                            try:
                                json_obj = json.loads(valid_json_str)
                                return valid_json_str, json_obj
                            except json.JSONDecodeError:
                                return None, None
                    else:
                        return None, None

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

    # Setup paths
    try:
        project_root = "/mnt/data/CMU16825_Final_project"
        if not os.path.exists(project_root):
            print(f"❌ Project not found at {project_root}")
            return {"status": "error", "message": "Project not found"}

        sys.path.insert(0, project_root)
        michelangelo_path = os.path.join(project_root, "Michelangelo")
        sys.path.insert(0, michelangelo_path)
        print(f"✓ Paths configured")
    except Exception as e:
        print(f"❌ Path setup failed: {e}")
        return {"status": "error", "message": str(e)}

    print("\n" + "="*70)
    print(f"[DEBUG] run_single_inference called with {len(image_paths)} image(s)")
    print(f"[DEBUG] image_paths: {image_paths}")
    print("="*70)

    # Load models
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

    print(f"📦 Loading LoRA...")
    try:
        model = PeftModel.from_pretrained(llm, repo)
        model.eval()
    except Exception as e:
        print(f"⚠️ LoRA load failed: {e}")
        model = llm
        model.eval()

    # Load projectors
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

    weight_keys = [k for k in img_proj_state.keys() if 'weight' in k]
    img_in_dim = img_proj_state[weight_keys[0]].shape[1]
    out_dim = img_proj_state[weight_keys[-1]].shape[0]

    weight_keys_pc = [k for k in pc_proj_state.keys() if 'weight' in k]
    pc_in_dim = pc_proj_state[weight_keys_pc[0]].shape[1]

    class ProjectorWrapper(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return self.projector(x)

    image_projector = ProjectorWrapper(img_in_dim, 2048, out_dim).to(device).to(dtype)
    image_projector.load_state_dict(img_proj_state, strict=False)
    image_projector.eval()

    point_projector = ProjectorWrapper(pc_in_dim, 2048, out_dim).to(device).to(dtype)
    point_projector.load_state_dict(pc_proj_state, strict=False)
    point_projector.eval()

    # Load encoders
    print("📦 Loading encoders...")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-large", torch_dtype=dtype)
    image_encoder = image_encoder.to(device).eval()
    
    point_encoder = None
    print(f"   [DEBUG] CWD: {os.getcwd()}")
    # try:
    #     print("[DEBUG] try import pc encoder")
    #     from cad_mllm.encoders import MichelangeloPointEncoder, PointCloudEncoder
    #     cfg_path = os.path.join(project_root, "configs", "michelangelo_point_encoder_cfg.yaml")
    #     sd_path = None

    #     possible_sd_paths = [
    #         os.path.join(project_root, "checkpoints", "michelangelo_point_encoder_state_dict.pt"),
    #         os.path.join(project_root, "Michelangelo", "checkpoints", "michelangelo_point_encoder_state_dict.pt"),
    #     ]

    #     for path in possible_sd_paths:
    #         if os.path.exists(path):
    #             sd_path = path
    #             break

    #     if sd_path:
    #         print(f"[DEBUG] sd_path: {sd_path}")
    #         point_encoder = MichelangeloPointEncoder(
    #             encoder_cfg_path=cfg_path,
    #             encoder_sd_path=sd_path,
    #             num_points=2048,
    #             dtype=dtype,
    #             freeze=True,
    #             device=device,
    #         )
    #         print("   ✓ Point encoder loaded")
    # except Exception as e:
    #     print(f"   ⚠️ Point encoder not available: {e}")
    # ... inside run_single_inference ...

    try:
        print("[DEBUG] try import pc encoder")
        from cad_mllm.encoders import MichelangeloPointEncoder, PointCloudEncoder
        
        # 1. Fix Config Path (Ensure this exists too!)
        cfg_path = os.path.join(project_root, "configs", "michelangelo_point_encoder_cfg.yaml")
        
        sd_path = None

        # 2. UPDATE THESE PATHS to match your Volume layout
        possible_sd_paths = [
            # Original repo paths
            os.path.join(project_root, "checkpoints", "michelangelo_point_encoder_state_dict.pt"),
            os.path.join(project_root, "Michelangelo", "checkpoints", "michelangelo_point_encoder_state_dict.pt"),
            
            # ✅ ADD THIS: The likely location on your volume based on previous chats
            "/mnt/data/michelangelo_pt/michelangelo_point_encoder_state_dict.pt",
            "/mnt/data/checkpoints/michelangelo_point_encoder_state_dict.pt",
            
            # ✅ ADD THIS: Check the repo folder itself just in case
            "/mnt/data/CMU16825_Final_project/michelangelo_point_encoder_state_dict.pt"
        ]

        print(f"   [DEBUG] Searching for PC weights in: {possible_sd_paths}")

        for path in possible_sd_paths:
            if os.path.exists(path):
                sd_path = path
                print(f"   [DEBUG] Found PC weights at: {sd_path}")
                break

        if sd_path:
            # ... existing initialization code ...
            # FIX: Pass the found path explicitly
            point_encoder = MichelangeloPointEncoder(
                encoder_cfg_path=cfg_path,
                encoder_sd_path=sd_path, # <--- Pass the found path here
                num_points=2048,
                dtype=dtype,
                freeze=True,
                device=device
            )
            print("   ✓ Point encoder loaded")
        else:
            # 3. Add an explicit error print if not found
            print(f"   ❌ CRITICAL: Point encoder weights NOT found. listing /mnt/data to debug:")
            try:
                print(os.listdir("/mnt/data"))
            except:
                pass

    except Exception as e:
        print(f"   ⚠️ Point encoder init failed with error: {e}")
        import traceback
        traceback.print_exc() # Print full error stack trace

    # Process inputs
    print(f"\n🖼️  Processing {len(image_paths)} image(s) (multi-view)...")
    embeddings = []
    masks = []
    
    try:
        # Process each image individually and collect embeddings
        total_image_tokens = 0
        image_paths = image_paths[:1]
        print(f"⚠️ ⚠️ ⚠️ TAKE ONLY {len(image_paths)}")
        for img_idx, img_path in enumerate(image_paths):
            print(f"   [{img_idx + 1}/{len(image_paths)}] Reading: {os.path.basename(img_path)}")
            try:
                image = Image.open(img_path).convert("RGB")
                # print(f"       - Size: {image.size}") # Optional debug

                # Process single image
                inputs = image_processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                with torch.no_grad():
                    # Encode
                    img_feats = image_encoder(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
                    # Project
                    img_embeds = image_projector(img_feats.to(dtype))

                embeddings.append(img_embeds)
                masks.append(torch.ones(img_embeds.shape[:2], device=device))
                
                current_tokens = img_embeds.shape[1]
                total_image_tokens += current_tokens
                # print(f"       - Embedding shape: {img_embeds.shape}") # Optional debug

            except Exception as e:
                print(f"       ❌ Failed to process {img_path}: {e}")
                # Depending on strictness, you might want to return error or continue
                return {"status": "error", "message": f"Image error: {e}"}

        # Check if we actually got embeddings
        if not embeddings:
            return {"status": "error", "message": "No valid images processed"}

        # --- FIX IS HERE ---
        # Do NOT convert to tensor yet if you plan to append PC/Text tokens to this list later.
        # Just calculate the shape for printing.
        token_sum = sum(e.shape[1] for e in embeddings)
        embedding_dim = embeddings[0].shape[2]
        
        print(f"   ✓ Total Image Features: {total_image_tokens} tokens from {len(image_paths)} views")
        print(f"   ✓ Combined Image shape will be: [1, {token_sum}, {embedding_dim}]") 

    except Exception as e:
        print(f"   ❌ Image processing failed: {e}")
        return {"status": "error", "message": f"Image processing failed: {e}"}
    # print("⚠️ ⚠️ ⚠️ Test 5: Skip IMG")
    # print("⚠️ ⚠️ ⚠️ Test 7: DISABLE PC FOR TEST")
    # Point cloud
    print(f"☁️  Processing point cloud...")
    print(f"[DEBUG] point_encoder exists: {point_encoder is not None}")
    if point_encoder:
        try:
            data = np.load(pc_path)
            for key in ['points', 'xyz', 'point_cloud', 'data']:
                if key in data.files:
                    points = data[key].astype(np.float32)
                    break
            else:
                raise KeyError(f"No points found in {pc_path}")

            points = torch.from_numpy(points).unsqueeze(0).to(device).to(dtype)
            with torch.no_grad():
                pc_feats = point_encoder(points)
                pc_embeds = point_projector(pc_feats.to(dtype))
            embeddings.append(pc_embeds)
            
            masks.append(torch.ones(pc_embeds.shape[:2], device=device))
            print(f"   ✓ Point cloud shape: {pc_embeds.shape}")
        except Exception as e:
            print(f"   ⚠️ Point cloud processing failed: {e}")
    else:
        print(f"[DEBUG] exist: {point_encoder}")
        print()

    # Text
    print(f"📝 Processing text...")
    text_inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=512)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embeds = llm.model.embed_tokens(text_inputs["input_ids"])
    embeddings.append(text_embeds)
    masks.append(text_inputs["attention_mask"])
    print(f"   ✓ Text shape: {text_embeds.shape}")

    # Generate
    inputs_embeds = torch.cat(embeddings, dim=1)
    attention_mask = torch.cat(masks, dim=1)
    # max_tokens = 4096 #TODO: TEST CHANGE BACK LTR
    print(f"\n🚀 Generating {max_tokens} tokens, with temperature = {temperature}...")

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            # max_new_tokens=4096, #TODO: TEST CHANGE BACK LTR
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*70)
    print("RAW OUTPUT (first 1000 chars) ")
    print("="*70)
    print(result[:1000])
    if len(result) > 1000:
        print(f"\n... ({len(result) - 1000} more characters)")
    print("="*70)

    # Extract JSON
    clean_json_str, json_obj = extract_first_json(result)

    # Save raw output and JSON to Modal Volume
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sample_id = os.path.basename(image_paths[0]).split('_')[0:2]
    sample_id_str = "_".join(sample_id)

    OUTPUT_VER = "B4_1img_pc_fix_case_2"
    raw_output_dir = f"/mnt/data/output_ckpt_5_{OUTPUT_VER}/raw"
    json_output_dir = f"/mnt/data/output_ckpt_5_{OUTPUT_VER}/json"

    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    raw_filename = f"{sample_id_str}_repaired_{timestamp}.txt"
    json_filename = f"{sample_id_str}_repaired_{timestamp}.json"

    raw_path = os.path.join(raw_output_dir, raw_filename)
    json_path = os.path.join(json_output_dir, json_filename)

    # Save raw output
    with open(raw_path, 'w') as f:
        f.write(result)

    print(f"\n📄 DEBUG - Files saved:")
    print(f"   Raw output: {raw_path}")
    print(f"   Size: {len(result)} chars, {os.path.getsize(raw_path)} bytes")

    if clean_json_str:
        # Save JSON
        with open(json_path, 'w') as f:
            f.write(clean_json_str)

        print(f"   JSON output: {json_path}")
        print(f"   Size: {len(clean_json_str)} chars, {os.path.getsize(json_path)} bytes")
        print("✓ JSON extracted successfully")

        return {
            "status": "success",
            "sample_id": sample_id_str,
            "json_length": len(clean_json_str),
            "has_valid_json": True,
            "raw_file": raw_path,
            "json_file": json_path,
        }
    else:
        print("⚠️ Could not extract valid JSON")
        return {
            "status": "warning",
            "sample_id": sample_id_str,
            "message": "JSON extraction failed but generation succeeded",
            "raw_file": raw_path,
        }


@app.function(volumes={"/mnt/data": volume})
def prepare_samples(folder: str = "0090", num_samples: int = 2, max_new_tokens: int = 10240, min_tokens: int = 2048, already_generated: list = None):
    """Prepare samples from Modal Volume (runs on Modal, has access to volume).

    Filters samples based on ground truth JSON token count.
    Only selects samples where: min_tokens < gt_json_tokens < max_new_tokens
    """
    import os
    import glob
    import json
    from transformers import AutoTokenizer

    if already_generated is None:
        already_generated = ["00900284_00001", "00900312_00003", "00900387_00001", "00900654_00001", "00900730_00001"]

    # Use /mnt/data path (Modal Volume mount)
    vol_root = "/mnt/data"

    # Load tokenizer to match model's tokenization
    print("\n📦 Loading tokenizer for token counting...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    # Load captions
    json_path = os.path.join(vol_root, "gt/txt", f"{folder}.json")
    print(f"\n📄 Loading captions from: {json_path}")

    try:
        with open(json_path, 'r') as f:
            all_items = json.load(f)
        caption_map = {item['id']: item.get('text caption', '') for item in all_items if 'id' in item}
        print(f"   ✓ Loaded {len(caption_map)} captions")
    except Exception as e:
        print(f"   ❌ Failed to load captions: {e}")
        return []

    # Find point clouds and filter by GT JSON token count
    pc_folder = os.path.join(vol_root, "gt/pointcloud", folder)
    pc_files = sorted(glob.glob(os.path.join(pc_folder, "*.npz")))
    print(f"☁️  Found {len(pc_files)} point clouds")

    # Load ground truth JSON files for token filtering
    json_folder = os.path.join(vol_root, "gt/json", folder)
    gt_json_files = glob.glob(os.path.join(json_folder, "*.json"))

    print(f"\n🔍 Filtering by GT JSON token count ({min_tokens} < tokens < {max_new_tokens})...")
    print(f"{'SAMPLE_ID':<25} | {'GT_TOKENS':<12} | {'STATUS':<15}")
    print("-" * 60)

    """
    FIXED SPECIAL COMAPRE SET:
    "00907893_00007": a key like thing (bigger)
    "00902663_00003": should be just a box
    "00901845_00003": a key like thing
    "00900387_00001": B1_COMPARE (likely nice in B1), cylinder with hollow
    "00901551_00005": B2_COMPARE, 2 cylinder-fail in B2 too, L shape things.
    """

    # SPECIAL_SET = ["00907893_00007", "00902663_00003", "00901845_00003", "00900387_00001", "00901551_00005"]
    SPECIAL_SET = ["00902693_00001", "00900730_00001", "00900903_00001","00901173_00001", "00901552_00001","00902084_00001","00902323_00001","00902594_00001","00903270_00001","00903601_00001","00903796_00001","00904065_00001","00904100_00001","00900654_00001","00900867_00001","00901147_00001","00903644_00001","00903823_00001"]

    valid_sample_ids = []
    valid_sample_ids.extend(SPECIAL_SET)
    print("ADDED SPECIAL TEST SET")

    for gt_json_file in gt_json_files:
        sample_id = os.path.splitext(os.path.basename(gt_json_file))[0]
        sample_key = f"{folder}/{sample_id}"

        # Skip already generated samples
        if sample_key in already_generated:
            print(f"{sample_id:<25} | {'--':<12} | {'SKIPPED':<15}")
            continue

        try:
            with open(gt_json_file, 'r', encoding='utf-8') as f:
                gt_content = f.read()

            token_count = len(tokenizer.encode(gt_content))

            # Check if within token range
            CAP_TOKEN = 2048
            if min_tokens < token_count < CAP_TOKEN:
                valid_sample_ids.append(sample_id)
                status = "✓ VALID"
                # print(f"{sample_id:<25} | {token_count:<12} | {status:<15}")
            # else:
            #     status = f"✗ OUT_OF_RANGE"
            #     print(f"{sample_id:<25} | {token_count:<12} | {status:<15}")

            # Continue scanning ALL files - don't stop early!
            # We'll filter by file availability in the next step

        except Exception as e:
            print(f"{sample_id:<25} | {'ERROR':<12} | {str(e):<15}")

    print("-" * 60)
    print(f"\n✅ Found {len(valid_sample_ids)} valid samples (filtered by GT JSON tokens)")

    # Prepare samples using filtered IDs - continue until we have enough with matching files
    samples = []
    img_folder = os.path.join(vol_root, "gt/img", folder)

    print(f"\n🔎 Matching images and point clouds...")
    print(f"{'SAMPLE_ID':<25} | {'IMAGES':<10} | {'PC':<5} | {'STATUS':<20}")
    print("-" * 70)

    skipped_count = 0
    for sample_id in valid_sample_ids:
        # Check if we have enough samples
        if len(samples) >= num_samples:
            print(f"\n✅ Collected {len(samples)} samples with matching files")
            break

        # Find images
        img_pattern = os.path.join(img_folder, f"{sample_id}_*.png")
        img_files = sorted([f for f in glob.glob(img_pattern) if not f.endswith(':Zone.Identifier')])

        # Check point cloud exists
        pc_path = os.path.join(pc_folder, f"{sample_id}.npz")
        pc_exists = os.path.exists(pc_path)

        num_images = len(img_files)
        pc_status = "✓" if pc_exists else "✗"

        if not img_files:
            status = "❌ No images"
            skipped_count += 1
            print(f"{sample_id:<25} | {num_images:<10} | {pc_status:<5} | {status:<20}")
            continue

        if not pc_exists:
            status = "❌ No point cloud"
            skipped_count += 1
            print(f"{sample_id:<25} | {num_images:<10} | {pc_status:<5} | {status:<20}")
            continue

        # Get caption
        caption_key = f"{folder}/{sample_id}"
        caption = caption_map.get(caption_key, "Generate a CAD model based on the image and point cloud")

        samples.append({
            "sample_id": sample_id,
            "image_paths": img_files,  # Use ALL images (multi-view)
            "pc_path": pc_path,
            "prompt": caption,
        })

        status = "✓ READY"
        print(f"{sample_id:<25} | {num_images:<10} | {pc_status:<5} | {status:<20}")

    print("-" * 70)
    print(f"\n📊 Summary:")
    print(f"   Valid samples found: {len(valid_sample_ids)}")
    print(f"   Samples prepared:    {len(samples)}")
    print(f"   Skipped (missing files): {skipped_count}")

    if len(samples) == 0:
        print(f"\n❌ No samples with matching images and point clouds found!")
        print(f"   Check that image and point cloud folders exist:")
        print(f"   Images: {img_folder}")
        print(f"   Point clouds: {pc_folder}")

    return samples

def zip_output_versions(output_versions: list):
    """Zip output directories for easy download."""
    print("\n" + "="*70)
    print("ZIPPING OUTPUT DIRECTORIES")
    print("="*70)

    VOL_ROOT = "/mnt/data"
    zip_info = []

    for output_version in output_versions:
        output_dir = os.path.join("/mnt/data", f"output_ckpt_5_{output_version}")

        if not os.path.exists(output_dir):
            print(f"⚠️  Directory not found: {output_dir}")
            continue

        # Create zip file
        zip_name = f"output_ckpt_5_{output_version}"
        zip_path = os.path.join(VOL_ROOT, zip_name)

        try:
            print(f"\n📦 Zipping {output_version}...")
            shutil.make_archive(zip_path, 'zip', output_dir)

            zip_size_mb = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
            file_count = sum([len(files) for _, _, files in os.walk(output_dir)])

            print(f"   ✅ Created: {zip_name}.zip")
            print(f"   Size: {zip_size_mb:.1f} MB")
            print(f"   Files: {file_count}")

            zip_info.append({
                'name': zip_name,
                'size_mb': zip_size_mb,
                'files': file_count
            })
        except Exception as e:
            print(f"   ❌ Error: {e}")

    if zip_info:
        print(f"\n{'='*70}")
        print("📥 READY FOR DOWNLOAD:")
        print(f"{'='*70}")
        for info in zip_info:
            print(f"  • {info['name']}.zip ({info['size_mb']:.1f} MB, {info['files']} files)")
        print(f"{'='*70}\n")


@app.local_entrypoint()
def main(
    folder: str = "0090",
    num_samples: int = 2,
    # repo: str = "omnicad-lab-L3d/stage3-epoch0-step100-20251128_220651",
    repo: str = "omnicad-lab-L3d/stage-3-4096-20251129_213538",
    max_new_tokens: int = 10240,
    min_tokens: int = 2048,
    already_generated: str = "",
):
    """Main entrypoint: Prepare samples and run batch inference.

    Args:
        folder: Data folder (e.g., "0090")
        num_samples: Number of samples to process
        repo: HuggingFace repo ID
        max_new_tokens: Max tokens for generation (default: 10240)
        min_tokens: Minimum GT JSON tokens to include (default: 2048)
        already_generated: Comma-separated list of already generated IDs (e.g., "0090/00900284_00001,0090/00900387_00001")
    """

    # Parse already_generated string into list
    already_gen_list = [s.strip() for s in already_generated.split(',') if s.strip()] if already_generated else []

    print("\n" + "="*70)
    print("BATCH MODAL INFERENCE")
    print("="*70)
    print(f"Max tokens: {max_new_tokens}")
    print(f"Min tokens: {min_tokens}")
    print(f"Already generated: {len(already_gen_list)} samples")

    # Prepare samples on Modal (has access to volume)
    print("\n📦 Preparing samples on Modal...")
    samples = prepare_samples.remote(
        folder=folder,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        min_tokens=min_tokens,
        already_generated=already_gen_list,
    )

    if not samples:
        print("❌ No samples prepared")
        return

    print(f"\n✅ Got {len(samples)} samples:\n")
    for sample in samples:
        print(f"  • {sample['sample_id']}")
        print(f"    Images: {len(sample['image_paths'])} views")
        for img in sample['image_paths'][:3]:
            print(f"      - {os.path.basename(img)}")
        if len(sample['image_paths']) > 3:
            print(f"      ... and {len(sample['image_paths']) - 3} more")
        print(f"    PC: {os.path.basename(sample['pc_path'])}")
        print(f"    Prompt: {sample['prompt'][:60]}...\n")

    # Run inference for each sample
    print("="*70)
    print("STARTING BATCH INFERENCE ON MODAL")
    print("="*70 + "\n")

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] Processing: {sample['sample_id']}")
        result = run_single_inference.remote(
            repo=repo,
            prompt=sample['prompt'],
            image_paths=sample['image_paths'],  # Pass ALL images (multi-view)
            pc_path=sample['pc_path'],
            max_tokens=max_new_tokens,
        )
        results.append(result)
        print(f"      Status: {result.get('status', 'unknown')}\n")

    # Summary
    print("="*70)
    print("BATCH INFERENCE COMPLETE")
    print("="*70)
    print(f"\n📊 RESULTS SUMMARY:")
    print("-" * 70)
    for i, (sample, result) in enumerate(zip(samples, results), 1):
        status = "✓" if result.get("status") == "success" else "⚠"
        print(f"{status} [{i}] {sample['sample_id']}: {result.get('status', 'unknown')}")
        if result.get('json_file'):
            print(f"      JSON: {result.get('json_file')}")
        if result.get('raw_file'):
            print(f"      Raw:  {result.get('raw_file')}")
        print()

    print("-" * 70)
    OUTPUT_VER = "B1_1img_pc_fix_case_set2"
    print(f"\n💾 FILES SAVED TO MODAL VOLUME:")
    print(f"   Raw outputs:  /mnt/data/output_ckpt_5_{OUTPUT_VER}/raw/")
    print(f"   JSON outputs: /mnt/data/output_ckpt_5_{OUTPUT_VER}/json/")
    print(f"\n📥 Download with: ./retrieve_results.sh")
    print("="*70)

    # Zip output directories for download (optional)
    # Uncomment the line below to automatically zip after inference
    zip_output_versions([OUTPUT_VER])


