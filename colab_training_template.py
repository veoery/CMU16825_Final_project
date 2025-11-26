"""Google Colab Training Template for CAD-MLLM

Copy this code into your Google Colab notebook cells and adapt as needed.
"""

# ============================================================================
# CELL 1: Mount Google Drive
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2: Install dependencies (if needed)
# ============================================================================
# %pip install -q torch transformers peft accelerate pillow numpy trimesh

# ============================================================================
# CELL 3: Setup paths and import dataloader
# ============================================================================
import os
import glob
import json
import torch
from pathlib import Path

# ADJUST THESE PATHS TO YOUR GOOGLE DRIVE STRUCTURE
ROOT = "/content/drive/MyDrive/CAD_Project"

truncated_dir = f"{ROOT}/truncated_json"
full_dir = f"{ROOT}/full_json"
image_dir = f"{ROOT}/images"
pc_dir = f"{ROOT}/point_clouds"

# Create path lists
truncated_paths = sorted(glob.glob(truncated_dir + "/*.json"))
full_paths = []

for tr in truncated_paths:
    base = os.path.basename(tr).split("_tr_")[0] + ".json"
    full_paths.append(os.path.join(full_dir, base))

print(f"Found {len(truncated_paths)} training samples")
print(f"First truncated: {Path(truncated_paths[0]).name}")
print(f"First full: {Path(full_paths[0]).name}")

# ============================================================================
# CELL 4: Import model and encoders
# ============================================================================
from cad_mllm import CADMLLMModel, Config
from cad_mllm.encoders import ImageEncoder, MichelangeloPointEncoder
from cad_mllm.data import get_autocomplete_dataloader
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# ============================================================================
# CELL 5: Initialize encoders (optional)
# ============================================================================
# Option 1: Use pre-trained encoders (recommended)
device = "cuda" if torch.cuda.is_available() else "cpu"

image_encoder = ImageEncoder(
    model_name="facebook/dinov2-large",
    torch_dtype=torch.float32,
    freeze=True
).to(device)

# For point cloud encoder, you need the config and state dict
# Adjust these paths to where you have them in Google Drive
pc_encoder = MichelangeloPointEncoder(
    encoder_cfg_path=f"{ROOT}/configs/michelangelo_point_encoder_cfg.yaml",
    encoder_sd_path=f"{ROOT}/models/michelangelo_encoder.pt",
    freeze=True,
    device=device
)

# ============================================================================
# CELL 6: Create dataloader
# ============================================================================
train_loader = get_autocomplete_dataloader(
    truncated_paths=truncated_paths,
    full_paths=full_paths,
    image_dir=image_dir,
    pc_dir=pc_dir,
    tokenizer=tokenizer,
    image_encoder=image_encoder,
    pc_encoder=pc_encoder,
    batch_size=4,
    shuffle=True,
    max_seq_length=512
)

print(f"DataLoader created with {len(train_loader)} batches")

# ============================================================================
# CELL 7: Initialize model
# ============================================================================
config = Config(
    llm_model_name="Qwen/Qwen2.5-7B-Instruct",
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,
    freeze_encoders=True
)

model = CADMLLMModel(config).to(device)

# ============================================================================
# CELL 8: Setup training (optimizer, loss, etc.)
# ============================================================================
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * 3)

# ============================================================================
# CELL 9: Training loop
# ============================================================================
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_embeds=batch["img_embeds"],
            pc_embeds=batch["pc_embeds"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Loss: {avg_loss:.4f}"
            )

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}\n")

# ============================================================================
# CELL 10: Save model checkpoint
# ============================================================================
checkpoint_dir = f"{ROOT}/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = f"{checkpoint_dir}/model_epoch_3.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")

# ============================================================================
# CELL 11: (Optional) Test inference
# ============================================================================
# model.eval()
# with torch.no_grad():
#     # Sample from first batch
#     batch = next(iter(train_loader))
#     batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
#
#     # Generate CAD sequence
#     generated_ids = model.generate(
#         input_ids=batch["input_ids"][:1],
#         attention_mask=batch["attention_mask"][:1],
#         img_embeds=batch["img_embeds"][:1],
#         pc_embeds=batch["pc_embeds"][:1],
#         max_new_tokens=100
#     )
#
#     generated_text = tokenizer.decode(generated_ids[0])
#     print("Generated CAD sequence:")
#     print(generated_text)
