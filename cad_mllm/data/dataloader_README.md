# Dataloader Implementation
test colab notebook: `https://colab.research.google.com/drive/1e1h205yykYG7mr-qV4xfiYN9Xfbyjg7v?usp=sharing`

## Key Features

1. **On-the-Fly Encoding**: No need to pre-compute and store embeddings
   - Images → DINOv2 → embeddings (on GPU)
   - Point clouds → Michelangelo → embeddings (on GPU)

2. **Multi-view Support**: Handles single or multiple images per sample
   - Single view: `sample.png`
   - Multi-view: `sample_000.png`, `sample_001.png`, etc.

3. **Flexible Input Formats**:
   - Images: `.png`, `.jpg`, `.jpeg`
   - Point clouds: `.npy`, `.ply`, `.obj`, `.npz`

4. **Batch Contents**:
   ```python
   {
       "input_ids": Tensor,        # Tokenized CAD instructions
       "attention_mask": Tensor,
       "img_embeds": Tensor,       # Image embeddings
       "pc_embeds": Tensor,        # Point cloud embeddings
       "labels": Tensor,           # For training loss
   }
   ```

## How to Use (Colab)
The dataloader correctly:
- Loads truncated JSON and full JSON pairs (_tr_XX.json → .json).
- Finds multi-view images per CAD sample (e.g. 00000071_00005_000.png ~ _007.png).
- Loads point clouds from the corresponding .npz files.

From the validation run, the batch shapes are:
img_embeds: (B, V, 3, H, W) = (2, 8, 3, 768, 1024)
→ B = batch size, V = number of views (8 per CAD), 3 channels (RGB), 768×1024 resolution.
→ These are raw image tensors, not visual embeddings yet.

pc_embeds: (B, 2048, 3) = (2, 2048, 3)
→ 2048 3D points per sample (x, y, z).
→ These are raw point clouds, not point embeddings yet.