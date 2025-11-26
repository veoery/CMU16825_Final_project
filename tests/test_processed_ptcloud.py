import numpy as np
import torch
from pathlib import Path

from cad_mllm.encoders.pointcloud_encoder import PointCloudEncoder  # 用安装好的包

def main():
    # 1. 你的本地点云目录：包含很多 .npz 文件
    pc_root = Path(r"test_npz")

    pc_files = sorted(pc_root.rglob("*.npz"))[:4]   # 取前 4 个测试
    print("Testing files:")
    for f in pc_files:
        print("  ", f)

    if not pc_files:
        print("❌ No .npz files found under:", pc_root)
        return

    # 2. 加载 & 堆成 batch (B, N, 3)
    pcs = []
    for f in pc_files:
        data = np.load(f)
        pts = data["points"]         # 应该是 (N, 3)

        print(f.name, "shape:", pts.shape, "dtype:", pts.dtype,
              "min:", pts.min(), "max:", pts.max())

        # 安全检查
        assert pts.ndim == 2 and pts.shape[1] == 3, "Each point cloud must be (N, 3)"
        assert np.isfinite(pts).all(), "NaN/Inf found in point cloud"

        pcs.append(pts.astype("float32"))  # encoder 用 float32

    batch_np = np.stack(pcs, axis=0)       # (B, N, 3)
    print("batch_np shape:", batch_np.shape)

    batch = torch.from_numpy(batch_np)     # torch.Size([B, N, 3])

    # 3. 创建 encoder & 前向
    encoder = PointCloudEncoder(
        input_dim=3,
        hidden_dim=512,
        output_dim=1024,
        freeze=True,     # 测试时先不冻结
    )

    with torch.no_grad():
        feats = encoder(batch)             # (B, 1, 1024)

    print("encoder output shape:", feats.shape)

if __name__ == "__main__":
    main()
