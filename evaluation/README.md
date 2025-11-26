# CAD Evaluation Methods

## Available Metrics

### 1. Command Accuracy (eval_ae_acc.py)
- **What**: Measures how accurately your model predicts CAD commands and parameters
- **Metrics**: ACC_cmd (command accuracy), ACC_param (parameter accuracy)
- **Needs**: Mock CAD definitions included, cadlib optional for real definitions
- **Status**: ✓ Works with mock data

### 2. Topology Metrics (eval_topology.py)
- **What**: Evaluates mesh quality (boundaries, intersections, closedness)
- **Metrics**:
  - **SegE**: Difference in number of mesh components
  - **DangEL**: Length of boundary edges (0 = closed mesh)
  - **SIR**: Percentage of self-intersecting faces
  - **FluxEE**: How well the mesh is enclosed
- **Needs**: trimesh only
- **Status**: ✓ Fully functional

### 3. Sequence Evaluation (eval_seq.py)
- **What**: Evaluates CAD sequence generation (lines, arcs, circles, extrusions)
- **Metrics**: Recall, Precision, F1 scores for each primitive type
- **Needs**: CadSeqProc (external) for full evaluation
- **Status**: ✓ Data validation works without dependencies

### 4. Chamfer Distance (eval_ae_cd.py)
- **What**: Measures point cloud similarity between generated and ground truth
- **Metrics**: CD (Chamfer Distance - lower is better)
- **Needs**:
  - cadlib to convert CAD → point cloud
  - Ground truth PLY files
- **Status**: ⚠ Core metric works, needs cadlib for full pipeline

---

## Quick Start

### Test Everything
```bash
cd evaluation
pip install -r ../requirements.txt
python test_eval_methods.py
```

### Visualize Mock Data
```bash
# Quick text view
python quick_view.py

# Full graphical visualization (creates PNG images)
python visualize_mock_data.py
```

### Test Individual Methods

**1. Command Accuracy** (works immediately)
```bash
python generate_mock_data.py --output_dir ./mock_data
python eval_ae_acc.py --src mock_data/h5_data
cat mock_data/h5_data_acc_stat.txt
```

**2. Sequence Evaluation** (data validation only)
```bash
python eval_seq.py \
    --input_path mock_data/seq_data/mock_sequences.pkl \
    --output_dir ./results \
    --verbose
```

**3. Chamfer Distance** (needs cadlib)
```bash
# First install cadlib (see below)
python eval_ae_cd.py --src mock_data/h5_data --n_points 2000
```

---

## Data Formats

### H5 Files (for eval_ae_acc.py, eval_ae_cd.py)
```python
with h5py.File('prediction.h5', 'w') as f:
    f.create_dataset('out_vec', data=predictions)  # Shape: (N, 15)
    f.create_dataset('gt_vec', data=ground_truth)  # Shape: (N, 15)
```

### Pickle Files (for eval_seq.py)
```python
data = {
    "uid_00001": {
        "level_1": {
            "gt_cad_vec": ground_truth_array,
            "pred_cad_vec": [pred1, pred2, pred3],  # Multiple predictions
            "cd": [0.001, 0.002, 0.003]  # Chamfer distances
        }
    }
}
```

### Mesh Files (for eval_topology.py)
```python
import trimesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export('output.obj')
```

---

## External Dependencies (Optional)

### Install cadlib
```bash
git clone https://github.com/DavidXu-JJ/DeepCAD.git
cd DeepCAD && git checkout tags/CAD-MLLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Install CadSeqProc
```bash
git clone https://github.com/SadilKhan/Text2CAD.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/Text2CAD
```

---

## What Works Without External Dependencies

| Evaluation | Works? | What You Can Do |
|------------|--------|-----------------|
| **eval_ae_acc.py** | ✓ Yes | Full accuracy evaluation with mock CAD definitions |
| **eval_topology.py** | ✓ Yes | All 4 topology metrics (SegE, DangEL, SIR, FluxEE) |
| **eval_seq.py** | ✓ Partial | Validate data format, check structure |
| **eval_ae_cd.py** | ⚠ Partial | Load H5 files, needs cadlib for CAD→point cloud |

---

## Files

**Evaluation Scripts:**
- **eval_ae_acc.py** - Command/parameter accuracy
- **eval_ae_cd.py** - Chamfer distance
- **eval_seq.py** - Sequence metrics
- **eval_topology.py** - Mesh topology metrics

**Utilities:**
- **utils.py** - PLY reader and utilities
- **config.py** - Path configuration

**Testing & Visualization:**
- **generate_mock_data.py** - Generate test data
- **test_eval_methods.py** - Test all methods
- **visualize_mock_data.py** - Create graphical visualizations (PNG)
- **quick_view.py** - Quick text-based data viewer

---

## References

- [DeepCAD Evaluation](https://github.com/DavidXu-JJ/DeepCAD/tree/tags/CAD-MLLM/evaluation)
- [Text2CAD Evaluation](https://github.com/SadilKhan/Text2CAD/tree/main/Evaluation)
- [CAD-MLLM Paper](https://arxiv.org/abs/2411.04954)