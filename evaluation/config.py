"""
Configuration file for evaluation scripts.

This allows you to customize paths without modifying the evaluation scripts directly.
"""
import os

# Get the project root directory (parent of evaluation/)
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PC_ROOT = os.path.join(DATA_DIR, 'pc_cad')  # Point cloud data root

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'evaluation', 'results')
MOCK_DATA_DIR = os.path.join(EVAL_DIR, 'mock_data')

# Evaluation parameters
DEFAULT_N_POINTS = 2000  # Number of points for point cloud sampling
TOLERANCE = 3  # Tolerance for parameter accuracy in eval_ae_acc.py

# Data to skip (add problematic data IDs here)
SKIP_DATA = []

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PC_ROOT, exist_ok=True)


def get_pc_path(data_id):
    """
    Get the path to a point cloud file given its data_id.

    Args:
        data_id: 8-character data ID (e.g., '12340001')

    Returns:
        Path to the PLY file
    """
    truck_id = data_id[:4]
    return os.path.join(PC_ROOT, truck_id, f'{data_id}.ply')


def get_h5_output_path(result_dir):
    """
    Get the output path for H5 evaluation results.

    Args:
        result_dir: Directory containing H5 result files

    Returns:
        Path for the statistics file
    """
    return result_dir + "_acc_stat.txt"


# Mock data paths (for testing)
MOCK_H5_DIR = os.path.join(MOCK_DATA_DIR, 'h5_data')
MOCK_SEQ_PKL = os.path.join(MOCK_DATA_DIR, 'seq_data', 'mock_sequences.pkl')
MOCK_MESH_DIR = os.path.join(MOCK_DATA_DIR, 'meshes')


if __name__ == '__main__':
    # Print configuration
    print("Evaluation Configuration:")
    print("=" * 60)
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Evaluation Dir:   {EVAL_DIR}")
    print(f"Data Dir:         {DATA_DIR}")
    print(f"Point Cloud Root: {PC_ROOT}")
    print(f"Output Dir:       {OUTPUT_DIR}")
    print(f"Mock Data Dir:    {MOCK_DATA_DIR}")
    print("=" * 60)

    # Check which directories exist
    print("\nDirectory Status:")
    for name, path in [
        ('Data', DATA_DIR),
        ('Point Clouds', PC_ROOT),
        ('Output', OUTPUT_DIR),
        ('Mock Data', MOCK_DATA_DIR),
        ('Mock H5', MOCK_H5_DIR),
        ('Mock Meshes', MOCK_MESH_DIR),
    ]:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name:15s} {path}")
