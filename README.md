# CAD-MLLM Reproduction

This is a reproduction of the paper "CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM" ([arXiv:2411.04954](https://arxiv.org/abs/2411.04954)).


## Current Implementation Status

- [x] Project structure and environment setup
- [x] Text-only input pipeline
- [x] Image encoder integration
- [ ] Point cloud encoder integration
- [ ] Training pipeline
  - [x] LoRA for LLM (text only for now)

- [ ] Evaluation metrics

## Setup

### Prerequisites
- Python >= 3.10
- `uv` package manager

### Installation

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -e .
```

### Dateset

Follow the original repo for the dataset: 
https://github.com/CAD-MLLM/CAD-MLLM?tab=readme-ov-file#data



## Usage

### Text-to-CAD Generation

```bash
python scripts/inference.py --prompt "Generate a CAD model of a simple cube." --device mps --image_path "data/Omni-CAD/img/cube.jpeg" --dtype bfloat16
```

Check the scripts/inference.py and config.py for more details.



### LoRA for LLM (text only for now)

```bash
python scripts/train.py \
    --create_dummy_data \
    --num_dummy_samples 100 \
    --num_epochs 1 \
    --device mps \
    --lora_r 4 \
    --llm_model_name "Qwen/Qwen3-0.6B"
```

```bash
python scripts/train.py \
    --omnicad_txt_path data/Omni-CAD/txt/0000.json \
    --omnicad_json_root data/Omni-CAD/json \
    --num_epochs 3 \
    --batch_size 2 \
    --device cuda
```





## Project Structure

```
CAD-MLLM/
├── assets/
├── data/                   # follow the original repo for the dataset
├── cad_mllm/
│   ├── __init__.py
│   ├── config.py           # Configuration classes
│   ├── model.py            # Main CAD-MLLM model
│   ├── encoders/           # Modality encoders
│   │   ├── __init__.py
│   │   ├── text_encoder.py
│   │   ├── image_encoder.py
│   │   └── pointcloud_encoder.py
│   ├── projectors/         # Projection layers
│   │   ├── __init__.py
│   │   └── mlp_projector.py
├── scripts/
│   └── inference.py       # Inference script
├── pyproject.toml
└── README.md
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@misc{xu2024CADMLLM,
    title={CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM},
    author={Jingwei Xu and Chenyu Wang and Zibo Zhao and Wen Liu and Yi Ma and Shenghua Gao},
    year={2024},
    eprint={2411.04954},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

This is a research reproduction for educational purposes.
