"""Analyze token lengths in the training data to determine optimal max_seq_length."""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Paths
DATA_ROOT = Path("data/Omni-CAD-subset")
TXT_PATH = DATA_ROOT / "txt"
TRUNCATED_JSON_ROOT = DATA_ROOT / "json_truncated"
FULL_JSON_ROOT = DATA_ROOT / "json"

# Initialize tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # Faster loading for analysis
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

def load_text_captions():
    """Load text captions from JSON files."""
    text_captions = {}
    if TXT_PATH.is_dir():
        for caption_file in sorted(TXT_PATH.glob("*.json")):
            with open(caption_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    text_captions[entry["id"]] = entry["text caption"]
    return text_captions

def analyze_dataset(max_samples=None):
    """Analyze token lengths in the dataset."""

    print("\nLoading text captions...")
    text_captions = load_text_captions()
    print(f"Loaded {len(text_captions)} captions")

    print(f"\nFinding truncated JSON files in {TRUNCATED_JSON_ROOT}...")
    truncated_files = sorted(TRUNCATED_JSON_ROOT.rglob("*_tr_*.json"))

    if max_samples:
        truncated_files = truncated_files[:max_samples]

    print(f"Found {len(truncated_files)} truncated JSON files")

    # Statistics
    prompt_lengths = []
    full_lengths = []
    truncated_lengths = []

    skipped = 0

    print("\nAnalyzing token lengths...")
    for truncated_path in tqdm(truncated_files, desc="Processing"):
        try:
            # Extract CAD ID
            rel_path = truncated_path.relative_to(TRUNCATED_JSON_ROOT)
            stem = rel_path.stem
            base_name = "_".join(stem.split("_")[:-2])
            parent = rel_path.parent
            cad_id = str(parent / base_name)

            # Load truncated JSON
            with open(truncated_path, 'r') as f:
                truncated_json = json.load(f)

            # Load full JSON
            full_json_path = FULL_JSON_ROOT / f"{cad_id}.json"
            if not full_json_path.exists():
                skipped += 1
                continue

            with open(full_json_path, 'r') as f:
                full_json = json.load(f)

            # Get caption
            caption = text_captions.get(cad_id, "")

            # Format exactly as in training (autocomplete_2 masking strategy)
            # This uses truncated JSON for prompt, full JSON for completion
            truncated_entities_only = {"entities": truncated_json.get("entities", {})}
            truncated_seq = json.dumps(truncated_entities_only, separators=(',', ':'))
            full_seq = json.dumps(full_json, separators=(',', ':'))

            # Tokenize prompt (truncated)
            prompt = f"Complete this CAD sequence: {caption}\n{truncated_seq}"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]

            # Tokenize full sequence
            full_text = f"Complete this CAD sequence: {caption}\n{full_seq}"
            full_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            # Tokenize just truncated seq
            trunc_only_tokens = tokenizer(truncated_seq, add_special_tokens=False)["input_ids"]

            prompt_lengths.append(len(prompt_tokens))
            full_lengths.append(len(full_tokens))
            truncated_lengths.append(len(trunc_only_tokens))

        except Exception as e:
            skipped += 1
            continue

    print(f"\nProcessed {len(prompt_lengths)} samples ({skipped} skipped)")

    # Convert to numpy for statistics
    prompt_lengths = np.array(prompt_lengths)
    full_lengths = np.array(full_lengths)
    truncated_lengths = np.array(truncated_lengths)

    # Calculate statistics
    print("\n" + "="*80)
    print("TOKEN LENGTH ANALYSIS (autocomplete_2 masking strategy)")
    print("="*80)

    print("\n[PROMPT LENGTHS] (what gets masked during training):")
    print(f"  Mean:       {prompt_lengths.mean():.1f}")
    print(f"  Median:     {np.median(prompt_lengths):.1f}")
    print(f"  Std Dev:    {prompt_lengths.std():.1f}")
    print(f"  Min:        {prompt_lengths.min()}")
    print(f"  Max:        {prompt_lengths.max()}")
    print(f"  Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.5]:
        val = np.percentile(prompt_lengths, p)
        pct = (prompt_lengths <= val).sum() / len(prompt_lengths) * 100
        print(f"    {p:5.1f}%: {val:7.0f} tokens ({pct:.1f}% of samples)")

    print("\n[FULL SEQUENCE LENGTHS] (total input during training):")
    print(f"  Mean:       {full_lengths.mean():.1f}")
    print(f"  Median:     {np.median(full_lengths):.1f}")
    print(f"  Std Dev:    {full_lengths.std():.1f}")
    print(f"  Min:        {full_lengths.min()}")
    print(f"  Max:        {full_lengths.max()}")
    print(f"  Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.5]:
        val = np.percentile(full_lengths, p)
        pct = (full_lengths <= val).sum() / len(full_lengths) * 100
        print(f"    {p:5.1f}%: {val:7.0f} tokens ({pct:.1f}% of samples)")

    print("\n[TRUNCATED ENTITIES ONLY] (what teammate's masking uses):")
    print(f"  Mean:       {truncated_lengths.mean():.1f}")
    print(f"  Median:     {np.median(truncated_lengths):.1f}")
    print(f"  Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(truncated_lengths, p)
        print(f"    {p:5.1f}%: {val:7.0f} tokens")

    # Coverage analysis
    print("\n[COVERAGE ANALYSIS]:")
    for max_len in [2048, 4096, 6000, 8000, 10000, 13000]:
        coverage = (full_lengths <= max_len).sum() / len(full_lengths) * 100
        num_fit = (full_lengths <= max_len).sum()
        num_truncated = len(full_lengths) - num_fit
        print(f"  max_seq_length = {max_len:5d}: {coverage:5.1f}% coverage ({num_fit}/{len(full_lengths)} samples, {num_truncated} truncated)")

    # Learning token analysis
    learning_tokens = full_lengths - prompt_lengths
    print("\n[LEARNING TOKENS] (tokens model actually learns from):")
    print(f"  Mean:       {learning_tokens.mean():.1f}")
    print(f"  Median:     {np.median(learning_tokens):.1f}")
    print(f"  Min:        {learning_tokens.min()}")
    print(f"  Max:        {learning_tokens.max()}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    coverage_4k = (full_lengths <= 4096).sum() / len(full_lengths) * 100
    coverage_8k = (full_lengths <= 8000).sum() / len(full_lengths) * 100

    print(f"\n1. max_seq_length = 4096:")
    print(f"   [+] Covers {coverage_4k:.1f}% of samples")
    print(f"   [+] Can use higher LR (2e-4, proven by teammate)")
    print(f"   [+] Faster training convergence")
    print(f"   [-] Truncates {100-coverage_4k:.1f}% of samples")

    print(f"\n2. max_seq_length = 8000:")
    print(f"   [+] Covers {coverage_8k:.1f}% of samples")
    print(f"   [-] Requires lower LR (1.4e-4, NOT 2e-5!)")
    print(f"   [-] Slower convergence")
    print(f"   [-] Higher memory usage")

    median_full = np.median(full_lengths)
    p95_full = np.percentile(full_lengths, 95)

    if median_full < 4096:
        print(f"\n[VERDICT] Use max_seq_length = 4096")
        print(f"   Median length ({median_full:.0f}) is well below 4096")
        print(f"   Most samples fit comfortably")
    elif p95_full < 4096:
        print(f"\n[VERDICT] Use max_seq_length = 4096")
        print(f"   95th percentile ({p95_full:.0f}) is below 4096")
    else:
        print(f"\n[VERDICT] Consider max_seq_length = {int(p95_full)}")
        print(f"   95th percentile is {p95_full:.0f} tokens")

    return {
        'prompt_lengths': prompt_lengths,
        'full_lengths': full_lengths,
        'truncated_lengths': truncated_lengths,
    }

if __name__ == "__main__":
    # Analyze representative sample (2000 samples for quick analysis)
    results = analyze_dataset(max_samples=2000)

    print("\n[DONE] Analysis complete!")
