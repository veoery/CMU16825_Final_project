"""
Script to analyze Omni-CAD dataset structure and create subsets.
"""
import os
import random
import json
from pathlib import Path
from collections import defaultdict
import shutil

def analyze_dataset(data_dir="data/Omni-CAD"):
    """Analyze the structure of the Omni-CAD dataset."""
    json_dir = Path(data_dir) / "json"
    step_dir = Path(data_dir) / "step"
    
    # Get all batches
    json_batches = sorted([d.name for d in json_dir.iterdir() if d.is_dir()])
    step_batches = sorted([d.name for d in step_dir.iterdir() if d.is_dir()])
    
    print(f"Number of JSON batches: {len(json_batches)}")
    print(f"Number of STEP batches: {len(step_batches)}")
    print(f"Batches match: {json_batches == step_batches}")
    
    # Count files per batch
    batch_counts = {}
    total_files = 0
    
    for batch in json_batches:
        json_files = list((json_dir / batch).glob("*.json"))
        step_files = list((step_dir / batch).glob("*.step"))
        batch_counts[batch] = {
            'json': len(json_files),
            'step': len(step_files)
        }
        total_files += len(json_files)
    
    print(f"\nTotal JSON files: {total_files}")
    print(f"\nSample batch statistics (first 10 batches):")
    for batch in json_batches[:10]:
        print(f"  Batch {batch}: {batch_counts[batch]['json']} JSON, {batch_counts[batch]['step']} STEP")
    
    # Check if files are random or ordered
    print(f"\nAnalyzing file naming pattern...")
    sample_files = []
    for batch in json_batches[:5]:
        files = sorted([f.name for f in (json_dir / batch).glob("*.json")])
        sample_files.extend(files[:5])
    
    print(f"Sample file names (first 25):")
    for f in sample_files[:25]:
        print(f"  {f}")
    
    # Check if files are distributed randomly
    print(f"\nChecking distribution pattern...")
    file_ids_by_batch = defaultdict(list)
    for batch in json_batches:
        files = [f.name for f in (json_dir / batch).glob("*.json")]
        for f in files:
            # Extract file ID (middle part before version)
            parts = f.split('_')
            if len(parts) >= 2:
                file_id = parts[0]
                file_ids_by_batch[batch].append(file_id)
    
    # Check if file IDs are sequential or random
    print(f"File IDs in batch 0000 (first 20): {sorted(set(file_ids_by_batch['0000']))[:20]}")
    print(f"File IDs in batch 0050 (first 20): {sorted(set(file_ids_by_batch['0050']))[:20]}")
    
    return json_batches, batch_counts, total_files


def load_txt_captions(txt_dir, batch):
    """Load text captions from txt/{batch}.json file."""
    txt_file = txt_dir / f"{batch}.json"
    if not txt_file.exists():
        return {}
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        # Create a dictionary mapping file_id to caption entry
        caption_dict = {}
        for entry in captions:
            file_id = entry.get('id', '').split('/')[-1]  # Extract filename from "batch/filename"
            caption_dict[file_id] = entry
        return caption_dict
    except Exception as e:
        print(f"Warning: Could not load captions from {txt_file}: {e}")
        return {}


def create_subset(data_dir="data/Omni-CAD", output_dir="data/Omni-CAD-subset", percentage=10, seed=42):
    """
    Create a subset of the Omni-CAD dataset.
    
    Args:
        data_dir: Source directory
        output_dir: Destination directory for subset
        percentage: Percentage of data to include (e.g., 10 for 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    json_dir = Path(data_dir) / "json"
    step_dir = Path(data_dir) / "step"
    txt_dir = Path(data_dir) / "txt"
    output_json_dir = Path(output_dir) / "json"
    output_step_dir = Path(output_dir) / "step"
    output_txt_dir = Path(output_dir) / "txt"
    
    # Create output directories
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_step_dir.mkdir(parents=True, exist_ok=True)
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all batches
    batches = sorted([d.name for d in json_dir.iterdir() if d.is_dir()])
    
    # Collect all file pairs (json + step)
    all_files = []
    for batch in batches:
        json_files = list((json_dir / batch).glob("*.json"))
        for json_file in json_files:
            # Find corresponding step file
            step_file = step_dir / batch / json_file.name.replace('.json', '.step')
            if step_file.exists():
                all_files.append((batch, json_file, step_file))
    
    print(f"Total file pairs found: {len(all_files)}")
    
    # Sample subset
    num_samples = int(len(all_files) * percentage / 100)
    sampled_files = random.sample(all_files, num_samples)
    
    print(f"Sampling {num_samples} files ({percentage}% of {len(all_files)})")
    
    # Load text captions for each batch (lazy loading)
    batch_captions = {}
    
    # Copy files and collect captions
    copied = 0
    captions_by_batch = defaultdict(list)  # Store captions for each batch
    
    for batch, json_file, step_file in sampled_files:
        # Create batch directory in output
        (output_json_dir / batch).mkdir(parents=True, exist_ok=True)
        (output_step_dir / batch).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        shutil.copy2(json_file, output_json_dir / batch / json_file.name)
        shutil.copy2(step_file, output_step_dir / batch / step_file.name)
        copied += 1
        
        # Load captions for this batch if not already loaded
        if batch not in batch_captions:
            batch_captions[batch] = load_txt_captions(txt_dir, batch)
        
        # Extract caption for this file
        file_id = json_file.stem  # filename without extension
        if file_id in batch_captions[batch]:
            captions_by_batch[batch].append(batch_captions[batch][file_id])
        
        if copied % 100 == 0:
            print(f"  Copied {copied}/{num_samples} files...")
    
    # Write txt JSON files for each batch
    print(f"\nWriting text caption files...")
    for batch, captions in captions_by_batch.items():
        if captions:
            txt_output_file = output_txt_dir / f"{batch}.json"
            with open(txt_output_file, 'w', encoding='utf-8') as f:
                json.dump(captions, f, indent=4, ensure_ascii=False)
            print(f"  Wrote {len(captions)} captions to {txt_output_file}")
    
    print(f"\nSubset created successfully!")
    print(f"  Source: {data_dir}")
    print(f"  Destination: {output_dir}")
    print(f"  Files copied: {copied}")
    print(f"  Batches with captions: {len(captions_by_batch)}")
    
    return output_dir


def create_subset_by_batches(data_dir="data/Omni-CAD", output_dir="data/Omni-CAD-subset", 
                             num_batches=10, seed=42):
    """
    Create a subset by sampling entire batches.
    
    Args:
        data_dir: Source directory
        output_dir: Destination directory for subset
        num_batches: Number of batches to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    json_dir = Path(data_dir) / "json"
    step_dir = Path(data_dir) / "step"
    txt_dir = Path(data_dir) / "txt"
    output_json_dir = Path(output_dir) / "json"
    output_step_dir = Path(output_dir) / "step"
    output_txt_dir = Path(output_dir) / "txt"
    
    # Create output directories
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_step_dir.mkdir(parents=True, exist_ok=True)
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all batches
    batches = sorted([d.name for d in json_dir.iterdir() if d.is_dir()])
    
    # Sample batches
    sampled_batches = random.sample(batches, min(num_batches, len(batches)))
    print(f"Sampling {len(sampled_batches)} batches: {sampled_batches}")
    
    # Copy entire batches
    total_files = 0
    for batch in sampled_batches:
        # Create batch directory in output
        (output_json_dir / batch).mkdir(parents=True, exist_ok=True)
        (output_step_dir / batch).mkdir(parents=True, exist_ok=True)
        
        # Copy all files in batch
        json_files = list((json_dir / batch).glob("*.json"))
        for json_file in json_files:
            step_file = step_dir / batch / json_file.name.replace('.json', '.step')
            if step_file.exists():
                shutil.copy2(json_file, output_json_dir / batch / json_file.name)
                shutil.copy2(step_file, output_step_dir / batch / step_file.name)
                total_files += 1
        
        # Copy corresponding txt file
        txt_file = txt_dir / f"{batch}.json"
        if txt_file.exists():
            shutil.copy2(txt_file, output_txt_dir / f"{batch}.json")
            print(f"  Copied captions for batch {batch}")
    
    print(f"\nSubset created successfully!")
    print(f"  Source: {data_dir}")
    print(f"  Destination: {output_dir}")
    print(f"  Batches: {len(sampled_batches)}")
    print(f"  Files copied: {total_files}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and create subsets of Omni-CAD dataset")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset structure")
    parser.add_argument("--subset", type=int, help="Create subset with given percentage (e.g., 10 for 10%%)")
    parser.add_argument("--subset-batches", type=int, help="Create subset by sampling N batches")
    parser.add_argument("--data-dir", default="data/Omni-CAD", help="Source data directory")
    parser.add_argument("--output-dir", default="data/Omni-CAD-subset", help="Output directory for subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.data_dir)
    elif args.subset:
        create_subset(args.data_dir, args.output_dir, args.subset, args.seed)
    elif args.subset_batches:
        create_subset_by_batches(args.data_dir, args.output_dir, args.subset_batches, args.seed)
    else:
        print("Please specify --analyze, --subset, or --subset-batches")
        print("\nExamples:")
        print("  python analyze_omni_cad.py --analyze")
        print("  python analyze_omni_cad.py --subset 10")
        print("  python analyze_omni_cad.py --subset-batches 10")

