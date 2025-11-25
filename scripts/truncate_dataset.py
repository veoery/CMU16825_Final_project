"""
CAD Sequence Truncation Script
Truncates CAD JSON sequences at specified percentages while maintaining semantic validity.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
import copy


class CADTruncator:
    """Handles intelligent truncation of CAD JSON sequences."""

    def __init__(self, min_operations: int = 1):
        self.min_operations = min_operations

    def load_json(self, filepath: Path) -> Dict:
        """Load CAD JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_json(self, data: Dict, filepath: Path):
        """Save CAD JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def find_valid_truncation_points(self, sequence: List[Dict]) -> List[int]:
        """
        Find valid truncation points in the sequence.
        For training autocompletion models, we allow truncation at any operation.

        Returns list of indices where truncation can occur (inclusive).
        """
        # Allow truncation at any operation index
        # This means we can have orphaned sketches, which is fine for training data
        valid_points = list(range(len(sequence)))

        # Ensure we have at least min_operations
        valid_points = [p for p in valid_points if p + 1 >= self.min_operations]

        return valid_points

    def check_orphaned_sketches(self, sequence: List[Dict], truncate_at: int) -> bool:
        """
        Check if truncating at this point would create orphaned sketches.
        For autocompletion training, we allow orphaned sketches, so always return False.
        Returns True if there are orphaned sketches (bad truncation).
        """
        # For training autocompletion models, orphaned sketches are acceptable
        return False

    def get_referenced_entities(self, sequence: List[Dict], entities: Dict) -> Set[str]:
        """
        Get all entity IDs referenced in the sequence.
        Also includes entities referenced within those entities (like profiles).
        """
        referenced = set()

        for op in sequence:
            entity_id = op.get('entity')
            if entity_id:
                referenced.add(entity_id)

                # Recursively get referenced entities within this entity
                self._collect_nested_references(entity_id, entities, referenced)

        return referenced

    def _collect_nested_references(self, entity_id: str, entities: Dict, referenced: Set[str]):
        """Recursively collect all entities referenced within an entity."""
        if entity_id not in entities:
            return

        entity = entities[entity_id]

        # Check profiles in ExtrudeFeature
        if 'profiles' in entity:
            for profile in entity['profiles']:
                if 'sketch' in profile:
                    sketch_id = profile['sketch']
                    if sketch_id not in referenced:
                        referenced.add(sketch_id)
                        self._collect_nested_references(sketch_id, entities, referenced)

        # Check any other reference fields (add more as needed)
        # For now, profiles and sketches are the main ones

    def calculate_truncation_points(self, total_ops: int, percentages: List[float]) -> List[int]:
        """
        Calculate operation counts for each percentage.
        Round up and ensure minimum operations.

        Returns list of operation counts (not indices).
        """
        points = []
        for pct in percentages:
            count = math.ceil(total_ops * pct)
            count = max(count, self.min_operations)
            count = min(count, total_ops)  # Don't exceed total
            points.append(count)

        return points

    def select_best_truncation_point(self, target_count: int, valid_points: List[int]) -> int:
        """
        Select the best valid truncation point closest to target count.
        Returns index (0-based), where truncation includes this index.
        """
        if not valid_points:
            # Fallback: return target - 1 (as index)
            return max(0, target_count - 1)

        # Target index is target_count - 1 (since we're keeping target_count operations)
        target_idx = target_count - 1

        # Find closest valid point that doesn't exceed target (or is closest)
        best_point = None
        min_diff = float('inf')

        for point in valid_points:
            # Prefer points at or before target
            if point <= target_idx:
                diff = target_idx - point
                if diff < min_diff:
                    min_diff = diff
                    best_point = point

        # If no point before target, take the first valid point after
        if best_point is None:
            for point in valid_points:
                if point > target_idx:
                    best_point = point
                    break

        # Last resort: use target_idx itself
        if best_point is None:
            best_point = min(target_idx, len(valid_points) - 1)

        return best_point

    def truncate_json(self, data: Dict, truncate_at_idx: int) -> Dict:
        """
        Truncate JSON at specified index (inclusive).
        Returns new truncated JSON.
        """
        truncated = copy.deepcopy(data)

        original_sequence = data['sequence']
        total_ops = len(original_sequence)

        # Step 1: Truncate sequence
        truncated['sequence'] = original_sequence[:truncate_at_idx + 1]
        kept_ops = len(truncated['sequence'])

        # Step 2: Get referenced entities
        referenced_entities = self.get_referenced_entities(
            truncated['sequence'],
            data['entities']
        )

        # Step 3: Clean entities dictionary
        truncated['entities'] = {
            eid: entity
            for eid, entity in data['entities'].items()
            if eid in referenced_entities
        }

        # Step 4: Add truncation metadata
        truncation_pct = (kept_ops / total_ops * 100) if total_ops > 0 else 0

        truncated['truncation_metadata'] = {
            'is_truncated': True,
            'original_operations': total_ops,
            'kept_operations': kept_ops,
            'truncation_percentage': round(truncation_pct, 2)
        }

        return truncated

    def generate_truncations(
        self,
        input_path: Path,
        output_dir: Path,
        max_versions: int = 5
    ) -> List[Tuple[Path, Dict]]:
        """
        Generate multiple truncated versions of a CAD JSON.
        Creates up to max_versions (default 5) evenly-spaced truncations.
        Versions are numbered consecutively from tr_01 to tr_05.

        Logic:
        - 2 ops → tr_01 (1 op)
        - 3 ops → tr_01, tr_02 (1, 2 ops)
        - 4 ops → tr_01, tr_02, tr_03 (1, 2, 3 ops)
        - 10 ops → tr_01, tr_02, tr_03, tr_04, tr_05 (1, 3, 5, 7, 9 ops)

        Args:
            input_path: Path to input JSON file
            output_dir: Directory to save truncated files
            max_versions: Maximum number of truncation versions (default: 5)

        Returns:
            list of (output_path, truncated_data) tuples.
        """
        # Load original
        original_data = self.load_json(input_path)
        sequence = original_data.get('sequence', [])
        total_ops = len(sequence)

        # Need at least 2 operations to truncate (keep min 1, have something to remove)
        if total_ops <= 1:
            return []

        # Find valid truncation points
        valid_points = self.find_valid_truncation_points(sequence)
        if not valid_points:
            return []

        # Filter valid points to exclude 100%
        valid_points = [p for p in valid_points if p + 1 < total_ops]
        if not valid_points:
            return []

        # Determine how many versions we can create
        # Possible operations: [1, 2, ..., total_ops-1]
        max_possible_ops = total_ops - 1
        num_versions = min(max_possible_ops, max_versions)

        if num_versions < 1:
            return []

        # Calculate evenly-spaced target operation counts starting from 1
        if num_versions == 1:
            target_counts = [1]
        else:
            step_size = (max_possible_ops - 1) / (num_versions - 1)
            target_counts = [
                max(1, round(1 + i * step_size))
                for i in range(num_versions)
            ]

        # Generate truncations with consecutive numbering
        results = []
        stem = input_path.stem
        seen_operation_counts = set()

        for target_count in target_counts:
            # Select best truncation point
            truncate_idx = self.select_best_truncation_point(target_count, valid_points)
            actual_ops = truncate_idx + 1

            # Skip if 100%
            if actual_ops >= total_ops:
                continue

            # Skip duplicates
            if actual_ops in seen_operation_counts:
                continue

            # Check for orphaned sketches
            if self.check_orphaned_sketches(sequence, truncate_idx):
                valid_before = [p for p in valid_points if p < truncate_idx]
                if valid_before:
                    truncate_idx = valid_before[-1]
                    actual_ops = truncate_idx + 1
                    if actual_ops in seen_operation_counts or actual_ops >= total_ops:
                        continue
                else:
                    continue

            # Truncate
            truncated_data = self.truncate_json(original_data, truncate_idx)
            kept_ops = truncated_data['truncation_metadata']['kept_operations']
            seen_operation_counts.add(kept_ops)

            # Generate output path with consecutive numbering
            output_filename = f"{stem}_tr_{len(results)+1:02d}.json"
            output_path = output_dir / output_filename

            # Save
            self.save_json(truncated_data, output_path)
            results.append((output_path, truncated_data))

            print(f"  Created {output_filename}: {kept_ops}/{total_ops} ops ({truncated_data['truncation_metadata']['truncation_percentage']:.1f}%)")

        return results


def main():
    """Example usage."""
    truncator = CADTruncator(min_operations=1)

    # Example: truncate a single file
    input_file = Path("data/Omni-CAD-subset/json/0021/00210058_00006.json")
    output_dir = Path("data/Omni-CAD-subset/json_truncated_test/0021")

    if input_file.exists():
        print(f"Truncating {input_file.name}...")
        results = truncator.generate_truncations(input_file, output_dir)
        print(f"Generated {len(results)} truncated versions.")
    else:
        print(f"File not found: {input_file}")


if __name__ == "__main__":
    main()
