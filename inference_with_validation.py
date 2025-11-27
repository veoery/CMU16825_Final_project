#!/usr/bin/env python3
"""
Enhanced inference script for CAD generation with validation.

This script performs CAD JSON generation with strict schema validation
to ensure generated outputs meet requirements before saving.

Key improvements:
1. Explicit JSON schema specification in generation prompt
2. Generation-time validation using cad_validation.py
3. Automatic retry on validation failure (up to N attempts)
4. Detailed logging of generation issues
"""

import json
import os
import sys
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

# Add project root to path for imports
sys.path.insert(0, '/root/cmu/16825_l43d/CMU16825_Final_project')

from cad_validation import CADValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED INFERENCE PROMPT WITH STRICT JSON SCHEMA SPECIFICATION
# ============================================================================

GENERATION_PROMPT_WITH_SCHEMA = """You are a CAD program generator.
Given a natural language description of a CAD part, generate a valid Omni-CAD JSON program.

**CRITICAL JSON SCHEMA REQUIREMENTS:**

The JSON must have this exact structure:
{{
  "type": "CADSequence",
  "entities": {{
    "<entity_id>": {{
      "type": "Sketch" or "ExtrudeFeature",
      ... entity-specific fields
    }}
  }},
  "sequence": [
    {{"type": "ExtrudeFeature", "entity": "<extrude_id>"}},
    ...
  ],
  "bounding_box": {{"min_point": {{...}}, "max_point": {{...}}}},
  "properties": {{...}}
}}

**MANDATORY VALIDATION RULES:**

1. **profiles MUST be a DICT (not a list)**
   - Correct: "profiles": {{"profile_0": {{...}}, "profile_1": {{...}}}}
   - WRONG: "profiles": [{{...}}, {{...}}]

2. **ALL extrude features MUST have extent_one.distance.value > 0**
   - Example: "extent_one": {{"distance": {{"value": 5.0}}}}
   - value must be positive number, NOT zero or null

3. **ALL curves MUST have complete definitions**
   - Required: "start_point": {{"x": ..., "y": ..., "z": ...}}
   - Required: "end_point": {{"x": ..., "y": ..., "z": ...}}
   - Required: "type": "Line" or "Arc" or "Circle" etc.

4. **ALL profiles MUST have at least one curve**
   - Each loop must have at least one element in "profile_curves"
   - Curves must be geometrically valid

5. **extrudes MUST reference valid profiles**
   - Each extrude's profiles must reference profiles that exist
   - Referenced profiles must have curves

6. **sketch transforms are required**
   - Each Sketch must have: "transform": {{"origin": {{...}}, "x_axis": {{...}}, ...}}

Description:
{description}

JSON:
"""

# ============================================================================
# VALIDATION AND RETRY LOGIC
# ============================================================================

def validate_generated_json(json_str: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate generated JSON string.

    Returns:
        (is_valid, error_message, parsed_data)
    """
    # 1. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, f"JSON Parse Error: {str(e)[:200]}", None

    # 2. Check entities structure
    entities = data.get('entities')
    if not isinstance(entities, dict):
        return False, f"entities must be dict, got {type(entities).__name__}", None

    if not entities:
        return False, "entities is empty (no sketches or features)", None

    # 3. Check for malformed profiles (LIST instead of DICT)
    for ent_id, ent in entities.items():
        if ent.get('type') == 'Sketch':
            profiles = ent.get('profiles')
            if not isinstance(profiles, dict):
                return False, \
                    f"Sketch {ent_id}: 'profiles' must be dict, got {type(profiles).__name__} (CRITICAL)", \
                    None

    # 4. Run CADValidator checks
    is_valid, all_issues = CADValidator.validate_all(entities, strict=True)

    if not is_valid:
        issue_summary = []
        for check_name, issues in all_issues.items():
            if issues:
                issue_summary.append(f"{check_name}: {len(issues)} issue(s)")
        error_msg = "Validation failed: " + ", ".join(issue_summary)
        return False, error_msg, None

    return True, "Valid", data


def generate_with_retry(
    model,
    processor,
    description: str,
    max_retries: int = 3,
    max_new_tokens: int = 10240,
    temperature: float = 0.2,
    **generation_kwargs
) -> Tuple[bool, Optional[str], Optional[Dict], list]:
    """
    Generate CAD JSON with automatic retry on validation failure.

    Returns:
        (success, json_string, parsed_data, generation_log)
    """
    generation_log = []

    for attempt in range(1, max_retries + 1):
        logger.info(f"Generation attempt {attempt}/{max_retries}")
        generation_log.append(f"Attempt {attempt}/{max_retries}")

        try:
            # Build prompt with schema specification
            prompt = GENERATION_PROMPT_WITH_SCHEMA.format(description=description)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }]

            # Generate
            chat_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            proc_inputs = processor(
                text=[chat_prompt],
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=proc_inputs["input_ids"],
                    attention_mask=proc_inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    **generation_kwargs
                )

            # Extract JSON
            out_text = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)

            # Try to extract JSON block
            start = out_text.find('{')
            end = out_text.rfind('}')
            if start == -1 or end == -1 or end <= start:
                json_str = out_text.strip()
            else:
                json_str = out_text[start:end+1].strip()

            # Validate
            is_valid, error_msg, data = validate_generated_json(json_str)

            if is_valid:
                generation_log.append(f"✓ Generation SUCCESS at attempt {attempt}")
                return True, json_str, data, generation_log
            else:
                generation_log.append(f"✗ Validation failed: {error_msg}")
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed validation: {error_msg}")
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts: {error_msg}")
                    return False, None, None, generation_log

        except Exception as e:
            error_msg = f"Generation error: {str(e)[:200]}"
            generation_log.append(f"✗ {error_msg}")
            logger.error(error_msg)
            if attempt == max_retries:
                return False, None, None, generation_log

    return False, None, None, generation_log


def save_generation_result(
    output_dir: str,
    base_name: str,
    description: str,
    json_str: str,
    generation_log: list,
    success: bool,
) -> str:
    """Save generated JSON and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w') as f:
        f.write(json_str)

    # Save description
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, 'w') as f:
        f.write(description)

    # Save generation log
    log_path = os.path.join(output_dir, f"{base_name}_gen_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Success: {success}\n")
        f.write(f"Generation Log:\n")
        for line in generation_log:
            f.write(f"  {line}\n")

    return json_path


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def main():
    """Main inference function with validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CAD JSON with validation"
    )
    parser.add_argument("--description", type=str, required=True, help="CAD description")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base-name", type=str, required=True, help="Base filename")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-8B", help="Base LLM model")
    parser.add_argument("--max-retries", type=int, default=3, help="Max generation retries")
    parser.add_argument("--max-new-tokens", type=int, default=10240, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    logger.info(f"Loading model: {args.llm_model}")

    # Load tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    # Processor acts as tokenizer for this model
    processor = type('Processor', (), {
        'tokenizer': tokenizer,
        'apply_chat_template': lambda msgs, **kwargs: tokenizer.apply_chat_template(msgs, **kwargs),
    })()

    logger.info("Model loaded successfully")

    # Generate with validation
    success, json_str, data, gen_log = generate_with_retry(
        model,
        processor,
        description=args.description,
        max_retries=args.max_retries,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if success:
        logger.info("✓ Generation successful and validated")
        save_generation_result(
            args.output_dir,
            args.base_name,
            args.description,
            json_str,
            gen_log,
            success=True,
        )
        logger.info(f"Saved to {args.output_dir}/{args.base_name}.json")
        return 0
    else:
        logger.error("✗ Generation failed after retries")
        save_generation_result(
            args.output_dir,
            args.base_name,
            args.description,
            "{}",
            gen_log,
            success=False,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
