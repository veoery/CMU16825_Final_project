## running cell:
```
import torch
import os  

# 1. Define your prompt
prompt_list = [
    "Generate a CAD model with a cylinder with radius is half of it's height",
    "Generate a CAD model with a central cylindrical hole in a cubic base.",
    "Generate a CAD model with a cylinder"
]

# 2. Prepare Input
for idx in range(len(prompt_list)):
    # 3. Generate with Anti-Repetition Settings
    prompt = prompt_list[idx]
    name = prompt[28:32] + "cylinder" # Exclude "Generate a CAD model with "
    print(f"Generating for prompt {idx+1}: '{prompt}'...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # original
    with torch.no_grad():
      output_ids = model.generate(
          **inputs,
          max_length=4096,
          temperature=0.5,
          top_p=0.9,
          do_sample=True,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
    
    # 4. Decode to variable (used in next cell)
    raw_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 5. Preview
    print("\n" + "="*40)
    print("RAW OUTPUT PREVIEW (First 500 chars)")
    print("="*40)
    print(raw_output_text[:500] + "...")
    print("="*40)

    # Step 2: save raw_ouput_text to debug:
    output_dir = "./generated_cad"
    os.makedirs(output_dir, exist_ok=True)
    
    raw_txt_path = os.path.join(output_dir, f"generated_cad_{idx}_{name}.txt")
    
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(raw_output_text)

    print(f"✓ Saved raw output text to: {raw_txt_path}")

    # Step 3: output json
    import sys
    import subprocess
    try:
        import json_repair
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "json_repair"])
        import json_repair
    
    import json
    import os
    
    print("\n" + "="*60)
    print("FINAL MERGE & SAVE STRATEGY")
    print("="*60)
    
    # 1. Repair the JSON
    decoded_object = json_repair.repair_json(raw_output_text, return_objects=True)
    
    # 2. Smart Merge Logic
    final_cad_model = {}
    
    if isinstance(decoded_object, list):
        print(f"⚠ Found a LIST of {len(decoded_object)} objects. Merging them...")
        # Iterate through every fragment and merge them into one master dictionary
        for item in decoded_object:
            if isinstance(item, dict):
                # .update() adds new keys and overwrites existing ones
                # This combines the "entities" block with the "bounding_box" block
                final_cad_model.update(item)
    elif isinstance(decoded_object, dict):
        print("✓ Found a single valid dictionary.")
        final_cad_model = decoded_object
    else:
        print(f"✗ Unexpected type: {type(decoded_object)}")
    
    # 3. Validation and Save
    if final_cad_model and "entities" in final_cad_model:
        entity_count = len(final_cad_model['entities'])
        print(f"✓ Valid CAD Model Constructed!")
        print(f"✓ Found {entity_count} entities.")
    
        # Save
        output_dir = "./generated_cad"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"generated_cad_{idx}_{name}.json")
    
        with open(save_path, 'w') as f:
            json.dump(final_cad_model, f, indent=2)
        
        print(f"✓ Saved to: {save_path}")
    
    elif final_cad_model:
        print("⚠ Saved, but 'entities' key is missing. Check output manually.")
        # Save anyway to inspect
        with open("./generated_cad/debug_fragment.json", 'w') as f:
            json.dump(final_cad_model, f, indent=2)
    else:
        print("✗ Failed to construct any valid object.")
```

python -c "
import json

def check_extrudes(filename, name):
    with open(filename) as f:
        data = json.load(f)

    print(f'\n{name}')
    print('=' * 80)

    extrude_count = 0
    zero_extrude = 0

    for ent_id, ent in data['entities'].items():
        if ent.get('type') == 'ExtrudeFeature':
            extrude_count += 1
            extent_one = ent.get('extent_one', {})
            distance = extent_one.get('distance', {})
            value = distance.get('value', 'MISSING')

            # Check if extrude is complete
            has_all_fields = all(k in ent for k in ['extent_one', 'extent_two',
'operation', 'start_extent', 'extent_type'])
            status = '✓' if has_all_fields else '❌'

            if value == 0.0:
                zero_extrude += 1
                status += ' [ZERO VALUE]'

            print(f'{status} {ent_id}: extent_one.distance.value = {value}')

    print(f'\nTotal extrudes: {extrude_count}')
    print(f'Extrudes with 0.0 value: {zero_extrude}')

check_extrudes('gen_cad_all/v5_cylinder_fixed/generated_cad_0_cylicylinder.json',
'FILE 0 (FAILS)')
check_extrudes('gen_cad_all/v5_cylinder_fixed/generated_cad_1_centcylinder.json',
'FILE 1 (WORKS)')
check_extrudes('gen_cad_all/v5_cylinder_fixed/generated_cad_2_cylicylinder.json',
'FILE 2 (FAILS)')
"