- use model: model_path = "/__modal/volumes/vo-xU2jv4A8E4IUtMxIszxpju/Qwen3-8B-lora-r32-lr2e-05-bs64-ep20"
cell:
```
import torch
import os  

# 1. Define your prompt
prompt_list = [
    # "Generate a CAD model with a cylindrical component featuring two large circular ends connected by a rectangular section. The cylindrical ends have hollow centers, and the rectangular section has a smaller circular hole near one end.",
    "Generate a CAD model with a cylinder with radius is half of it's height",
    "Generate a CAD model with a central cylindrical hole in a cubic base.",
    "Generate a CAD model with a cylinder"
    "Generate a CAD model with a cube"
]

# 2. Prepare Input
for idx in range(len(prompt_list)):
    # 3. Generate with Anti-Repetition Settings
    prompt = prompt_list[idx]
    name = prompt[28:32] + "_10240" # Exclude "Generate a CAD model with "
    print(f"Generating for prompt {idx+1}: '{prompt}'...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # generated_cad_0_10k.txt:
    # with torch.no_grad():
    #     output_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=10240,    
    #         repetition_penalty=1.2,  # <--- KEY FIX: Prevents infinite loops
    #         temperature=0.2,         # Low temp = more logical/structured output
    #         top_p=0.95,
    #         do_sample=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #     )

    # generated_cad_0_4096.txt:
    # with torch.no_grad():
    #   output_ids = model.generate(
    #       **inputs,
    #       max_length=4096,
    #       repetition_penalty=1.2,
    #       temperature=0.7,
    #       top_p=0.9,
    #       do_sample=True,
    #       pad_token_id=tokenizer.pad_token_id,
    #       eos_token_id=tokenizer.eos_token_id,
    #   )

    # generated_cad_0_stopping_criteria.txt:
    # with torch.no_grad():
    #     output_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=10240,     
    #         repetition_penalty=1.2,
    #         temperature=0.2,
    #         top_p=0.95,
    #         do_sample=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         stopping_criteria=stopping_criteria 
    # )

    # with torch.no_grad():
    #     output_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=10240,
    #         repetition_penalty=1.2,
    #         temperature=0.2,
    #         top_p=0.95,
    #         do_sample=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         # ✗ REMOVE THIS: stopping_criteria=stopping_criteria
    #     )

    # with torch.no_grad():
    #     output_ids = model.generate(
    #         **inputs,
    #         max_length=4096,
    #         temperature=0.7,
    #         top_p=0.9,
    #         do_sample=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #   )

    # original
    with torch.no_grad():
      output_ids = model.generate(
          **inputs,
          max_length=10240,
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