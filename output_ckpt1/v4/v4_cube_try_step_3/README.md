- `generated_cad_0_cubeoriginal.json` is successfully export a cuboid, 
```
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=4096,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
```
- `generated_cad_0_cubehigher_temp.json` is export extrude surface

python -c "
import json

# Load the repaired files
with open('gen_cad_all/v5_cylinder_fixed/generated_cad_0_cylicylinder.json') as f:
    data0 = json.load(f)

with open('gen_cad_all/v5_cylinder_fixed/generated_cad_1_centcylinder.json') as f:
    data1 = json.load(f)

with open('gen_cad_all/v5_cylinder_fixed/generated_cad_2_cylicylinder.json') as f:
    data2 = json.load(f)

def analyze_profiles(data, name):
    print(f'\n{name}')
    print('=' * 60)
    for ent_id, ent in data['entities'].items():
        if ent.get('type') == 'Sketch':
            profiles = ent.get('profiles')
            print(f'{ent_id}: profiles type = {type(profiles).__name__}')
            if isinstance(profiles, dict):
                print(f'  - Profile keys: {list(profiles.keys())}')
                for pid, prof in profiles.items():
                    loops = prof.get('loops', [])
                    print(f'    {pid}: {len(loops)} loops')
                    for i, loop in enumerate(loops):
                        curves = loop.get('profile_curves', [])
                        print(f'      loop[{i}]: {len(curves)} curves -
{type(curves).__name__}')
            elif isinstance(profiles, list):
                print(f'  ❌ PROFILES IS A LIST (should be dict)!')
                for i, prof in enumerate(profiles):
                    print(f'    [{i}] = {type(prof).__name__}')

analyze_profiles(data0, 'FILE 0: generated_cad_0_cylicylinder.json')
analyze_profiles(data1, 'FILE 1: generated_cad_1_centcylinder.json (WORKS)')
analyze_profiles(data2, 'FILE 2: generated_cad_2_cylicylinder.json')
"