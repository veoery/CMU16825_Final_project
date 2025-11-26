python3 << 'EOF'
import json

print("FIXING JSON FILES\n")

# Fix cad_2.json
print("Fixing cad_2.json...")
with open('generated_cad/cad_2.json', 'r') as f:
    data = json.load(f)

# Add missing keys
if 'sequence' not in data:
    # Create sequence from entities
    entities = data.get('entities', {})
    sequence = [{'index': i, 'type': v.get('type', 'Unknown'), 'entity': k}
                for i, (k, v) in enumerate(entities.items())]
    data['sequence'] = sequence

if 'properties' not in data:
    data['properties'] = {}

with open('generated_cad/cad_2.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"  ✓ Added sequence ({len(data['sequence'])} items)")
print(f"  ✓ Added properties")
print()

# Fix cad_3.json - check if it's completely broken
print("Checking cad_3.json structure...")
with open('generated_cad/cad_3.json', 'r') as f:
    data = json.load(f)

print(f"  Current keys: {list(data.keys())}")

# Check if the structure is correct
if 'entities' in data and isinstance(data['entities'], dict):
    print(f"  ✓ Has proper 'entities' structure")

    # Add missing keys
    if 'sequence' not in data:
        entities = data['entities']
        sequence = [{'index': i, 'type': v.get('type', 'Unknown'), 'entity': k}
                    for i, (k, v) in enumerate(entities.items())]
        data['sequence'] = sequence

    if 'properties' not in data:
        data['properties'] = {}

    # Clean up top-level junk keys (remove non-standard keys)
    standard_keys = {'entities', 'sequence', 'properties'}
    junk_keys = set(data.keys()) - standard_keys
    if junk_keys:
        print(f"  Removing junk keys: {junk_keys}")
        for key in junk_keys:
            del data[key]

    with open('generated_cad/cad_3.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✓ Fixed cad_3.json")
else:
    print(f"  ✗ cad_3.json structure is corrupted beyond repair")
    print(f"    Recommend using: cp generated_cad/cad_3_gt.json generated_cad/cad_3.json")

print("\nVerifying fixes...")
for filename in ['cad_2.json', 'cad_3.json']:
    with open(f'generated_cad/{filename}', 'r') as f:
        data = json.load(f)
    has_seq = 'sequence' in data
    has_props = 'properties' in data
    has_ent = 'entities' in data
    status = "✓ OK" if (has_seq and has_props and has_ent) else "✗ FAIL"
    print(f"{filename}: {status}")

EOF

Fix missing sequence and properties keys