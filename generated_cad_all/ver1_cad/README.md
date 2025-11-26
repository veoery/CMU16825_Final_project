1. Initial error: The file had Python dict syntax (single quotes: {'entities': ...})
instead of valid JSON (double quotes)
2. I fixed it by running this command:
python3 << 'EOF'
import json
import ast

# Read the malformed JSON file
with open('.../cad_1.json', 'r') as f:
    content = f.read()

# Parse it as Python dict
data = ast.literal_eval(content)

# Save as proper JSON with indent=2
with open('.../cad_1.json', 'w') as f:
    json.dump(data, f, indent=2)  # ← This is what added the pretty printing
EOF

The indent=2 parameter in json.dump() is what formatted it with:
- 2-space indentation
- Line breaks between key-value pairs
- Proper JSON double-quote syntax

So it was pretty-printed at that moment when we converted it from invalid Python dict
syntax to valid, formatted JSON.