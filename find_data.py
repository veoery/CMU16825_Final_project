"""Quick script to find actual data paths in Colab."""

import os
from pathlib import Path

print("=" * 80)
print("SEARCHING FOR DATA FILES")
print("=" * 80)

# Check common data locations
data_roots = [
    "/content/data",
    "/content/Omni-CAD-subset",
    "/content/data/Omni-CAD-subset",
]

for root in data_roots:
    if os.path.exists(root):
        print(f"\n✅ Found: {root}")
        print(f"   Contents:")
        try:
            for item in os.listdir(root)[:20]:  # Show first 20 items
                item_path = os.path.join(root, item)
                if os.path.isdir(item_path):
                    print(f"     📁 {item}/")
                else:
                    print(f"     📄 {item}")
        except Exception as e:
            print(f"     Error: {e}")
    else:
        print(f"\n❌ Not found: {root}")

# Try to find json_truncated directory
print("\n" + "=" * 80)
print("SEARCHING FOR json_truncated DIRECTORY")
print("=" * 80)

import subprocess
result = subprocess.run(
    ["find", "/content", "-type", "d", "-name", "json_truncated", "-maxdepth", "5"],
    capture_output=True,
    text=True,
    timeout=30,
)

if result.stdout.strip():
    print("Found json_truncated at:")
    for path in result.stdout.strip().split('\n'):
        print(f"  ✅ {path}")

        # Show some files in this directory
        try:
            files = list(Path(path).rglob("*.json"))[:5]
            print(f"     Sample files:")
            for f in files:
                print(f"       - {f}")
        except:
            pass
else:
    print("❌ json_truncated directory not found")

# Also search for any .json files with "tr_" pattern
print("\n" + "=" * 80)
print("SEARCHING FOR TRUNCATED JSON FILES (pattern: *tr_*.json)")
print("=" * 80)

result = subprocess.run(
    ["find", "/content", "-type", "f", "-name", "*tr_*.json", "-maxdepth", "6"],
    capture_output=True,
    text=True,
    timeout=30,
)

if result.stdout.strip():
    files = result.stdout.strip().split('\n')[:10]
    print(f"Found {len(result.stdout.strip().split())} truncated files. First 10:")
    for f in files:
        print(f"  - {f}")
else:
    print("❌ No truncated JSON files found")

print("\n" + "=" * 80)
print("CHECKING GOOGLE DRIVE MOUNT")
print("=" * 80)

gdrive_path = "/content/gdrive/MyDrive"
if os.path.exists(gdrive_path):
    print(f"✅ Google Drive mounted at: {gdrive_path}")

    # Look for data in Google Drive
    data_in_drive = Path(gdrive_path) / "data"
    if data_in_drive.exists():
        print(f"\n✅ Found data directory in Google Drive: {data_in_drive}")
        print("   Contents:")
        for item in os.listdir(data_in_drive)[:20]:
            print(f"     - {item}")
else:
    print("❌ Google Drive not mounted")
