import requests
import sys
from tqdm import tqdm

def download_file(url, output_path):
    """Download large file with progress bar using streaming."""
    print(f"Downloading {url}...")

    # Stream the download to avoid loading everything into memory
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8 KB chunks

    # Download with progress bar
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  unit_divisor=1024, desc=output_path) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"\nDownload complete: {output_path}")

if __name__ == "__main__":
    url = "https://huggingface.co/datasets/jingwei-xu-00/Omni-CAD/resolve/main/Omni-CAD.zip"
    output = "../data/Omni-CAD.zip"

    download_file(url, output)
