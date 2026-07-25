"""
ANLP Dataset Loader
Smart loader that downloads datasets from OneDrive (stable) with HuggingFace fallback.

Usage in notebooks:
    from dataset_loader import load_anlp_dataset
    
    dataset = load_anlp_dataset("opus_books")
    # Returns same format as datasets.load_dataset()
"""

import requests
import tarfile
import hashlib
import json
from pathlib import Path
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# Configuration
CACHE_DIR = Path.home() / ".cache" / "anlp_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Google Cloud Storage base URL - um-anlp-datasets bucket
GCS_BASE_URL = "https://storage.googleapis.com/um-anlp-datasets"

# Dataset registry with OneDrive URLs and checksums
DATASETS_REGISTRY = {
    "opus_books": {
        "filename": "opus_books.tar.gz",
        "sha256": "b0f627ab8d10e930cf9337a5bd6f5a839c54b7b51941e47bb77c402396866115",
        "hf_fallback": {
            "name": "Helsinki-NLP/opus_books",
            "config": "en-nl"
        }
    },
    "ml_spoken_words": {
        "filename": "ml_spoken_words.tar.gz",
        "sha256": None,
        "hf_fallback": {
            "name": "MLCommons/ml_spoken_words",
            "languages": ["nl"]
        }
    },
    "imdb": {
        "filename": "imdb.tar.gz",
        "sha256": "b19ea24ec44e37a28b2d64446d0ed5d7c71202b0e59bad2d121c985860a9b3bb",
        "hf_fallback": {
            "name": "stanfordnlp/imdb"
        }
    },
    "conll2003": {
        "filename": "conll2003.tar.gz",
        "sha256": "8d8606c497bd88cec7b7d3983d9979feb6fabb301708e191f926a7f447c44a21",
        "hf_fallback": {
            "name": "hgissbkh/conll2003-en"
        }
    },
    "universal_dependencies": {
        "filename": "universal_dependencies.tar.gz",
        "sha256": None,
        "hf_fallback": {
            "name": "universal_dependencies",
            "config": "en_ewt"
        }
    },
    "glue_sst2": {
        "filename": "glue_sst2.tar.gz",
        "sha256": "74a4b2fddccdab58a8c22d2fd357bee1922e048f572365362fcc3e21ee441030",
        "hf_fallback": {
            "name": "nyu-mll/glue",
            "config": "sst2"
        }
    },
    "ag_news": {
        "filename": "ag_news.tar.gz",
        "sha256": "f28a9c7a97adf7388245654c65365af12ef7a5bd2649eb6ddeef35e40c0f2025",
        "hf_fallback": {
            "name": "fancyzhx/ag_news"
        }
    },
    "rotten_tomatoes": {
        "filename": "rotten_tomatoes.tar.gz",
        "sha256": "d010743da855783602564ffc8c901bee1624da3a78536a226f36f19d027bea38",
        "hf_fallback": {
            "name": "cornell-movie-review-data/rotten_tomatoes"
        }
    },
    "cnn_dailymail": {
        "filename": "cnn_dailymail.tar.gz",
        "sha256": "8caba5332b2c11b600bea2cb12a50031e56385ea59bbe249aad6929d45bc5e2e",
        "hf_fallback": {
            "name": "abisee/cnn_dailymail",
            "config": "3.0.0"
        }
    }
}


def verify_checksum(file_path, expected_sha256):
    """Verify SHA256 checksum of downloaded file."""
    if expected_sha256 is None:
        print("⚠️  No checksum available, skipping verification")
        return True
    
    print("🔐 Verifying checksum...")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    computed = sha256_hash.hexdigest()
    if computed == expected_sha256:
        print("✅ Checksum verified")
        return True
    else:
        print(f"❌ Checksum mismatch!")
        print(f"   Expected: {expected_sha256}")
        print(f"   Got:      {computed}")
        return False


def download_from_onedrive(dataset_name, config):
    """Download dataset package from Google Cloud Storage."""
    
    url = f"{GCS_BASE_URL}/{config['filename']}"
    output_path = CACHE_DIR / config['filename']
    
    print(f"📥 Downloading from Google Cloud Storage: {config['filename']}")
    print(f"   URL: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=config['filename']
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        # Verify checksum
        if not verify_checksum(output_path, config['sha256']):
            output_path.unlink()
            return None
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Google Cloud Storage download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def extract_dataset(archive_path, dataset_name):
    """Extract .tar.gz archive to cache directory."""
    extract_dir = CACHE_DIR / dataset_name
    
    if extract_dir.exists():
        print(f"✅ Dataset already extracted: {extract_dir}")
        return extract_dir
    
    print(f"📦 Extracting {archive_path.name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(CACHE_DIR)
    
    print(f"✅ Extracted to: {extract_dir}")
    return extract_dir


def load_from_hf_fallback(dataset_name, fallback_config):
    """Fallback: load dataset from HuggingFace."""
    print(f"🔄 Falling back to HuggingFace: {fallback_config['name']}")
    
    kwargs = {
        "trust_remote_code": True
    }
    
    if "config" in fallback_config:
        kwargs["name"] = fallback_config["config"]
    
    if "languages" in fallback_config:
        kwargs["languages"] = fallback_config["languages"]
    
    try:
        dataset = load_dataset(fallback_config["name"], **kwargs)
        print("✅ Loaded from HuggingFace")
        return dataset
    except Exception as e:
        print(f"❌ HuggingFace fallback also failed: {e}")
        raise


def load_anlp_dataset(dataset_name, force_hf=False, force_download=False):
    """
    Load ANLP dataset with smart caching.
    
    Priority:
    1. Check local cache
    2. Download from OneDrive (stable URLs)
    3. Fallback to HuggingFace
    
    Args:
        dataset_name: Name of dataset (e.g., "opus_books", "imdb")
        force_hf: Skip OneDrive, use HuggingFace directly
        force_download: Re-download even if cached
        
    Returns:
        Dataset object (same as datasets.load_dataset)
    """
    
    if dataset_name not in DATASETS_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"Available: {', '.join(DATASETS_REGISTRY.keys())}"
        )
    
    config = DATASETS_REGISTRY[dataset_name]
    cached_dir = CACHE_DIR / dataset_name
    
    # Check if already cached (unless force_download)
    if cached_dir.exists() and not force_download:
        print(f"✅ Using cached dataset: {cached_dir}")
        return load_from_disk(str(cached_dir))
    
    # Use HuggingFace directly if forced
    if force_hf:
        return load_from_hf_fallback(dataset_name, config["hf_fallback"])
    
    # Try OneDrive download
    print(f"🎯 Loading dataset: {dataset_name}")
    archive_path = download_from_onedrive(dataset_name, config)
    
    if archive_path:
        # OneDrive download succeeded, extract
        extract_dir = extract_dataset(archive_path, dataset_name)
        return load_from_disk(str(extract_dir))
    else:
        # OneDrive failed, use HuggingFace fallback
        print("⚠️  OneDrive unavailable, using HuggingFace fallback")
        return load_from_hf_fallback(dataset_name, config["hf_fallback"])


def update_checksums(manifest_path):
    """
    Update checksums from manifest.json generated by download_and_package_datasets.py
    
    Args:
        manifest_path: Path to manifest.json file
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    for package in manifest["packages"]:
        dataset_name = package["dataset"]
        if dataset_name in DATASETS_REGISTRY:
            DATASETS_REGISTRY[dataset_name]["sha256"] = package["sha256"]
            print(f"✅ Updated checksum for {dataset_name}")


# Example usage
if __name__ == "__main__":
    print("ANLP Dataset Loader")
    print("=" * 60)
    print("\nAvailable datasets:")
    for name, config in DATASETS_REGISTRY.items():
        hf = config["hf_fallback"]["name"]
        print(f"  - {name:25} (HF: {hf})")
    
    print("\nUsage in notebooks:")
    print("  from dataset_loader import load_anlp_dataset")
    print("  dataset = load_anlp_dataset('opus_books')")
