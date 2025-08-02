# Tissue Microarray Core Similarity Matching

A complete implementation of the 4-step tissue microarray image matching pipeline using the state-of-the-art UNI2-h pathology foundation model.

## Overview

This project implements exact one-to-one matching between two sets of Tissue Microarray (TMA) core images:
- **Set A**: Clean grayscale images (no calibration dots)
- **Set B**: H&E stained images with black calibration dots superimposed

The pipeline uses classical computer vision techniques for preprocessing and the UNI2-h Vision Transformer for robust feature extraction.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup UNI2-h Model
```bash
python setup_uni.py
```
This will guide you through:
- Hugging Face authentication 
- UNI2-h model access request
- Model download and validation

### 3. Test Implementation
```bash
python test_step1.py
```

### 4. Run Full Pipeline
```bash
python main.py match-tissues path/to/set_a path/to/set_b
```
python main.py match-tissues  /cwork/pyl10/projects/core_similarity/fullres/tissue_dapi_fullres_processed/cores_filtered /cwork/pyl10/projects/core_similarity/hires/tissue_hires_image_processed/cores_filtered --model biomedclip --flip-set-b
## Implementation Details

### Step 1: Set B Image Standardization & Healing
- **A**: Convert H&E images to grayscale
- **B**: Create damage mask using thresholding (pixel value < 50) to detect calibration dots
- **C**: Apply `cv2.inpaint()` with Navier-Stokes algorithm to heal images

### Step 2: Feature Extraction with UNI2-h
- **A**: Convert grayscale to 3-channel RGB for model compatibility
- **B**: Load UNI2-h (ViT-h/14) foundation model trained on 200M+ pathology images
- **C**: Extract 1536-dimensional CLS token embeddings

### Step 3: Normalize & Match Embeddings
- **A**: Apply L2 normalization to all feature vectors
- **B**: Compute cosine distances between Set A and Set B embeddings
- **C**: Find best matches using minimum cosine distance

### Step 4: Output Generation
- Generate matched pairs in required tuple format
- Save results as both JSON and Python list

## Key Features

- **UNI2-h Integration**: Uses latest pathology foundation model (Jan 2025)
- **Robust Preprocessing**: Handles various image formats (TIFF, PNG, JPG)
- **Memory Efficient**: Processes images individually to handle large datasets
- **Clean Architecture**: Simple, atomic implementation without fallbacks
- **Rich CLI**: Beautiful progress bars and detailed status output

## Architecture

The codebase follows a clean separation of concerns with modular components:

### Core Modules
- `image_io.py`: Pure image loading/saving utilities
- `image_processing.py`: Image preprocessing operations (grayscale, healing, filters)
- `feature_extractor.py`: UNI2-h model wrapper for feature extraction
- `matcher.py`: Embedding normalization and matching logic
- `pipeline.py`: High-level pipeline orchestration

### Interface & Setup
- `main.py`: Pure CLI interface using Typer
- `setup_uni.py`: UNI2-h model setup and authentication
- `test_step1.py`: Comprehensive testing of all components
- `requirements.txt`: All necessary dependencies

## CLI Commands

### Match Tissues
```bash
python main.py match-tissues <set_a_dir> <set_b_dir> [options]
```

Options:
- `--output, -o`: Output file for results (default: matches.json)
- `--verbose, -v`: Enable detailed output

### Info
```bash
python main.py info <directory>
```
Display information about images in a directory.

### Process Directories
```bash
python main.py process-directories <dir1> <dir2> [options]
```
Basic preprocessing without UNI2-h feature extraction.

## Output Format

Results are saved in two formats:

**JSON format (matches.json):**
```json
{
  "matches": [
    {
      "set_a": "image1.jpg",
      "set_b": "image5.jpg", 
      "cosine_distance": 0.1234
    }
  ],
  "summary": {
    "set_a_count": 10,
    "set_b_count": 10,
    "successful_matches": 10,
    "average_distance": 0.1234
  }
}
```

**Python tuple format (matches.py):**
```python
matched_pairs = [
    ('image1.jpg', 'image5.jpg'),
    ('image2.jpg', 'image3.jpg'),
    # ...
]
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Hugging Face account with UNI2-h access
- ~10GB disk space for model weights

## License

This implementation follows the UNI2-h model license (CC-BY-NC-ND 4.0) for non-commercial, academic research purposes only.