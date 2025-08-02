#!/usr/bin/env python3
"""
Script to visualize the best match for each core in Set B
Shows which Set A core best matches each Set B core
"""

import os
import sys
from PIL import Image
from src.visualization.match_visualizer import MatchVisualizer

# Suppress PIL decompression bomb warnings for large medical images
Image.MAX_IMAGE_PIXELS = None

def main():
    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    matches_file = os.path.join(project_root, "matches.json")
    
    if not os.path.exists(matches_file):
        print(f"Error: matches.json not found at {matches_file}")
        return
    
    # Create visualizer with explicit directories
    set_a_dir = "/cwork/pyl10/projects/core_similarity/fullres/tissue_dapi_fullres_processed/cores_filtered"
    set_b_dir = "/cwork/pyl10/projects/core_similarity/hires/tissue_hires_image_processed/cores_filtered"
    visualizer = MatchVisualizer(project_root, set_a_dir=set_a_dir, set_b_dir=set_b_dir)
    
    print("Creating visualization showing the best match for each Set B core...")
    print("Set B cores: H&E images with calibration dots (smaller set)")
    print("Set A cores: Grayscale images (larger set)")
    print()
    
    try:
        # Show best match for each Set B core
        visualizer.visualize_best_matches_per_set_b(
            matches_file, 
            thumbnail_size=(200, 200),
            cols=3,  # 3 columns, so 3 matches per row
            save_path=os.path.join(project_root, "best_matches_per_set_b.png")
        )
        
        print("\nVisualization complete!")
        print("File saved: best_matches_per_set_b.png")
        print("\nThis shows:")
        print("- Left image (red border): Set B core (target)")
        print("- Right image (blue border): Best matching Set A core")
        print("- Distance score: Lower is better")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
