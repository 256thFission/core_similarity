"""
Visualization tools for tissue matching results.
Provides functions to visualize matched tissue pairs side by side.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


class MatchVisualizer:
    """Visualizer for tissue matching results."""
    
    def __init__(self, project_root: str, set_a_dir: Optional[str] = None, set_b_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            project_root: Root directory of the project
            set_a_dir: Directory containing Set A images (grayscale). If None, auto-detects.
            set_b_dir: Directory containing Set B images (H&E). If None, auto-detects.
        """
        self.project_root = project_root
        
        # Auto-detect directories if not provided
        if set_a_dir is None:
            set_a_dir = self._find_image_directory(project_root, ["fullres", "greyscale"], "Set A")
        if set_b_dir is None:
            set_b_dir = self._find_image_directory(project_root, ["hires", "outputs"], "Set B")
            
        self.set_a_dir = set_a_dir
        self.set_b_dir = set_b_dir
        
        # Validate directories exist
        if not os.path.exists(self.set_a_dir):
            raise ValueError(f"Set A directory not found: {self.set_a_dir}")
        if not os.path.exists(self.set_b_dir):
            raise ValueError(f"Set B directory not found: {self.set_b_dir}")
        
        print(f"Using Set A directory: {self.set_a_dir}")
        print(f"Using Set B directory: {self.set_b_dir}")
    
    def _find_image_directory(self, project_root: str, possible_root_dirs: List[str], set_name: str) -> str:
        """
        Auto-detect image directory by searching for common patterns.
        
        Args:
            project_root: Root directory of the project
            possible_root_dirs: List of possible root directory names to search
            set_name: Name of the set for error messages
            
        Returns:
            Path to the detected image directory
            
        Raises:
            ValueError: If no suitable directory is found
        """
        # Common patterns for image directories
        common_patterns = [
            "cores_filtered",
            "individual_cores", 
            "cores",
            "processed"
        ]
        
        for root_dir in possible_root_dirs:
            root_path = os.path.join(project_root, root_dir)
            if os.path.exists(root_path):
                # Search recursively for directories containing image files
                for root, dirs, files in os.walk(root_path):
                    # Check if this directory contains PNG files
                    png_files = [f for f in files if f.lower().endswith('.png')]
                    if png_files:
                        print(f"Found {len(png_files)} PNG files in: {root}")
                        return root
                        
                    # Also check for common pattern directories
                    for pattern in common_patterns:
                        if pattern in dirs:
                            pattern_path = os.path.join(root, pattern)
                            # Check if the pattern directory contains images
                            try:
                                pattern_files = os.listdir(pattern_path)
                                pattern_pngs = [f for f in pattern_files if f.lower().endswith('.png')]
                                if pattern_pngs:
                                    print(f"Found {len(pattern_pngs)} PNG files in: {pattern_path}")
                                    return pattern_path
                            except OSError:
                                continue
        
        raise ValueError(f"Could not auto-detect {set_name} image directory. "
                        f"Please provide explicit paths. Searched in: {possible_root_dirs}")
    
    def load_matches(self, matches_file: str) -> List[Dict]:
        """
        Load matches from JSON file.
        
        Args:
            matches_file: Path to matches.json file
            
        Returns:
            List of match dictionaries
        """
        with open(matches_file, 'r') as f:
            data = json.load(f)
        return data['matches']
    
    def get_best_matches(self, matches: List[Dict], n: int = 20) -> List[Dict]:
        """
        Get the best n matches sorted by cosine distance (lower is better).
        
        Args:
            matches: List of match dictionaries
            n: Number of best matches to return
            
        Returns:
            List of best matches
        """
        return sorted(matches, key=lambda x: x['cosine_distance'])[:n]
    
    def get_best_match_per_set_b(self, matches: List[Dict]) -> List[Dict]:
        """
        Get the best match for each unique core in Set B.
        This shows the best matching Set A core for each Set B core.
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            List of best matches, one per Set B core
        """
        # Group matches by Set B core
        set_b_groups = {}
        for match in matches:
            set_b_core = match['set_b']
            if set_b_core not in set_b_groups:
                set_b_groups[set_b_core] = []
            set_b_groups[set_b_core].append(match)
        
        # Find best match for each Set B core
        best_matches = []
        for set_b_core, core_matches in set_b_groups.items():
            best_match = min(core_matches, key=lambda x: x['cosine_distance'])
            best_matches.append(best_match)
        
        # Sort by Set B core name for consistent ordering
        return sorted(best_matches, key=lambda x: x['set_b'])
    
    def load_image_safely(self, image_path: str, thumbnail_size: Tuple[int, int] = (150, 150)) -> Optional[np.ndarray]:
        """
        Safely load and resize an image.
        
        Args:
            image_path: Path to the image
            thumbnail_size: Target thumbnail size (width, height)
            
        Returns:
            Image array or None if loading fails
        """
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                return None
            
            img = Image.open(image_path)
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            return np.array(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_placeholder(self, size: Tuple[int, int] = (150, 150), text: str = "Not Found") -> np.ndarray:
        """
        Create a placeholder image when the actual image cannot be loaded.
        
        Args:
            size: Size of the placeholder (width, height)
            text: Text to display on placeholder
            
        Returns:
            Placeholder image array
        """
        placeholder = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200
        return placeholder
    
    def visualize_matches(self, matches_file: str, n_matches: int = 20, 
                         thumbnail_size: Tuple[int, int] = (150, 150),
                         cols: int = 4, save_path: Optional[str] = None) -> None:
        """
        Visualize best matches side by side in a grid layout.
        
        Args:
            matches_file: Path to matches.json file
            n_matches: Number of best matches to visualize
            thumbnail_size: Size of thumbnails (width, height)
            cols: Number of columns in the grid (each match takes 2 columns)
            save_path: Optional path to save the visualization
        """
        # Load and get best matches
        matches = self.load_matches(matches_file)
        best_matches = self.get_best_matches(matches, n_matches)
        
        # Calculate grid dimensions
        rows = (n_matches + cols - 1) // cols
        fig_width = cols * 2 * (thumbnail_size[0] / 100) + 2  # 2 images per match, convert px to inches
        fig_height = rows * (thumbnail_size[1] / 100) + 2
        
        fig, axes = plt.subplots(rows, cols * 2, figsize=(fig_width, fig_height))
        fig.suptitle(f'Best {n_matches} Tissue Matches (Sorted by Cosine Distance)', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, match in enumerate(best_matches):
            if i >= n_matches:
                break
                
            row = i // cols
            col_start = (i % cols) * 2
            
            # Load images
            set_a_path = os.path.join(self.set_a_dir, match['set_a'])
            set_b_path = os.path.join(self.set_b_dir, match['set_b'])
            
            img_a = self.load_image_safely(set_a_path, thumbnail_size)
            img_b = self.load_image_safely(set_b_path, thumbnail_size)
            
            # Use placeholders if images can't be loaded
            if img_a is None:
                img_a = self.create_placeholder(thumbnail_size, "Set A\nNot Found")
            if img_b is None:
                img_b = self.create_placeholder(thumbnail_size, "Set B\nNot Found")
            
            # Plot Set A image
            ax_a = axes[row * cols * 2 + col_start]
            ax_a.imshow(img_a, cmap='gray' if len(img_a.shape) == 2 else None)
            ax_a.set_title(f'Set A\n{match["set_a"]}', fontsize=8, fontweight='bold')
            ax_a.axis('off')
            
            # Add colored border for Set A
            rect_a = patches.Rectangle((0, 0), img_a.shape[1]-1, img_a.shape[0]-1, 
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax_a.add_patch(rect_a)
            
            # Plot Set B image
            ax_b = axes[row * cols * 2 + col_start + 1]
            ax_b.imshow(img_b, cmap='gray' if len(img_b.shape) == 2 else None)
            ax_b.set_title(f'Set B\n{match["set_b"]}', fontsize=8, fontweight='bold')
            ax_b.axis('off')
            
            # Add colored border for Set B
            rect_b = patches.Rectangle((0, 0), img_b.shape[1]-1, img_b.shape[0]-1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax_b.add_patch(rect_b)
            
            # Add distance information
            distance_text = f'Distance: {match["cosine_distance"]:.4f}'
            fig.text(0.1 + (col_start / (cols * 2)) * 0.8 + 0.1 / (cols * 2), 
                    0.95 - (row / rows) * 0.85 - 0.12, distance_text, 
                    fontsize=9, ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
        
        # Hide unused subplots
        total_subplots = rows * cols * 2
        used_subplots = min(n_matches * 2, total_subplots)
        for i in range(used_subplots, total_subplots):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def visualize_best_matches_per_set_b(self, matches_file: str, 
                                        thumbnail_size: Tuple[int, int] = (200, 200),
                                        cols: int = 3, save_path: Optional[str] = None) -> None:
        """
        Visualize the best match for each core in Set B.
        Shows which Set A core best matches each Set B core.
        
        Args:
            matches_file: Path to matches.json file
            thumbnail_size: Size of thumbnails (width, height)
            cols: Number of columns in the grid (each match takes 2 columns)
            save_path: Optional path to save the visualization
        """
        # Load matches and get best match per Set B core
        matches = self.load_matches(matches_file)
        best_matches_per_b = self.get_best_match_per_set_b(matches)
        
        n_matches = len(best_matches_per_b)
        print(f"Found {n_matches} unique cores in Set B")
        
        # Calculate grid dimensions
        rows = (n_matches + cols - 1) // cols
        fig_width = cols * 2 * (thumbnail_size[0] / 100) + 2  # 2 images per match
        fig_height = rows * (thumbnail_size[1] / 100) + 3
        
        fig, axes = plt.subplots(rows, cols * 2, figsize=(fig_width, fig_height))
        fig.suptitle(f'Best Match for Each Set B Core ({n_matches} cores)', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, match in enumerate(best_matches_per_b):
            if i >= n_matches:
                break
                
            row = i // cols
            col_start = (i % cols) * 2
            
            # Load images
            set_a_path = os.path.join(self.set_a_dir, match['set_a'])
            set_b_path = os.path.join(self.set_b_dir, match['set_b'])
            
            img_a = self.load_image_safely(set_a_path, thumbnail_size)
            img_b = self.load_image_safely(set_b_path, thumbnail_size)
            
            # Use placeholders if images can't be loaded
            if img_a is None:
                img_a = self.create_placeholder(thumbnail_size, "Set A\nNot Found")
            if img_b is None:
                img_b = self.create_placeholder(thumbnail_size, "Set B\nNot Found")
            
            # Plot Set B image (target)
            ax_b = axes[row * cols * 2 + col_start]
            ax_b.imshow(img_b, cmap='gray' if len(img_b.shape) == 2 else None)
            ax_b.set_title(f'Set B (Target)\n{match["set_b"]}', fontsize=9, fontweight='bold', color='red')
            ax_b.axis('off')
            
            # Add colored border for Set B
            rect_b = patches.Rectangle((0, 0), img_b.shape[1]-1, img_b.shape[0]-1, 
                                     linewidth=3, edgecolor='red', facecolor='none')
            ax_b.add_patch(rect_b)
            
            # Plot Set A image (best match)
            ax_a = axes[row * cols * 2 + col_start + 1]
            ax_a.imshow(img_a, cmap='gray' if len(img_a.shape) == 2 else None)
            ax_a.set_title(f'Set A (Best Match)\n{match["set_a"]}', fontsize=9, fontweight='bold', color='blue')
            ax_a.axis('off')
            
            # Add colored border for Set A
            rect_a = patches.Rectangle((0, 0), img_a.shape[1]-1, img_a.shape[0]-1, 
                                     linewidth=3, edgecolor='blue', facecolor='none')
            ax_a.add_patch(rect_a)
            
            # Add distance information
            distance_text = f'Distance: {match["cosine_distance"]:.4f}'
            fig.text(0.1 + (col_start / (cols * 2)) * 0.8 + 0.1 / (cols * 2), 
                    0.92 - (row / rows) * 0.8 - 0.08, distance_text, 
                    fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        
        # Hide unused subplots
        total_subplots = rows * cols * 2
        used_subplots = min(n_matches * 2, total_subplots)
        for i in range(used_subplots, total_subplots):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def create_detailed_comparison(self, matches_file: str, match_index: int = 0,
                                  save_path: Optional[str] = None) -> None:
        """
        Create a detailed side-by-side comparison of a specific match.
        
        Args:
            matches_file: Path to matches.json file
            match_index: Index of the match to visualize (0 = best match)
            save_path: Optional path to save the visualization
        """
        matches = self.load_matches(matches_file)
        best_matches = self.get_best_matches(matches, match_index + 1)
        
        if match_index >= len(best_matches):
            print(f"Match index {match_index} out of range. Only {len(best_matches)} matches available.")
            return
        
        match = best_matches[match_index]
        
        # Load images at higher resolution
        set_a_path = os.path.join(self.set_a_dir, match['set_a'])
        set_b_path = os.path.join(self.set_b_dir, match['set_b'])
        
        img_a = self.load_image_safely(set_a_path, (300, 300))
        img_b = self.load_image_safely(set_b_path, (300, 300))
        
        if img_a is None:
            img_a = self.create_placeholder((300, 300), "Set A\nNot Found")
        if img_b is None:
            img_b = self.create_placeholder((300, 300), "Set B\nNot Found")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot Set A
        ax1.imshow(img_a, cmap='gray' if len(img_a.shape) == 2 else None)
        ax1.set_title(f'Set A (Grayscale)\n{match["set_a"]}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax1.add_patch(patches.Rectangle((0, 0), img_a.shape[1]-1, img_a.shape[0]-1, 
                                       linewidth=3, edgecolor='blue', facecolor='none'))
        
        # Plot Set B
        ax2.imshow(img_b, cmap='gray' if len(img_b.shape) == 2 else None)
        ax2.set_title(f'Set B (H&E)\n{match["set_b"]}', fontsize=12, fontweight='bold')
        ax2.axis('off')
        ax2.add_patch(patches.Rectangle((0, 0), img_b.shape[1]-1, img_b.shape[0]-1, 
                                       linewidth=3, edgecolor='red', facecolor='none'))
        
        # Add match information
        match_info = f'Cosine Distance: {match["cosine_distance"]:.6f}\nRank: #{match_index + 1}'
        fig.suptitle(f'Detailed Match Comparison\n{match_info}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Detailed comparison saved to: {save_path}")
        
        plt.show()


def visualize_best_matches(project_root: str, matches_file: str, 
                          n_matches: int = 20, cols: int = 4,
                          save_path: Optional[str] = None) -> None:
    """
    Convenience function to visualize best matches.
    
    Args:
        project_root: Root directory of the project
        matches_file: Path to matches.json file
        n_matches: Number of best matches to show
        cols: Number of columns in the grid
        save_path: Optional path to save the visualization
    """
    visualizer = MatchVisualizer(project_root)
    visualizer.visualize_matches(matches_file, n_matches, cols=cols, save_path=save_path)


if __name__ == "__main__":
    # Example usage
    project_root = "/cwork/pyl10/projects/core_similarity"
    matches_file = os.path.join(project_root, "matches.json")
    
    # Create visualizer
    visualizer = MatchVisualizer(project_root)
    
    # Show best 20 matches in a grid
    visualizer.visualize_matches(matches_file, n_matches=20, cols=4)
    
    # Show detailed view of the best match
    visualizer.create_detailed_comparison(matches_file, match_index=0)
