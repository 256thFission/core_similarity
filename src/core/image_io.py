"""
Pure image I/O utilities for loading and saving various image formats
"""

import cv2
import numpy as np
import tifffile
from pathlib import Path
from typing import Optional, List


class ImageLoader:
    """Handles loading images from various formats"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from various formats
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                return ImageLoader._load_tiff(image_path)
            else:
                return ImageLoader._load_standard(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def _load_tiff(image_path: str) -> Optional[np.ndarray]:
        """Load TIFF image with multi-dimensional handling"""
        img = tifffile.imread(image_path)
        
        # Handle multi-dimensional TIFF
        if len(img.shape) > 2:
            if len(img.shape) == 4: 
                img = img[0, 0]  
            elif len(img.shape) == 3:
                if img.shape[0] < img.shape[2]:  
                    img = img[0]  
                else:  # Likely (Y, X, C)
                    img = img[:, :, 0]  # Take first channel
        
        return img
    
    @staticmethod
    def _load_standard(image_path: str) -> Optional[np.ndarray]:
        """Load standard image formats (PNG, JPG, etc.)"""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return img if img is not None else None


class ImageSaver:
    """Handles saving processed images"""
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """
        Save image to file
        
        Args:
            image: Image array to save
            output_path: Path to save image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure image is in correct format for saving
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            return cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False


class ImagePathUtils:
    """Utilities for working with image file paths"""
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    
    @classmethod
    def get_images_from_directory(cls, directory, filename_filter: Optional[str] = None) -> List[Path]:
        """
        Get all image files from a directory
        
        Args:
            directory: Directory to search (str or Path)
            filename_filter: Optional substring that must be present in the filename
            
        Returns:
            List of image file paths
        """
        if isinstance(directory, str):
            directory = Path(directory)
            
        images = []
        for ext in cls.IMAGE_EXTENSIONS:
            # Add both lowercase and uppercase extensions
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        
        # Apply filename filter if specified
        if filename_filter:
            images = [img for img in images if filename_filter in img.name]
            
        return sorted(images)
    
    @classmethod
    def count_images_in_directory(cls, directory, filename_filter: Optional[str] = None) -> int:
        """
        Count images in directory without loading them
        
        Args:
            directory: Directory to count images in (str or Path)
            filename_filter: Optional substring that must be present in the filename
            
        Returns:
            Number of image files
        """
        if isinstance(directory, str):
            directory = Path(directory)
        return len(cls.get_images_from_directory(directory, filename_filter))