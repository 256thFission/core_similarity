"""
Image preprocessing operations for tissue core processing
"""

import cv2
import numpy as np
from typing import Optional


class ImageProcessor:
    """Handles image preprocessing operations"""
    
    def __init__(self, downsample_factor: float = 1.0, median_filter_size: int = 1):
        """
        Initialize image processor
        
        Args:
            downsample_factor: Factor to downsample images by
            median_filter_size: Size of median filter kernel (1 = no filtering)
        """
        self.downsample_factor = downsample_factor
        self.median_filter_size = median_filter_size
    



class TissueProcessor:
    """High-level tissue processing pipelines"""
    
    def __init__(self, processor: ImageProcessor):
        """
        Initialize tissue processor
        
        Args:
            processor: ImageProcessor instance
        """
        self.processor = processor
    
    def process_set_a_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process Set A image (clean grayscale)
        
        Args:
            image: Raw image from Set A
            
        Returns:
            Processed 3-channel RGB image ready for feature extraction
        """
        # Convert to 3-channel RGB (convert_to_grayscale_rgb)
        # Handle different input formats
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            gray_image = image
            
        # Convert to 3-channel RGB 
        processed = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        return processed
    
    def process_set_b_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process Set B image (H&E with calibration dots)
        
        Args:
            image: Raw image from Set B
            
        Returns:
            Processed 3-channel RGB image ready for feature extraction
        """
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create damage mask for calibration dots (create_damage_mask)
        # Create mask for pixels below threshold (dark dots)
        mask = (gray < 50).astype(np.uint8) * 255
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        damage_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Heal the image using inpainting (heal_image)
        healed_gray = cv2.inpaint(gray, damage_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        
        # Convert to 3-channel RGB
        healed_3ch = cv2.cvtColor(healed_gray, cv2.COLOR_GRAY2RGB)
        
        return healed_3ch