"""
UNI2-h feature extraction wrapper
"""

import torch
import timm
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from typing import Optional


class UNI2hFeatureExtractor:
    """Wrapper for UNI2-h pathology foundation model"""
    
    def __init__(self):
        """Initialize UNI2-h feature extractor"""
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self) -> None:
        """Load the UNI2-h model using timm"""
        if self.model is None:
            print("Loading UNI2-h model...")
            
            # UNI2-h configuration from official documentation
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
            
            # Load model with pretrained weights
            self.model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", 
                pretrained=True, 
                **timm_kwargs
            )
            
            # Create transform
            self.transform = create_transform(
                **resolve_data_config(self.model.pretrained_cfg, model=self.model)
            )
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"UNI2-h model loaded successfully on {self.device}")
            print(f"Model embedding dimension: {self.model.embed_dim}")
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image using UNI2-h model
        
        Args:
            image: 3-channel RGB image (numpy array)
            
        Returns:
            1536-dimensional feature vector (numpy array)
        """
        # Ensure model is loaded
        self.load_model()
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Apply UNI2-h transforms (resize to 224x224, normalize)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features using UNI2-h
        with torch.no_grad():
            features = self.model(input_tensor)  # Shape: [1, 1536]
            
        return features.cpu().numpy().flatten()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.model is not None:
            return self.model.embed_dim
        return 1536  # UNI2-h embedding dimension