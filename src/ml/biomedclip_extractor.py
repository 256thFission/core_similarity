"""
BioMedCLIP feature extraction wrapper
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional


class BioMedCLIPFeatureExtractor:
    """Wrapper for BioMedCLIP multimodal foundation model"""
    
    def __init__(self):
        """Initialize BioMedCLIP feature extractor"""
        self.model = None
        self.preprocess = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> None:
        """Load the BioMedCLIP model from HuggingFace Hub"""
        if self.model is None:
            print("Loading BioMedCLIP model from HuggingFace Hub...")
            
            try:
                from open_clip import create_model_from_pretrained
                
                # Load BioMedCLIP model and preprocessing from HuggingFace Hub
                # Model: BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
                self.model, self.preprocess = create_model_from_pretrained(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                )
                
                # Move model to device and set to eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                print(f"BioMedCLIP model loaded successfully on {self.device}")
                print(f"Model embedding dimension: {self.get_embedding_dim()}")
                
            except ImportError as e:
                raise RuntimeError(
                    "BioMedCLIP dependencies not available. Install with:\n"
                    "pip install open_clip_torch==2.23.0 transformers==4.35.2"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load BioMedCLIP model: {e}") from e
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image using BioMedCLIP model
        
        Args:
            image: 3-channel RGB image (numpy array)
            
        Returns:
            512-dimensional feature vector (numpy array) for ViT-B/16 architecture
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
        
        # Apply BioMedCLIP preprocessing
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract image features using BioMedCLIP
        with torch.no_grad():
            # BioMedCLIP returns image_features, text_features, logit_scale
            # We only need the image features for our embedding similarity task
            image_features, _, _ = self.model(input_tensor, None)
            
            # Normalize features (standard practice for CLIP models)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy().flatten()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        # BioMedCLIP uses ViT-B/16 architecture which outputs 512-dimensional embeddings
        return 512
    
    def get_model_name(self) -> str:
        """Get model name for identification"""
        return "biomedclip"


# BioMedCLIP Implementation Notes:
# 
# Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# - Vision Transformer Base with patch size 16x16
# - Input size: 224x224 pixels
# - Embedding dimension: 512 (ViT-B/16 standard)
# - Trained on PMC-15M dataset (15M figure-caption pairs from PubMed Central)
# 
# Key advantages for cross-modal histology matching:
# 1. Domain-specific: Trained on biomedical images including histopathology
# 2. Multimodal: Joint vision-language training may improve cross-modal understanding
# 3. State-of-the-art: Establishes new SOTA on multiple biomedical VLP benchmarks
# 
# Dependencies:
# - open_clip_torch==2.23.0
# - transformers==4.35.2
# 
# Usage in pipeline:
# pipeline = TissueMatchingPipeline(model="biomedclip")
# results = pipeline.match_tissues(set_a_dir, set_b_dir)