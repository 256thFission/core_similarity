"""
Machine learning components: UNI2-h feature extraction
"""

# Conditional import - only available if dependencies are installed
try:
    from .feature_extractor import UNI2hFeatureExtractor
    __all__ = ["UNI2hFeatureExtractor"]
    ML_AVAILABLE = True
except ImportError:
    __all__ = []
    ML_AVAILABLE = False