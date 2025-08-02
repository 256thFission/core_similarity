"""
High-level pipeline orchestration for tissue matching
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from .image_io import ImageLoader, ImagePathUtils
from ..processing.image_processing import ImageProcessor, TissueProcessor
from ..matching.matcher import EmbeddingMatcher, MatchResults

# Conditional import for UNI2-h (requires timm)
try:
    from ..ml.feature_extractor import UNI2hFeatureExtractor
    UNI_AVAILABLE = True
except ImportError:
    UNI_AVAILABLE = False


class TissueMatchingPipeline:
    
    def __init__(self, model: str = "uni2h", downsample_factor: float = 1.0, median_filter_size: int = 1):
        """
        Initialize the tissue matching pipeline
        
        Args:
            model: Feature extraction model to use ("uni2h", "biomedclip", "conch", "virchow2", "kimianet")
            downsample_factor: Factor to downsample images by
            median_filter_size: Size of median filter kernel
        """
        self.image_loader = ImageLoader()
        self.image_processor = ImageProcessor(downsample_factor, median_filter_size)
        self.tissue_processor = TissueProcessor(self.image_processor)
        self.matcher = EmbeddingMatcher()
        
        # Initialize feature extractor based on model selection
        self.feature_extractor = self._create_feature_extractor(model)
    
    def _create_feature_extractor(self, model: str):
        """
        Create feature extractor based on model name
        
        Args:
            model: Model name to create extractor for
            
        Returns:
            Feature extractor instance
            
        Raises:
            ValueError: If model name is not recognized
            RuntimeError: If model dependencies are not available
        """
        if model == "uni2h":
            if UNI_AVAILABLE:
                return UNI2hFeatureExtractor()
            else:
                raise RuntimeError("UNI2-h model not available. Install timm and torch: pip install timm torch")
        elif model == "biomedclip":
            try:
                from ..ml.biomedclip_extractor import BioMedCLIPFeatureExtractor
                return BioMedCLIPFeatureExtractor()
            except ImportError:
                raise RuntimeError("BioMedCLIP model not available. Implementation pending.")
        elif model == "conch":
            try:
                from ..ml.conch_extractor import CONCHFeatureExtractor
                return CONCHFeatureExtractor()
            except ImportError:
                raise RuntimeError("CONCH model not available. Implementation pending.")
        elif model == "virchow2":
            try:
                from ..ml.virchow2_extractor import Virchow2FeatureExtractor
                return Virchow2FeatureExtractor()
            except ImportError:
                raise RuntimeError("Virchow2 model not available. Implementation pending.")
        elif model == "kimianet":
            try:
                from ..ml.kimianet_extractor import KimiaNetFeatureExtractor
                return KimiaNetFeatureExtractor()
            except ImportError:
                raise RuntimeError("KimiaNet model not available. Implementation pending.")
        else:
            available_models = ["uni2h", "biomedclip", "conch", "virchow2", "kimianet"]
            raise ValueError(f"Unknown model '{model}'. Available models: {available_models}")
    
    def process_directory(
        self, 
        directory: Path, 
        is_set_b: bool = False,
        progress_callback: Optional[callable] = None,
        filename_filter: Optional[str] = None,
        flip_horizontal: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process all images in a directory
        
        Args:
            directory: Directory containing images
            is_set_b: True if this is Set B (H&E with calibration dots)
            progress_callback: Optional callback for progress updates
            filename_filter: Optional substring that must be present in the filename
            flip_horizontal: Whether to flip images horizontally before processing
            
        Returns:
            Tuple of (embeddings_array, filenames_list)
        """
        # Get all image files
        image_paths = ImagePathUtils.get_images_from_directory(directory, filename_filter)
        
        if not image_paths:
            raise ValueError(f"No images found in directory: {directory}")
        
        embeddings = []
        filenames = []
        
        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, len(image_paths), image_path.name)
            
            try:
                # Load image
                image = self.image_loader.load_image(str(image_path))
                if image is None:
                    continue
                
                # Apply horizontal flip if requested
                if flip_horizontal:
                    image = np.fliplr(image)
                
                # Process image based on set type
                if is_set_b:
                    processed_image = self.tissue_processor.process_set_b_image(image)
                else:
                    processed_image = self.tissue_processor.process_set_a_image(image)
                
                if processed_image is None:
                    continue
                
                # Extract features if extractor is available
                if self.feature_extractor is not None:
                    features = self.feature_extractor.extract_features(processed_image)
                    if features is not None:
                        embeddings.append(features)
                        filenames.append(image_path.name)
                else:
                    raise RuntimeError("Feature extractor not available. Check model initialization.")
                    
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        if not embeddings:
            raise ValueError(f"No valid embeddings extracted from {directory}")
        
        return np.array(embeddings), filenames
    
    def match_tissues(
        self,
        set_a_dir: Path,
        set_b_dir: Path,
        output_file: str = "matches.json",
        progress_callback: Optional[callable] = None,
        filename_filter: Optional[str] = None,
        flip_set_b: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Complete tissue matching pipeline
        
        Args:
            set_a_dir: Directory containing Set A images (clean grayscale)
            set_b_dir: Directory containing Set B images (H&E with dots)
            output_file: Output file for results
            progress_callback: Optional callback for progress updates
            filename_filter: Optional substring that must be present in the filename
            flip_set_b: Whether to flip Set B images horizontally before processing
            
        Returns:
            Tuple of (matched_pairs, summary_stats)
        """
        # Process Set A images
        if progress_callback:
            progress_callback("Processing Set A images...")
        
        embeddings_a, filenames_a = self.process_directory(
            set_a_dir, 
            is_set_b=False,
            progress_callback=lambda i, total, name: progress_callback(f"Set A: {name} ({i+1}/{total})"),
            filename_filter=filename_filter
        )
        
        # Process Set B images  
        if progress_callback:
            progress_callback("Processing Set B images...")
            
        embeddings_b, filenames_b = self.process_directory(
            set_b_dir,
            is_set_b=True,
            progress_callback=lambda i, total, name: progress_callback(f"Set B: {name} ({i+1}/{total})"),
            filename_filter=filename_filter,
            flip_horizontal=flip_set_b
        )
        
        # Match embeddings
        if progress_callback:
            progress_callback("Computing matches...")
            
        matched_pairs, summary = self.matcher.match_embeddings(
            embeddings_a, embeddings_b, filenames_a, filenames_b
        )
        
        # Save results
        if progress_callback:
            progress_callback("Saving results...")
            
        MatchResults.save_results(matched_pairs, summary, output_file)
        
        return matched_pairs, summary


class BasicImageProcessor:
    """Basic image processing pipeline without UNI2-h features"""
    
    def __init__(self, downsample_factor: float = 1.0, median_filter_size: int = 1):
        """
        Initialize basic processor
        
        Args:
            downsample_factor: Factor to downsample images by
            median_filter_size: Size of median filter kernel
        """
        self.image_loader = ImageLoader()
        self.image_processor = ImageProcessor(downsample_factor, median_filter_size)
        self.tissue_processor = TissueProcessor(self.image_processor)
    
    def process_images_from_directories(
        self,
        directories: List[Path],
        output_dir: Optional[Path] = None,
        normalize: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, int]:
        """
        Process images from multiple directories with basic preprocessing
        
        Args:
            directories: List of directories to process
            output_dir: Optional output directory for processed images
            normalize: Whether to normalize image intensities
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        from .image_io import ImageSaver
        
        stats = {'processed': 0, 'failed': 0, 'total': 0}
        
        # Count total images
        for directory in directories:
            stats['total'] += ImagePathUtils.count_images_in_directory(directory)
        
        processed_count = 0
        
        for directory in directories:
            image_paths = ImagePathUtils.get_images_from_directory(directory)
            
            for image_path in image_paths:
                processed_count += 1
                
                if progress_callback:
                    progress_callback(processed_count, stats['total'], image_path.name)
                
                try:
                    # Load image
                    image = self.image_loader.load_image(str(image_path))
                    if image is None:
                        stats['failed'] += 1
                        continue
                    
                    # Apply basic processing
                    processed_image = image.copy()
                    
                    # Convert to grayscale if needed (convert_to_grayscale_rgb)
                    if len(processed_image.shape) == 3:
                        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                        processed_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
                    
                    # Downsample (downsample_image)
                    if self.image_processor.downsample_factor != 1.0:
                        height, width = processed_image.shape[:2]
                        new_height = int(height / self.image_processor.downsample_factor)
                        new_width = int(width / self.image_processor.downsample_factor)
                        processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Apply median filter (apply_median_filter)
                    if self.image_processor.median_filter_size > 1:
                        processed_image = cv2.medianBlur(processed_image, self.image_processor.median_filter_size)
                    
                    # Normalize (normalize_image)
                    if normalize:
                        p_low = np.percentile(processed_image, 1)
                        p_high = np.percentile(processed_image, 99)
                        image_norm = np.clip((processed_image - p_low) / (p_high - p_low), 0, 1)
                        processed_image = (image_norm * 255).astype(np.uint8)
                    
                    # Save if output directory specified
                    if output_dir:
                        output_filename = f"{directory.name}_{image_path.stem}_processed{image_path.suffix}"
                        output_path = output_dir / output_filename
                        
                        if ImageSaver.save_image(processed_image, str(output_path)):
                            stats['processed'] += 1
                        else:
                            stats['failed'] += 1
                    else:
                        stats['processed'] += 1
                        
                except Exception as e:
                    print(f"Error processing {image_path.name}: {e}")
                    stats['failed'] += 1
        
        return stats