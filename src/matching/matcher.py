"""
Embedding matching and similarity computation utilities
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from typing import List, Tuple, Dict, Any


class EmbeddingMatcher:
    """Handles embedding normalization and matching operations"""
    
    def match_embeddings(
        self, 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray,
        filenames_a: List[str],
        filenames_b: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Complete matching pipeline for embeddings
        
        Args:
            embeddings_a: Embeddings from Set A
            embeddings_b: Embeddings from Set B  
            filenames_a: Filenames corresponding to Set A embeddings
            filenames_b: Filenames corresponding to Set B embeddings
            
        Returns:
            Tuple of (matched_pairs, summary_stats)
        """
        # Normalize embeddings (normalize_embeddings)
        embeddings_a_norm = normalize(embeddings_a, norm='l2', axis=1)
        embeddings_b_norm = normalize(embeddings_b, norm='l2', axis=1)
        
        # Compute cosine distances (compute_cosine_distances)
        distances = cosine_distances(embeddings_a_norm, embeddings_b_norm)
        
        # Find best matches (find_best_matches) - for each Set B, find best Set A
        best_matches = np.argmin(distances, axis=0)
        
        # Create matched pairs with metadata
        matched_pairs = []
        for j, match_idx in enumerate(best_matches):
            set_b_file = filenames_b[j]
            set_a_file = filenames_a[match_idx]
            distance = distances[match_idx, j]
            
            matched_pairs.append({
                'set_a': set_a_file,
                'set_b': set_b_file,
                'cosine_distance': float(distance)
            })
        
        # Calculate summary statistics
        distances_used = [pair['cosine_distance'] for pair in matched_pairs]
        summary = {
            'set_a_count': len(filenames_a),
            'set_b_count': len(filenames_b),
            'successful_matches': len(matched_pairs),
            'average_distance': float(np.mean(distances_used)),
            'min_distance': float(np.min(distances_used)),
            'max_distance': float(np.max(distances_used)),
            'std_distance': float(np.std(distances_used))
        }
        
        return matched_pairs, summary


class MatchResults:
    """Handles formatting and saving of match results"""
    
    @staticmethod
    def save_results(
        matched_pairs: List[Dict[str, Any]], 
        summary: Dict[str, Any],
        output_file: str
    ) -> None:
        """
        Save results in both JSON and tuple formats
        
        Args:
            matched_pairs: List of match dictionaries
            summary: Summary statistics
            output_file: Base output filename
        """
        import json
        from pathlib import Path
        
        # Save JSON format
        output_data = {
            'matches': matched_pairs,
            'summary': summary
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save tuple format (to_tuple_format)
        tuple_format = [(pair['set_a'], pair['set_b']) for pair in matched_pairs]
        tuple_output_file = Path(output_file).with_suffix('.py')
        
        with open(tuple_output_file, 'w') as f:
            f.write("# Matched pairs in required tuple format\n")
            f.write("matched_pairs = [\n")
            for pair in tuple_format:
                f.write(f"    {repr(pair)},\n")
            f.write("]\n")