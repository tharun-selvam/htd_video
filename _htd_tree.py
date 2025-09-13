import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import math

@dataclass
class TreeNode:
    """Represents a node in the hierarchical binary tree."""
    start_frame: int       # Start frame number
    end_frame: int         # End frame number
    frame_count: int       # Number of frames in this node
    depth: int             # Depth in tree (0 = root)
    node_id: str           # Unique identifier for the node
    
    # Children (None for leaf nodes)
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None
    
    # Data for leaf nodes
    embeddings: Optional[torch.Tensor] = None  # Shape: [N, 2048] for leaf nodes
    variance_score: Optional[float] = None     # V score for leaf nodes
    is_complex: Optional[bool] = None          # Whether V > threshold
    
    # For future VLM annotation
    vlm_labels: Optional[Dict] = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.left_child is None and self.right_child is None
    
    def get_duration_seconds(self, fps: float) -> float:
        """Convert frame count to duration in seconds."""
        return self.frame_count / fps


def build_hierarchical_tree(scene_start_frame: int, 
                          scene_end_frame: int, 
                          scene_embeddings: torch.Tensor,
                          fps: float,
                          d_min_frames: int,
                          scene_id: int = 0) -> TreeNode:
    """
    Builds a hierarchical binary tree for a scene using fixed-frame intervals.
    
    Args:
        scene_start_frame: Start frame of the scene
        scene_end_frame: End frame of the scene
        scene_embeddings: Embeddings for the entire scene [N, 2048]
        fps: Frames per second of the video (for duration display only)
        d_min_frames: Minimum number of frames for leaf nodes
        scene_id: ID of the scene (for node naming)
    
    Returns:
        TreeNode: Root node of the hierarchical tree
    """
    
    def _build_recursive(start_frame: int, 
                        end_frame: int, 
                        depth: int, 
                        node_path: str) -> TreeNode:
        """Recursive helper function to build the tree."""
        
        frame_count = end_frame - start_frame
        node_id = f"scene_{scene_id}_{node_path}"
        
        # Create current node
        node = TreeNode(
            start_frame=start_frame,
            end_frame=end_frame,
            frame_count=frame_count,
            depth=depth,
            node_id=node_id
        )
        
        # Base case: if frame_count <= d_min_frames, this is a leaf node
        if frame_count <= d_min_frames:
            # Extract embeddings for this frame range
            # Adjust for scene offset (embeddings start from scene beginning)
            relative_start = start_frame - scene_start_frame
            relative_end = end_frame - scene_start_frame
            
            # Clamp to valid embedding range
            relative_start = max(0, min(relative_start, len(scene_embeddings) - 1))
            relative_end = max(1, min(relative_end, len(scene_embeddings)))
            
            if relative_start < relative_end:
                node.embeddings = scene_embeddings[relative_start:relative_end]
                duration_sec = node.get_duration_seconds(fps)
                print(f"    -> Leaf {node_id}: {frame_count} frames ({duration_sec:.2f}s), embeddings shape: {node.embeddings.shape}")
            else:
                # Handle edge case where we have no embeddings
                node.embeddings = torch.zeros(1, 2048)  # Single zero embedding
                duration_sec = node.get_duration_seconds(fps)
                print(f"    -> Leaf {node_id}: {frame_count} frames ({duration_sec:.2f}s), no valid embeddings (edge case)")
            
            return node
        
        # Recursive case: split into two children
        mid_frame = start_frame + frame_count // 2
        
        duration_sec = node.get_duration_seconds(fps)
        print(f"  -> Internal node {node_id}: {frame_count} frames ({duration_sec:.2f}s), splitting at frame {mid_frame}")
        
        # Create left and right children
        node.left_child = _build_recursive(start_frame, mid_frame, depth + 1, node_path + "L")
        node.right_child = _build_recursive(mid_frame, end_frame, depth + 1, node_path + "R")
        
        return node
    
    total_frames = scene_end_frame - scene_start_frame
    duration_sec = total_frames / fps
    
    print(f"[INFO] Building hierarchical tree for scene {scene_id}")
    print(f"  -> Scene frames: {scene_start_frame} to {scene_end_frame} ({total_frames} frames, {duration_sec:.2f}s)")
    print(f"  -> d_min: {d_min_frames} frames ({d_min_frames/fps:.2f}s)")
    print(f"  -> Scene embeddings shape: {scene_embeddings.shape}")
    
    root = _build_recursive(scene_start_frame, scene_end_frame, 0, "root")
    
    # Count leaf nodes
    leaf_count = count_leaf_nodes(root)
    print(f"  -> Created tree with {leaf_count} leaf nodes")
    
    return root


def count_leaf_nodes(root: TreeNode) -> int:
    """Count the number of leaf nodes in the tree."""
    if root.is_leaf():
        return 1
    
    count = 0
    if root.left_child:
        count += count_leaf_nodes(root.left_child)
    if root.right_child:
        count += count_leaf_nodes(root.right_child)
    
    return count


def calculate_leaf_variances(root: TreeNode, threshold_tau: float = 100.0) -> Dict[str, float]:
    """
    Calculates motion variance scores for all leaf nodes and marks them as complex/simple.
    
    Args:
        root: Root node of the tree
        threshold_tau: Threshold for complex/simple classification
    
    Returns:
        Dict mapping node_id -> variance_score for all leaf nodes
    """
    
    def _calculate_recursive(node: TreeNode, results: Dict[str, float]):
        """Recursive helper to process all leaf nodes."""
        
        if node.is_leaf():
            if node.embeddings is not None and len(node.embeddings) > 1:
                # Calculate covariance matrix and variance score
                # embeddings shape: [N, 2048]
                covariance_matrix = torch.cov(node.embeddings.T)  # [2048, 2048]
                variance_score = torch.trace(covariance_matrix).item()
                
                node.variance_score = variance_score
                node.is_complex = variance_score > threshold_tau
                
                results[node.node_id] = variance_score
                
                status = "COMPLEX" if node.is_complex else "simple"
                print(f"    -> {node.node_id}: {node.frame_count} frames, V = {variance_score:.2f} ({status})")
                
            else:
                # Handle edge case with insufficient embeddings
                node.variance_score = 0.0
                node.is_complex = False
                results[node.node_id] = 0.0
                print(f"    -> {node.node_id}: {node.frame_count} frames, V = 0.00 (insufficient data)")
        
        else:
            # Recursively process children
            if node.left_child:
                _calculate_recursive(node.left_child, results)
            if node.right_child:
                _calculate_recursive(node.right_child, results)
    
    print(f"[INFO] Calculating variance scores for leaf nodes (τ = {threshold_tau})")
    
    variance_results = {}
    _calculate_recursive(root, variance_results)
    
    # Summary statistics
    if variance_results:
        scores = list(variance_results.values())
        complex_count = sum(1 for score in scores if score > threshold_tau)
        simple_count = len(scores) - complex_count
        
        print(f"[INFO] Variance calculation complete:")
        print(f"  -> Total leaf nodes: {len(scores)}")
        print(f"  -> Complex nodes (V > {threshold_tau}): {complex_count}")
        print(f"  -> Simple nodes (V ≤ {threshold_tau}): {simple_count}")
        print(f"  -> Mean variance: {np.mean(scores):.2f}")
        print(f"  -> Std variance: {np.std(scores):.2f}")
    
    return variance_results


def get_complex_leaf_nodes(root: TreeNode) -> List[TreeNode]:
    """
    Returns a list of all leaf nodes marked as complex (for VLM annotation).
    
    Args:
        root: Root node of the tree
    
    Returns:
        List of TreeNode objects that are leaves and marked as complex
    """
    
    def _collect_recursive(node: TreeNode, complex_leaves: List[TreeNode]):
        if node.is_leaf():
            if node.is_complex:
                complex_leaves.append(node)
        else:
            if node.left_child:
                _collect_recursive(node.left_child, complex_leaves)
            if node.right_child:
                _collect_recursive(node.right_child, complex_leaves)
    
    complex_leaves = []
    _collect_recursive(root, complex_leaves)
    
    return complex_leaves


def print_tree_summary(root: TreeNode, fps: float, max_depth: int = 3):
    """
    Prints a summary of the tree structure (useful for debugging).
    
    Args:
        root: Root node of the tree
        fps: Frames per second (for duration display)
        max_depth: Maximum depth to print (to avoid overwhelming output)
    """
    
    def _print_recursive(node: TreeNode, indent: str = "", depth: int = 0):
        if depth > max_depth:
            return
            
        # Print current node info
        node_type = "LEAF" if node.is_leaf() else "INTERNAL"
        duration_str = f"{node.frame_count} frames ({node.get_duration_seconds(fps):.2f}s)"
        
        if node.is_leaf():
            if node.variance_score is not None:
                complexity = "COMPLEX" if node.is_complex else "simple"
                print(f"{indent}{node.node_id} ({node_type}, {duration_str}, V={node.variance_score:.2f}, {complexity})")
            else:
                print(f"{indent}{node.node_id} ({node_type}, {duration_str})")
        else:
            print(f"{indent}{node.node_id} ({node_type}, {duration_str})")
        
        # Print children
        if not node.is_leaf() and depth < max_depth:
            if node.left_child:
                _print_recursive(node.left_child, indent + "  ├─ ", depth + 1)
            if node.right_child:
                _print_recursive(node.right_child, indent + "  └─ ", depth + 1)
    
    print(f"[INFO] Tree structure (showing up to depth {max_depth}):")
    _print_recursive(root)


def frames_to_seconds(frames: int, fps: float) -> float:
    """Convert frame count to seconds."""
    return frames / fps


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert seconds to frame count."""
    return int(seconds * fps)


# Example usage function to integrate with your main pipeline
def process_scene_with_tree(scene_embeddings: torch.Tensor,
                          scene_start_frame: int,
                          scene_end_frame: int,
                          fps: float,
                          scene_id: int,
                          d_min_frames: int,
                          threshold_tau: float = 100.0) -> Tuple[TreeNode, Dict[str, float], List[TreeNode]]:
    """
    Complete processing of a scene: build tree, calculate variances, identify complex nodes.
    
    Args:
        scene_embeddings: Embeddings for the scene [N, 2048]
        scene_start_frame: First frame of the scene
        scene_end_frame: Last frame of the scene
        fps: Frames per second
        scene_id: Scene identifier
        d_min_frames: Minimum frames for leaf nodes
        threshold_tau: Variance threshold for complex/simple classification
    
    Returns:
        Tuple of (root_node, variance_results, complex_leaf_nodes)
    """
    
    # Build the hierarchical tree
    root = build_hierarchical_tree(
        scene_start_frame, scene_end_frame, scene_embeddings, 
        fps, d_min_frames, scene_id
    )
    
    # Calculate variance scores for leaf nodes
    variance_results = calculate_leaf_variances(root, threshold_tau)
    
    # Get complex leaf nodes (these will need VLM annotation)
    complex_leaves = get_complex_leaf_nodes(root)
    
    # Print summary
    print_tree_summary(root, fps)
    
    return root, variance_results, complex_leaves