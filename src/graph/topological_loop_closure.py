# INFO:root:num timesteps 3670, episode_id 196 elapsed_time: 1h 36m 33.7s, Progress: 36.6% (366/1000), Average SR/SPL: 0.27049/0.13304, Avg episode time: 1m 53.1s, Est. remaining: 19h 54m 48.8s

import numpy as np
import cv2
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from typing import List, Tuple, Dict, Optional, Any
import warnings
from dataclasses import dataclass
from collections import defaultdict

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("GUDHI not installed. Using simplified persistence computation.")

@dataclass
class PersistencePoint:
    """Represents a point in persistence diagram"""
    dimension: int
    birth: float
    death: float
    
    @property
    def persistence(self):
        return self.death - self.birth
    
    @property
    def midpoint(self):
        return (self.birth + self.death) / 2


class TopologicalLoopDetector:
    """
    Training-free loop closure detection using topological signatures.
    Uses persistent homology to identify loops and spatial structures.
    """
    def __init__(self, 
                 max_edge_length: float = 5.0,
                 persistence_threshold: float = 0.1,
                 wasserstein_threshold: float = 2.0,
                 min_loop_size: int = 10):
        """
        Initialize the topological loop detector.
        
        Args:
            max_edge_length: Maximum edge length for Vietoris-Rips complex
            persistence_threshold: Minimum persistence to consider a feature significant
            wasserstein_threshold: Threshold for matching topological signatures
            min_loop_size: Minimum number of nodes to form a valid loop
        """
        self.max_edge_length = max_edge_length
        self.persistence_threshold = persistence_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.min_loop_size = min_loop_size
        
        self.historical_signatures = []
        self.signature_locations = []
        self.loop_closures_detected = []
        
    def extract_trajectory_points(self, trajectory: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Extract 3D points from robot trajectory.
        
        Args:
            trajectory: List of (x, y, theta) poses
            
        Returns:
            Array of 3D points (x, y, z) where z encodes orientation
        """
        points = []
        for x, y, theta in trajectory:
            # Include orientation as third dimension for richer topology
            z = np.sin(theta) * 0.5  # Scale orientation contribution
            points.append([x, y, z])
        return np.array(points)
    
    def build_vietoris_rips_complex(self, points: np.ndarray) -> 'SimplicalComplex':
        """
        Build Vietoris-Rips complex from point cloud.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Simplicial complex for persistence computation
        """
        if GUDHI_AVAILABLE:
            # Use GUDHI for efficient computation
            rips_complex = gudhi.RipsComplex(
                points=points,
                max_edge_length=self.max_edge_length
            )
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            return simplex_tree
        else:
            # Simplified implementation without GUDHI
            return self._build_simple_complex(points)
    
    def _build_simple_complex(self, points: np.ndarray) -> Dict:
        """
        Simplified complex construction when GUDHI is not available.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Dictionary representing simplicial complex
        """
        n_points = len(points)
        distances = squareform(pdist(points))
        
        # Build edge list (1-simplices)
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distances[i, j] <= self.max_edge_length:
                    edges.append((i, j, distances[i, j]))
        
        # Build triangle list (2-simplices)
        # triangles = []
        # for i in range(n_points):
        #     for j in range(i + 1, n_points):
        #         for k in range(j + 1, n_points):
        #             if (distances[i, j] <= self.max_edge_length and
        #                 distances[j, k] <= self.max_edge_length and
        #                 distances[i, k] <= self.max_edge_length):
        #                 max_edge = max(distances[i, j], distances[j, k], distances[i, k])
        #                 triangles.append((i, j, k, max_edge))
        
        return {
            'vertices': list(range(n_points)),
            'edges': edges,
            # 'triangles': triangles,
            'distances': distances
        }
    
    def compute_persistence(self, complex_data) -> List[PersistencePoint]:
        """
        Compute persistent homology of the complex.
        
        Args:
            complex_data: Simplicial complex
            
        Returns:
            List of persistence points
        """
        if GUDHI_AVAILABLE:
            # Use GUDHI's persistence computation
            complex_data.compute_persistence()
            persistence = complex_data.persistence()
            
            persistence_points = []
            for dim, (birth, death) in persistence:
                if death - birth > self.persistence_threshold:
                    persistence_points.append(
                        PersistencePoint(dim, birth, death)
                    )
            return persistence_points
        else:
            # Simplified persistence computation
            return self._compute_simple_persistence(complex_data)
    
    def _compute_simple_persistence(self, complex_data: Dict) -> List[PersistencePoint]:
        """
        Simplified persistence computation using Union-Find.
        
        Args:
            complex_data: Dictionary with complex information
            
        Returns:
            List of persistence points
        """
        persistence_points = []
        
        # Detect 1-dimensional features (loops)
        loops = self._detect_loops_union_find(complex_data)
        for birth, death in loops:
            if death - birth > self.persistence_threshold:
                persistence_points.append(
                    PersistencePoint(1, birth, death)
                )
        
        return persistence_points
    
    def _detect_loops_union_find(self, complex_data: Dict) -> List[Tuple[float, float]]:
        """
        Detect loops using Union-Find algorithm.
        
        Args:
            complex_data: Dictionary with edges and distances
            
        Returns:
            List of (birth, death) pairs for loops
        """
        edges = sorted(complex_data['edges'], key=lambda x: x[2])
        n_vertices = len(complex_data['vertices'])
        
        # Union-Find data structure
        parent = list(range(n_vertices))
        rank = [0] * n_vertices
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        loops = []
        for i, j, weight in edges:
            if find(i) == find(j):
                # Found a loop
                loops.append((0, weight))  # Birth at 0, death at weight
            else:
                union(i, j)
        
        return loops
    
    def compute_topological_signature(self, 
                                     trajectory: List[Tuple[float, float, float]],
                                     visual_features: Optional[np.ndarray] = None) -> Dict:
        """
        Compute complete topological signature for a trajectory.
        
        Args:
            trajectory: List of robot poses
            visual_features: Optional visual features for enriched topology
            
        Returns:
            Dictionary containing topological signature
        """
        # Extract trajectory points
        points = self.extract_trajectory_points(trajectory)
        
        # Enrich with visual features if available
        if visual_features is not None:
            # Combine spatial and visual information
            combined_points = self._combine_spatial_visual(points, visual_features)
        else:
            combined_points = points
        
        # Build complex and compute persistence
        complex_data = self.build_vietoris_rips_complex(combined_points)
        persistence_points = self.compute_persistence(complex_data)
        
        # Extract signature features
        signature = {
            'persistence_diagram': persistence_points,
            'betti_numbers': self._compute_betti_numbers(persistence_points),
            'persistence_landscape': self._compute_persistence_landscape(persistence_points),
            'total_persistence': sum(p.persistence for p in persistence_points),
            'max_persistence': max([p.persistence for p in persistence_points], default=0),
            'num_loops': sum(1 for p in persistence_points if p.dimension == 1),
            'trajectory_length': len(trajectory),
            'spatial_extent': self._compute_spatial_extent(points)
        }
        
        return signature
    
    def _combine_spatial_visual(self, 
                                spatial_points: np.ndarray,
                                visual_features: np.ndarray) -> np.ndarray:
        """
        Combine spatial and visual features for richer topology.
        
        Args:
            spatial_points: Nx3 spatial coordinates
            visual_features: NxD visual feature vectors
            
        Returns:
            Combined point cloud
        """
        # Normalize visual features
        visual_norm = visual_features / (np.linalg.norm(visual_features, axis=1, keepdims=True) + 1e-8)
        
        # Weight visual features appropriately
        visual_weight = 0.3
        visual_scaled = visual_norm * visual_weight
        
        # Concatenate or project to maintain 3D for visualization
        if visual_features.shape[1] > 3:
            # Use PCA-like projection (without training)
            u, s, vt = np.linalg.svd(visual_features, full_matrices=False)
            visual_projected = u[:, :3] * s[:3]
            visual_scaled = visual_projected * visual_weight
        
        # Combine spatial and visual
        combined = np.hstack([spatial_points, visual_scaled[:, :min(3, visual_scaled.shape[1])]])
        
        return combined
    
    def _compute_betti_numbers(self, persistence_points: List[PersistencePoint]) -> Dict[int, int]:
        """
        Compute Betti numbers from persistence diagram.
        
        Args:
            persistence_points: List of persistence points
            
        Returns:
            Dictionary mapping dimension to Betti number
        """
        betti = defaultdict(int)
        for point in persistence_points:
            if point.persistence > self.persistence_threshold:
                betti[point.dimension] += 1
        return dict(betti)
    
    def _compute_persistence_landscape(self, 
                                      persistence_points: List[PersistencePoint],
                                      resolution: int = 50) -> np.ndarray:
        """
        Compute persistence landscape for stable vectorization.
        
        Args:
            persistence_points: List of persistence points
            resolution: Number of samples in landscape
            
        Returns:
            Persistence landscape as vector
        """
        if not persistence_points:
            return np.zeros(resolution)
        
        # Sample points along diagonal
        t_values = np.linspace(0, self.max_edge_length, resolution)
        landscape = np.zeros(resolution)
        
        for point in persistence_points:
            if point.dimension == 1:  # Focus on loops
                for i, t in enumerate(t_values):
                    if point.birth <= t <= point.death:
                        height = min(t - point.birth, point.death - t)
                        landscape[i] = max(landscape[i], height)
        
        return landscape
    
    def _compute_spatial_extent(self, points: np.ndarray) -> float:
        """
        Compute spatial extent of trajectory.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Measure of spatial extent
        """
        if len(points) < 2:
            return 0.0
        
        # Compute convex hull volume as measure of extent
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points[:, :2])  # Use 2D projection
            return hull.volume  # Area in 2D
        except:
            # Fallback to bounding box
            bbox_size = points.max(axis=0) - points.min(axis=0)
            return np.prod(bbox_size[:2])
    
    def wasserstein_distance(self, 
                            signature1: Dict,
                            signature2: Dict,
                            p: int = 2) -> float:
        """
        Compute Wasserstein distance between two topological signatures.
        
        Args:
            signature1: First topological signature
            signature2: Second topological signature
            p: Order of Wasserstein distance (1 or 2)
            
        Returns:
            Wasserstein distance
        """
        diagram1 = signature1['persistence_diagram']
        diagram2 = signature2['persistence_diagram']
        
        if not diagram1 and not diagram2:
            return 0.0
        elif not diagram1 or not diagram2:
            return float('inf')
        
        # Separate by dimension
        dim1_diag1 = [(p.birth, p.death) for p in diagram1 if p.dimension == 1]
        dim1_diag2 = [(p.birth, p.death) for p in diagram2 if p.dimension == 1]
        
        # Compute Wasserstein distance
        distance = self._compute_wasserstein(dim1_diag1, dim1_diag2, p)
        
        # Also consider landscape distance for robustness
        landscape_dist = np.linalg.norm(
            signature1['persistence_landscape'] - signature2['persistence_landscape']
        )
        
        # Weighted combination
        return 0.7 * distance + 0.3 * landscape_dist
    
    def _compute_wasserstein(self, 
                           diagram1: List[Tuple[float, float]],
                           diagram2: List[Tuple[float, float]],
                           p: int = 2) -> float:
        """
        Compute Wasserstein distance between persistence diagrams.
        
        Args:
            diagram1: First persistence diagram as list of (birth, death) pairs
            diagram2: Second persistence diagram  
            p: Order of Wasserstein distance
            
        Returns:
            Wasserstein distance
        """
        if not diagram1 and not diagram2:
            return 0.0
        
        # Add diagonal points for optimal matching
        diag1_aug = diagram1 + [(0, 0)] * len(diagram2)
        diag2_aug = diagram2 + [(0, 0)] * len(diagram1)
        
        # Compute pairwise distances
        n = len(diag1_aug)
        cost_matrix = np.zeros((n, n))
        
        for i, (b1, d1) in enumerate(diag1_aug[:len(diagram1)]):
            for j, (b2, d2) in enumerate(diag2_aug[:len(diagram2)]):
                cost_matrix[i, j] = ((b1 - b2)**2 + (d1 - d2)**2) ** (p/2)
            
            # Distance to diagonal
            for j in range(len(diagram2), n):
                cost_matrix[i, j] = ((d1 - b1) / 2) ** p
        
        # Distance from diagonal to diagram2
        for i in range(len(diagram1), n):
            for j, (b2, d2) in enumerate(diag2_aug[:len(diagram2)]):
                cost_matrix[i, j] = ((d2 - b2) / 2) ** p
        
        # Use Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return (cost_matrix[row_ind, col_ind].sum()) ** (1/p)
    
    def detect_loop_closure(self,
                           current_trajectory: List[Tuple[float, float, float]],
                           current_visual_features: Optional[np.ndarray] = None,
                           search_radius: float = 10.0) -> Tuple[bool, Optional[int], float]:
        """
        Detect if current trajectory forms a loop with any historical trajectory.
        
        Args:
            current_trajectory: Current robot trajectory
            current_visual_features: Optional visual features
            search_radius: Spatial search radius for candidate loops
            
        Returns:
            Tuple of (loop_detected, matched_index, confidence)
        """
        if len(current_trajectory) < self.min_loop_size:
            return False, None, 0.0
        
        # Compute current signature
        current_signature = self.compute_topological_signature(
            current_trajectory,
            current_visual_features
        )
        
        # Get current position
        current_pos = np.array(current_trajectory[-1][:2])
        
        best_match = None
        best_distance = float('inf')
        
        # Search through historical signatures
        for idx, (hist_signature, hist_location) in enumerate(
            zip(self.historical_signatures, self.signature_locations)
        ):
            # Spatial gating for efficiency
            hist_pos = np.array(hist_location[:2])
            if np.linalg.norm(current_pos - hist_pos) > search_radius:
                continue
            
            # Compute topological distance
            distance = self.wasserstein_distance(current_signature, hist_signature)
            
            if distance < best_distance:
                best_distance = distance
                best_match = idx
        
        # Check if we found a loop
        loop_detected = best_distance < self.wasserstein_threshold
        
        # Compute confidence based on distance
        if loop_detected:
            confidence = np.exp(-best_distance / self.wasserstein_threshold)
        else:
            confidence = 0.0
        
        # Store current signature for future matching
        self.historical_signatures.append(current_signature)
        self.signature_locations.append(current_trajectory[-1])
        
        if loop_detected:
            self.loop_closures_detected.append({
                'current_idx': len(self.historical_signatures) - 1,
                'matched_idx': best_match,
                'distance': best_distance,
                'confidence': confidence
            })
        
        return loop_detected, best_match, confidence
    
    def visualize_persistence_diagram(self, signature: Dict) -> np.ndarray:
        """
        Create visualization of persistence diagram.
        
        Args:
            signature: Topological signature
            
        Returns:
            Image array for visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot persistence diagram
        diagram = signature['persistence_diagram']
        if diagram:
            births = [p.birth for p in diagram if p.dimension == 1]
            deaths = [p.death for p in diagram if p.dimension == 1]
            
            ax1.scatter(births, deaths, c='blue', s=50, alpha=0.6, label='1-cycles (loops)')
            
            # Plot diagonal
            max_val = max(deaths + births) if (deaths + births) else 1
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            ax1.set_xlabel('Birth')
            ax1.set_ylabel('Death')
            ax1.set_title('Persistence Diagram')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot persistence landscape
        landscape = signature['persistence_landscape']
        ax2.plot(landscape, 'g-', linewidth=2)
        ax2.fill_between(range(len(landscape)), landscape, alpha=0.3)
        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('Landscape Height')
        ax2.set_title('Persistence Landscape')
        ax2.grid(True, alpha=0.3)
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        return img
    
    def get_loop_closure_constraints(self) -> List[Dict]:
        """
        Get loop closure constraints for pose graph optimization.
        
        Returns:
            List of loop closure constraints
        """
        constraints = []
        for closure in self.loop_closures_detected:
            constraints.append({
                'type': 'loop_closure',
                'from_idx': closure['current_idx'],
                'to_idx': closure['matched_idx'],
                'confidence': closure['confidence'],
                'information_matrix': np.eye(3) * closure['confidence']  # Simple weighting
            })
        return constraints

class TopologicalNavigationIntegration:
    def __init__(self):
        self.loop_detector = TopologicalLoopDetector()
        self.trajectory = []
        self.visual_features_history = []
        
    def update(self, 
               current_pose: Tuple[float, float, float],
               current_observation: np.ndarray) -> Dict:
        """
        Update navigation with new observation and check for loops.
        
        Args:
            current_pose: Current robot pose (x, y, theta)
            current_observation: Current visual observation
            
        Returns:
            Navigation update including loop closure information
        """
        # Add to trajectory
        self.trajectory.append(current_pose)
        
        # Extract visual features (using pre-trained but frozen features)
        visual_features = self.extract_visual_features(current_observation)
        self.visual_features_history.append(visual_features)
        
        # Check for loop closure
        loop_detected, matched_idx, confidence = self.loop_detector.detect_loop_closure(
            self.trajectory,
            np.array(self.visual_features_history)
        )
        
        result = {
            'loop_detected': loop_detected,
            'matched_index': matched_idx,
            'confidence': confidence,
            'trajectory_length': len(self.trajectory)
        }
        
        if loop_detected:
            # Get topological constraints for pose correction
            constraints = self.loop_detector.get_loop_closure_constraints()
            result['constraints'] = constraints
            
            # Trigger pose graph optimization (external)
            result['trigger_optimization'] = True
            
        return result
    
    def extract_visual_features(self, observation: np.ndarray) -> np.ndarray:
        """
        Extract visual features using classical methods (training-free).
        
        Args:
            observation: RGB image
            
        -> Feature vector
        """
        orb = cv2.ORB_create(nfeatures=100)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Aggregate descriptors into fixed-size vector
            feature_vector = np.mean(descriptors.astype(np.float32), axis=0)
        else:
            feature_vector = np.zeros(32)  # ORB descriptor size
            
        return feature_vector


# if __name__ == "__main__":
#     # Demo of the topological loop detector
#     print("Topological Loop Closure Detection Demo")
#     print("=" * 50)
    
#     # Create detector
#     detector = TopologicalLoopDetector()
    
#     # Simulate a trajectory that forms a loop
#     trajectory = []
#     for t in np.linspace(0, 2*np.pi, 50):
#         x = 5 * np.cos(t)
#         y = 5 * np.sin(t)
#         theta = t + np.pi/2
#         trajectory.append((x, y, theta))
    
#     # Add some noise to make it realistic
#     trajectory = [(x + np.random.randn()*0.1, 
#                   y + np.random.randn()*0.1, 
#                   theta) for x, y, theta in trajectory]
    
#     # Compute topological signature
#     signature = detector.compute_topological_signature(trajectory)
    
#     print(f"Trajectory points: {len(trajectory)}")
#     print(f"Betti numbers: {signature['betti_numbers']}")
#     print(f"Number of loops detected: {signature['num_loops']}")
#     print(f"Total persistence: {signature['total_persistence']:.3f}")
#     print(f"Spatial extent: {signature['spatial_extent']:.3f}")
    
#     # Test loop closure detection
#     detector.historical_signatures.append(signature)
#     detector.signature_locations.append(trajectory[-1])
    
#     # Create a similar trajectory (should be detected as loop)
#     similar_trajectory = trajectory[:40] + [(x+0.5, y+0.5, theta) for x, y, theta in trajectory[40:]]
    
#     loop_detected, matched_idx, confidence = detector.detect_loop_closure(similar_trajectory)
    
#     print(f"\nLoop closure detection:")
#     print(f"Loop detected: {loop_detected}")
#     print(f"Confidence: {confidence:.3f}")
    
#     print("\nTopological Loop Closure module successfully implemented!")