import numpy as np
import cv2
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional, Any
import warnings
from dataclasses import dataclass
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import math

# ============================================================================
# COMPONENT 1: TEMPORAL GRAPH MEMORY NETWORKS (TGMN)
# ============================================================================

class TemporalGraphMemory:
    """
    Temporal Graph Memory Networks - Graphs that evolve over time
    Maintains historical graph states and temporal edges without training
    """
    
    def __init__(self, max_history: int = 100, temporal_decay: float = 0.95):
        self.graph_snapshots = deque(maxlen=max_history)
        self.temporal_edges = []
        self.temporal_decay = temporal_decay
        self.time_step = 0
        
    def add_snapshot(self, graph: Dict, timestamp: Optional[float] = None):
        """Add a new graph snapshot to temporal memory"""
        if timestamp is None:
            timestamp = self.time_step
            
        snapshot = {
            'graph': graph.copy(),
            'timestamp': timestamp,
            'nodes': list(graph.get('nodes', [])),
            'edges': list(graph.get('edges', []))
        }
        self.graph_snapshots.append(snapshot)
        self.time_step += 1
        
        # Create temporal edges to previous snapshots
        if len(self.graph_snapshots) > 1:
            self._create_temporal_edges()
    
    def _create_temporal_edges(self):
        """Create edges between same/similar nodes across time"""
        current = self.graph_snapshots[-1]
        previous = self.graph_snapshots[-2]
        
        for curr_node in current['nodes']:
            for prev_node in previous['nodes']:
                if self._nodes_similar(curr_node, prev_node):
                    # Physics-based temporal edge
                    delta_t = current['timestamp'] - previous['timestamp']
                    appearance_change = self._compute_appearance_delta(curr_node, prev_node)
                    temporal_stability = np.exp(-appearance_change * delta_t)
                    
                    edge = {
                        'type': 'temporal',
                        'from': (previous['timestamp'], prev_node),
                        'to': (current['timestamp'], curr_node),
                        'weight': temporal_stability,
                        'delta_t': delta_t
                    }
                    self.temporal_edges.append(edge)
    
    def _nodes_similar(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes represent the same object"""
        # Simple label matching - can be enhanced with visual features
        if node1.get('label') == node2.get('label'):
            # Check spatial proximity if positions available
            if 'position' in node1 and 'position' in node2:
                dist = np.linalg.norm(
                    np.array(node1['position']) - np.array(node2['position'])
                )
                return dist < 2.0  # Within 2 meters
            return True
        return False
    
    def _compute_appearance_delta(self, node1: Dict, node2: Dict) -> float:
        """Compute appearance change between nodes"""
        # Use color histogram difference if available
        if 'color_hist' in node1 and 'color_hist' in node2:
            return np.sum(np.abs(node1['color_hist'] - node2['color_hist']))
        return 0.1  # Default small change
    
    def predict_future_state(self, steps_ahead: int = 1) -> Dict:
        """Predict future graph state using physics-based forward model"""
        if len(self.graph_snapshots) < 2:
            return self.graph_snapshots[-1]['graph'] if self.graph_snapshots else {}
        
        # Extract motion patterns from temporal edges
        velocities = self._extract_velocities()
        current = self.graph_snapshots[-1]
        
        predicted = {
            'nodes': [],
            'edges': []
        }
        
        for node in current['nodes']:
            predicted_node = node.copy()
            if node['label'] in velocities:
                # Apply velocity to predict position
                vel = velocities[node['label']]
                if 'position' in predicted_node:
                    predicted_node['position'] = (
                        np.array(predicted_node['position']) + vel * steps_ahead
                    ).tolist()
            predicted['nodes'].append(predicted_node)
        
        return predicted
    
    def _extract_velocities(self) -> Dict:
        """Extract velocity patterns from temporal edges"""
        velocities = {}
        for edge in self.temporal_edges[-10:]:  # Use recent edges
            if edge['delta_t'] > 0:
                from_node = edge['from'][1]
                to_node = edge['to'][1]
                if 'position' in from_node and 'position' in to_node:
                    vel = (np.array(to_node['position']) - 
                          np.array(from_node['position'])) / edge['delta_t']
                    velocities[to_node['label']] = vel
        return velocities
    
    def get_temporal_context(self, node_label: str, time_window: int = 5) -> List[Dict]:
        """Get temporal context for a specific node"""
        context = []
        for snapshot in list(self.graph_snapshots)[-time_window:]:
            for node in snapshot['nodes']:
                if node.get('label') == node_label:
                    context.append({
                        'timestamp': snapshot['timestamp'],
                        'node': node
                    })
        return context


# ============================================================================
# COMPONENT 2: COUNTERFACTUAL GRAPH REASONING (CGR)
# ============================================================================

class CounterfactualGraphReasoner:
    """
    Counterfactual Graph Reasoning - Generate and evaluate "what-if" scenarios
    Training-free exploration of hypothetical graph modifications
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.generated_counterfactuals = []
        
    def generate_counterfactuals(self, 
                                scene_graph: Dict,
                                goal_graph: Dict,
                                frontier_nodes: List[Dict]) -> List[Dict]:
        """Generate plausible graph modifications without training"""
        counterfactuals = []
        
        # Hypothesis 1: Hidden connections behind frontiers
        hidden_connections = self._predict_hidden_connections(scene_graph, frontier_nodes)
        counterfactuals.extend(hidden_connections)
        
        # Hypothesis 2: Symmetric completions
        symmetric_completions = self._generate_symmetric_completions(scene_graph)
        counterfactuals.extend(symmetric_completions)
        
        # Hypothesis 3: Semantic predictions
        semantic_predictions = self._predict_semantic_extensions(scene_graph, goal_graph)
        counterfactuals.extend(semantic_predictions)
        
        # Hypothesis 4: Occlusion reasoning
        occlusion_predictions = self._predict_occluded_structure(scene_graph)
        counterfactuals.extend(occlusion_predictions)
        
        self.generated_counterfactuals = counterfactuals
        return counterfactuals
    
    def _predict_hidden_connections(self, 
                                   scene_graph: Dict,
                                   frontier_nodes: List[Dict]) -> List[Dict]:
        """Predict connections beyond visible frontiers using geometry"""
        counterfactuals = []
        
        for frontier in frontier_nodes:
            # Use geometric reasoning to predict unseen connections
            if 'position' in frontier:
                # Predict what might be beyond this frontier
                direction = self._estimate_frontier_direction(frontier, scene_graph)
                
                # Generate hypothetical nodes beyond frontier
                hypothetical_nodes = []
                for dist in [2, 4, 6]:  # meters
                    new_pos = np.array(frontier['position']) + direction * dist
                    hypothetical_nodes.append({
                        'label': 'predicted_space',
                        'position': new_pos.tolist(),
                        'confidence': np.exp(-dist/10)  # Decay with distance
                    })
                
                counterfactual = {
                    'hypothesis': 'hidden_path',
                    'type': 'spatial_extension',
                    'base_node': frontier,
                    'predicted_nodes': hypothetical_nodes,
                    'confidence': self._compute_geometric_plausibility(
                        frontier, hypothetical_nodes, scene_graph
                    )
                }
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def _generate_symmetric_completions(self, scene_graph: Dict) -> List[Dict]:
        """Generate completions based on detected symmetries"""
        counterfactuals = []
        
        # Detect symmetry axes
        symmetry_axes = self._detect_symmetry_axes(scene_graph)
        
        for axis in symmetry_axes:
            # Mirror graph across axis
            mirrored_nodes = []
            for node in scene_graph.get('nodes', []):
                if 'position' in node:
                    mirrored_pos = self._mirror_point(node['position'], axis)
                    # Check if this position is unobserved
                    if not self._position_exists(mirrored_pos, scene_graph):
                        mirrored_nodes.append({
                            'label': node['label'] + '_symmetric',
                            'position': mirrored_pos,
                            'original': node['label']
                        })
            
            if mirrored_nodes:
                counterfactual = {
                    'hypothesis': 'symmetric_structure',
                    'type': 'symmetry',
                    'axis': axis,
                    'predicted_nodes': mirrored_nodes,
                    'confidence': self._compute_symmetry_score(scene_graph, axis)
                }
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def _predict_semantic_extensions(self, 
                                    scene_graph: Dict,
                                    goal_graph: Dict) -> List[Dict]:
        """Predict extensions based on semantic relationships"""
        counterfactuals = []
        
        # Find partial matches between scene and goal
        partial_matches = self._find_partial_matches(scene_graph, goal_graph)
        
        for match in partial_matches:
            # Predict missing components
            missing_nodes = []
            for goal_node in goal_graph.get('nodes', []):
                if not self._node_in_graph(goal_node, scene_graph):
                    # Predict where this node might be
                    predicted_pos = self._predict_node_position(
                        goal_node, match, scene_graph
                    )
                    if predicted_pos is not None:
                        missing_nodes.append({
                            'label': goal_node['label'],
                            'position': predicted_pos,
                            'from_goal': True
                        })
            
            if missing_nodes:
                counterfactual = {
                    'hypothesis': 'semantic_completion',
                    'type': 'semantic',
                    'match_info': match,
                    'predicted_nodes': missing_nodes,
                    'confidence': len(match['matched_nodes']) / len(goal_graph.get('nodes', [1]))
                }
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def _predict_occluded_structure(self, scene_graph: Dict) -> List[Dict]:
        """Predict structure hidden by occlusions"""
        counterfactuals = []
        
        # Identify potential occlusions
        occlusions = self._detect_occlusions(scene_graph)
        
        for occlusion in occlusions:
            # Predict what might be behind
            hidden_structure = {
                'hypothesis': 'occluded_structure',
                'type': 'occlusion',
                'occluder': occlusion['occluder'],
                'predicted_nodes': [],
                'confidence': 0.0
            }
            
            # Use shape completion heuristics
            if occlusion['type'] == 'partial_object':
                completed = self._complete_partial_object(occlusion, scene_graph)
                hidden_structure['predicted_nodes'] = completed
                hidden_structure['confidence'] = 0.7
            
            if hidden_structure['predicted_nodes']:
                counterfactuals.append(hidden_structure)
        
        return counterfactuals
    
    def evaluate_counterfactual(self, 
                               counterfactual: Dict,
                               new_observations: Dict) -> float:
        """Evaluate how well a counterfactual matches new observations"""
        score = 0.0
        predicted_nodes = counterfactual.get('predicted_nodes', [])
        
        for pred_node in predicted_nodes:
            for obs_node in new_observations.get('nodes', []):
                if self._nodes_match(pred_node, obs_node):
                    score += pred_node.get('confidence', 1.0)
        
        return score / max(len(predicted_nodes), 1)
    
    def _estimate_frontier_direction(self, frontier: Dict, graph: Dict) -> np.ndarray:
        """Estimate the direction beyond a frontier"""
        # Simple heuristic: away from center of observed nodes
        positions = [n['position'] for n in graph.get('nodes', []) 
                    if 'position' in n]
        if positions:
            center = np.mean(positions, axis=0)
            direction = np.array(frontier['position']) - center
            return direction / (np.linalg.norm(direction) + 1e-6)
        return np.array([1, 0, 0])  # Default forward
    
    def _detect_symmetry_axes(self, graph: Dict) -> List[Dict]:
        """Detect potential symmetry axes in the graph"""
        axes = []
        positions = [n['position'] for n in graph.get('nodes', []) 
                    if 'position' in n]
        
        if len(positions) > 3:
            positions = np.array(positions)
            # PCA to find principal axes
            centered = positions - np.mean(positions, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Main axis
            main_axis = eigenvectors[:, np.argmax(eigenvalues)]
            axes.append({
                'direction': main_axis,
                'point': np.mean(positions, axis=0),
                'type': 'principal'
            })
        
        return axes
    
    def _mirror_point(self, point: List[float], axis: Dict) -> List[float]:
        """Mirror a point across an axis"""
        p = np.array(point)
        axis_point = np.array(axis['point'])
        axis_dir = np.array(axis['direction'])
        
        # Project point onto axis
        v = p - axis_point
        proj = np.dot(v, axis_dir) * axis_dir
        perp = v - proj
        
        # Mirror
        mirrored = axis_point + proj - perp
        return mirrored.tolist()
    
    def _compute_geometric_plausibility(self, base: Dict, predicted: List[Dict], graph: Dict) -> float:
        """Compute geometric plausibility of predictions"""
        # Check if predictions maintain reasonable spatial relationships
        score = 1.0
        for pred in predicted:
            if 'position' in pred:
                # Check distance constraints
                dist = np.linalg.norm(
                    np.array(pred['position']) - np.array(base['position'])
                )
                if dist > 10:  # Too far
                    score *= 0.5
        return score
    
    def _compute_symmetry_score(self, graph: Dict, axis: Dict) -> float:
        """Compute how symmetric the graph is around an axis"""
        score = 0.0
        count = 0
        
        for node in graph.get('nodes', []):
            if 'position' in node:
                mirrored = self._mirror_point(node['position'], axis)
                if self._position_exists(mirrored, graph, threshold=2.0):
                    score += 1.0
                count += 1
        
        return score / max(count, 1)
    
    def _position_exists(self, position: List[float], graph: Dict, threshold: float = 0.5) -> bool:
        """Check if a position already exists in the graph"""
        pos = np.array(position)
        for node in graph.get('nodes', []):
            if 'position' in node:
                if np.linalg.norm(np.array(node['position']) - pos) < threshold:
                    return True
        return False
    
    def _find_partial_matches(self, scene: Dict, goal: Dict) -> List[Dict]:
        """Find partial matches between scene and goal graphs"""
        matches = []
        # Simple label matching - can be enhanced
        scene_labels = {n['label'] for n in scene.get('nodes', [])}
        goal_labels = {n['label'] for n in goal.get('nodes', [])}
        
        common = scene_labels.intersection(goal_labels)
        if common:
            matches.append({
                'matched_nodes': list(common),
                'scene_only': list(scene_labels - goal_labels),
                'goal_only': list(goal_labels - scene_labels)
            })
        
        return matches
    
    def _node_in_graph(self, node: Dict, graph: Dict) -> bool:
        """Check if a node exists in graph"""
        for n in graph.get('nodes', []):
            if n.get('label') == node.get('label'):
                return True
        return False
    
    def _predict_node_position(self, node: Dict, match: Dict, graph: Dict) -> Optional[List[float]]:
        """Predict position of missing node based on partial match"""
        # Simple heuristic: place near matched nodes
        matched_positions = []
        for n in graph.get('nodes', []):
            if n.get('label') in match['matched_nodes'] and 'position' in n:
                matched_positions.append(n['position'])
        
        if matched_positions:
            # Place near center of matched nodes
            center = np.mean(matched_positions, axis=0)
            # Add small offset
            offset = np.random.randn(len(center)) * 2
            return (center + offset).tolist()
        return None
    
    def _detect_occlusions(self, graph: Dict) -> List[Dict]:
        """Detect potential occlusions in the scene"""
        occlusions = []
        # Simple heuristic: incomplete objects
        for node in graph.get('nodes', []):
            if node.get('label', '').endswith('_partial'):
                occlusions.append({
                    'type': 'partial_object',
                    'occluder': node,
                    'position': node.get('position')
                })
        return occlusions
    
    def _complete_partial_object(self, occlusion: Dict, graph: Dict) -> List[Dict]:
        """Complete a partially observed object"""
        completed = []
        # Simple completion heuristic
        if 'position' in occlusion['occluder']:
            pos = np.array(occlusion['occluder']['position'])
            # Add predicted completion
            completed.append({
                'label': occlusion['occluder']['label'].replace('_partial', '_completed'),
                'position': (pos + np.array([1, 0, 0])).tolist(),
                'type': 'completed'
            })
        return completed
    
    def _nodes_match(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes match"""
        if node1.get('label') == node2.get('label'):
            if 'position' in node1 and 'position' in node2:
                dist = np.linalg.norm(
                    np.array(node1['position']) - np.array(node2['position'])
                )
                return dist < 2.0
            return True
        return False


# ============================================================================
# COMPONENT 3: HIERARCHICAL TOPOLOGICAL-SEMANTIC GRAPHS (HTSG)
# ============================================================================

class HierarchicalLevel(Enum):
    PIXEL = 0
    OBJECT = 1
    REGION = 2
    ROOM = 3
    BUILDING = 4

class HierarchicalTopologicalGraph:
    """
    Multi-resolution graph representation capturing both fine and coarse semantics
    Uses algebraic multigrid and spectral methods without training
    """
    
    def __init__(self):
        self.levels = {
            HierarchicalLevel.PIXEL: None,
            HierarchicalLevel.OBJECT: None,
            HierarchicalLevel.REGION: None,
            HierarchicalLevel.ROOM: None,
            HierarchicalLevel.BUILDING: None
        }
        self.inter_level_connections = []
        
    def build_hierarchy(self, observation: np.ndarray, semantic_map: np.ndarray):
        """Build hierarchical graph from observation"""
        # Level 0: Pixel connectivity
        self.levels[HierarchicalLevel.PIXEL] = self._build_pixel_graph(observation)
        
        # Level 1: Object detection and graph
        self.levels[HierarchicalLevel.OBJECT] = self._build_object_graph(
            observation, semantic_map
        )
        
        # Level 2: Region clustering
        self.levels[HierarchicalLevel.REGION] = self._build_region_graph(
            self.levels[HierarchicalLevel.OBJECT]
        )
        
        # Level 3: Room detection
        self.levels[HierarchicalLevel.ROOM] = self._build_room_graph(
            self.levels[HierarchicalLevel.REGION]
        )
        
        # Level 4: Building structure
        self.levels[HierarchicalLevel.BUILDING] = self._build_building_graph(
            self.levels[HierarchicalLevel.ROOM]
        )
        
        # Build inter-level connections
        self._build_inter_level_connections()
    
    def _build_pixel_graph(self, observation: np.ndarray) -> nx.Graph:
        """Build pixel-level connectivity graph"""
        h, w = observation.shape[:2]
        G = nx.grid_2d_graph(h, w)
        
        # Add pixel attributes
        for (i, j) in G.nodes():
            if i < h and j < w:
                G.nodes[(i, j)]['color'] = observation[i, j]
                G.nodes[(i, j)]['level'] = HierarchicalLevel.PIXEL.value
        
        return G
    
    def _build_object_graph(self, observation: np.ndarray, semantic_map: np.ndarray) -> nx.Graph:
        """Build object-level graph from semantic segmentation"""
        G = nx.Graph()
        
        # Get unique objects
        unique_objects = np.unique(semantic_map)
        
        for obj_id in unique_objects:
            if obj_id == 0:  # Skip background
                continue
                
            # Find object pixels
            mask = (semantic_map == obj_id)
            positions = np.argwhere(mask)
            
            if len(positions) > 0:
                # Compute object center
                center = np.mean(positions, axis=0)
                
                # Add node
                G.add_node(obj_id, 
                          position=center,
                          size=len(positions),
                          level=HierarchicalLevel.OBJECT.value)
        
        # Add edges between nearby objects
        for n1 in G.nodes():
            for n2 in G.nodes():
                if n1 < n2:
                    dist = np.linalg.norm(
                        G.nodes[n1]['position'] - G.nodes[n2]['position']
                    )
                    if dist < 50:  # Proximity threshold
                        G.add_edge(n1, n2, weight=1.0/max(dist, 1))
        
        return G
    
    def _build_region_graph(self, object_graph: nx.Graph) -> nx.Graph:
        """Build region graph by clustering objects"""
        if object_graph is None or len(object_graph) == 0:
            return nx.Graph()
        
        # Use spectral clustering (training-free)
        G = nx.Graph()
        
        # Get Laplacian
        if len(object_graph) > 1:
            L = nx.laplacian_matrix(object_graph).toarray()
            
            # Compute eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(L)
            
            # Use Fiedler vector for bipartition
            fiedler_idx = np.argsort(eigenvalues)[1]
            fiedler = eigenvectors[:, fiedler_idx]
            
            # Partition nodes
            partition = fiedler > np.median(fiedler)
            
            # Create region nodes
            for region_id in [0, 1]:
                nodes_in_region = [n for i, n in enumerate(object_graph.nodes()) 
                                 if partition[i] == region_id]
                if nodes_in_region:
                    # Compute region center
                    positions = [object_graph.nodes[n]['position'] 
                               for n in nodes_in_region]
                    center = np.mean(positions, axis=0)
                    
                    G.add_node(region_id,
                             position=center,
                             objects=nodes_in_region,
                             level=HierarchicalLevel.REGION.value)
            
            # Connect adjacent regions
            if len(G) == 2:
                G.add_edge(0, 1)
        else:
            # Single object becomes single region
            G.add_node(0, 
                      position=list(object_graph.nodes())[0],
                      objects=list(object_graph.nodes()),
                      level=HierarchicalLevel.REGION.value)
        
        return G
    
    def _build_room_graph(self, region_graph: nx.Graph) -> nx.Graph:
        """Build room-level graph from regions"""
        G = nx.Graph()
        
        if region_graph is None or len(region_graph) == 0:
            return G
        
        # Simple heuristic: each connected component of regions forms a room
        components = list(nx.connected_components(region_graph))
        
        for room_id, component in enumerate(components):
            # Compute room center
            positions = [region_graph.nodes[n]['position'] for n in component]
            center = np.mean(positions, axis=0) if positions else [0, 0]
            
            G.add_node(room_id,
                      position=center,
                      regions=list(component),
                      level=HierarchicalLevel.ROOM.value)
        
        # Connect adjacent rooms (placeholder logic)
        for r1 in G.nodes():
            for r2 in G.nodes():
                if r1 < r2:
                    # Simple distance-based connection
                    dist = np.linalg.norm(
                        G.nodes[r1]['position'] - G.nodes[r2]['position']
                    )
                    if dist < 100:
                        G.add_edge(r1, r2)
        
        return G
    
    def _build_building_graph(self, room_graph: nx.Graph) -> nx.Graph:
        """Build building-level graph"""
        G = nx.Graph()
        
        if room_graph is None or len(room_graph) == 0:
            return G
        
        # Single building node containing all rooms
        G.add_node(0,
                  rooms=list(room_graph.nodes()),
                  level=HierarchicalLevel.BUILDING.value)
        
        return G
    
    def _build_inter_level_connections(self):
        """Build connections between hierarchy levels"""
        self.inter_level_connections = []
        
        # Connect objects to regions
        if self.levels[HierarchicalLevel.OBJECT] and self.levels[HierarchicalLevel.REGION]:
            for region in self.levels[HierarchicalLevel.REGION].nodes():
                region_data = self.levels[HierarchicalLevel.REGION].nodes[region]
                for obj in region_data.get('objects', []):
                    self.inter_level_connections.append({
                        'from': (HierarchicalLevel.OBJECT, obj),
                        'to': (HierarchicalLevel.REGION, region),
                        'type': 'contains'
                    })
    
    def cross_level_message_passing(self, iterations: int = 3):
        """Propagate information across hierarchy levels"""
        for _ in range(iterations):
            # Bottom-up aggregation
            self._bottom_up_aggregation()
            
            # Top-down propagation
            self._top_down_propagation()
    
    def _bottom_up_aggregation(self):
        """Aggregate information from fine to coarse levels"""
        # Object to Region
        if self.levels[HierarchicalLevel.OBJECT] and self.levels[HierarchicalLevel.REGION]:
            for region in self.levels[HierarchicalLevel.REGION].nodes():
                region_data = self.levels[HierarchicalLevel.REGION].nodes[region]
                objects = region_data.get('objects', [])
                
                if objects:
                    # Aggregate object features
                    sizes = [self.levels[HierarchicalLevel.OBJECT].nodes[o].get('size', 0) 
                            for o in objects]
                    region_data['total_size'] = sum(sizes)
    
    def _top_down_propagation(self):
        """Propagate context from coarse to fine levels"""
        # Room to Region
        if self.levels[HierarchicalLevel.ROOM] and self.levels[HierarchicalLevel.REGION]:
            for room in self.levels[HierarchicalLevel.ROOM].nodes():
                room_data = self.levels[HierarchicalLevel.ROOM].nodes[room]
                regions = room_data.get('regions', [])
                
                for region in regions:
                    if region in self.levels[HierarchicalLevel.REGION].nodes:
                        self.levels[HierarchicalLevel.REGION].nodes[region]['room'] = room
    
    def query_at_level(self, level: HierarchicalLevel) -> Optional[nx.Graph]:
        """Query graph at specific level"""
        return self.levels.get(level)
    
    def coarsen_graph(self, graph: nx.Graph, method: str = 'algebraic') -> nx.Graph:
        """Coarsen graph using algebraic multigrid (training-free)"""
        if len(graph) <= 1:
            return graph
        
        if method == 'algebraic':
            # Algebraic multigrid coarsening
            A = nx.adjacency_matrix(graph).toarray()
            
            # Strong connections (training-free threshold)
            threshold = 0.25 * np.max(A)
            strong_connections = A > threshold
            
            # Coarse nodes selection
            coarse_nodes = self._select_coarse_nodes(strong_connections)
            
            # Build coarse graph
            coarse_graph = nx.Graph()
            for i, c_node in enumerate(coarse_nodes):
                coarse_graph.add_node(i, fine_nodes=[c_node])
            
            return coarse_graph
        
        return graph
    
    def _select_coarse_nodes(self, connections: np.ndarray) -> List[int]:
        """Select coarse nodes using greedy algorithm"""
        n = connections.shape[0]
        selected = []
        available = set(range(n))
        
        while available:
            # Select node with most connections
            node = max(available, key=lambda x: np.sum(connections[x]))
            selected.append(node)
            available.remove(node)
            
            # Remove strongly connected neighbors
            neighbors = np.where(connections[node])[0]
            available -= set(neighbors)
        
        return selected


# ============================================================================
# COMPONENT 4: PROBABILISTIC GRAPH MATCHING WITH BELIEF PROPAGATION (PGM-BP)
# ============================================================================

class ProbabilisticGraphMatcher:
    """
    Probabilistic Graph Matching using Belief Propagation
    Training-free probabilistic inference for graph matching
    """
    
    def __init__(self, max_iterations: int = 50, convergence_threshold: float = 1e-4):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.node_beliefs = {}
        self.edge_beliefs = {}
        self.messages = {}
        
    def match_with_belief_propagation(self,
                                     scene_graph: nx.Graph,
                                     goal_graph: nx.Graph) -> Dict:
        """Match graphs using loopy belief propagation"""
        # Initialize beliefs
        self._initialize_beliefs(scene_graph, goal_graph)
        
        # Message passing iterations
        converged = False
        for iteration in range(self.max_iterations):
            old_beliefs = self.node_beliefs.copy()
            
            # Update messages
            self._update_node_to_edge_messages(scene_graph, goal_graph)
            self._update_edge_to_node_messages(scene_graph, goal_graph)
            
            # Update beliefs
            self._update_node_beliefs(scene_graph, goal_graph)
            self._update_edge_beliefs(scene_graph, goal_graph)
            
            # Check convergence
            if self._check_convergence(old_beliefs):
                converged = True
                break
        
        # Extract MAP assignment
        assignment = self._extract_map_assignment()
        
        return {
            'assignment': assignment,
            'node_beliefs': self.node_beliefs,
            'edge_beliefs': self.edge_beliefs,
            'converged': converged,
            'iterations': iteration + 1
        }
    
    def _initialize_beliefs(self, scene_graph: nx.Graph, goal_graph: nx.Graph):
        """Initialize node and edge beliefs"""
        # Node beliefs: probability of scene node matching goal node
        for s_node in scene_graph.nodes():
            self.node_beliefs[s_node] = {}
            for g_node in goal_graph.nodes():
                # Initial similarity based on attributes
                similarity = self._compute_node_similarity(
                    scene_graph.nodes[s_node],
                    goal_graph.nodes[g_node]
                )
                self.node_beliefs[s_node][g_node] = similarity
        
        # Normalize beliefs
        for s_node in self.node_beliefs:
            total = sum(self.node_beliefs[s_node].values())
            if total > 0:
                for g_node in self.node_beliefs[s_node]:
                    self.node_beliefs[s_node][g_node] /= total
        
        # Edge beliefs
        for s_edge in scene_graph.edges():
            self.edge_beliefs[s_edge] = {}
            for g_edge in goal_graph.edges():
                similarity = self._compute_edge_similarity(
                    scene_graph.edges[s_edge],
                    goal_graph.edges[g_edge]
                )
                self.edge_beliefs[s_edge][g_edge] = similarity
    
    def _compute_node_similarity(self, s_node_data: Dict, g_node_data: Dict) -> float:
        """Compute similarity between nodes without training"""
        similarity = 0.0
        
        # Label similarity
        if 'label' in s_node_data and 'label' in g_node_data:
            if s_node_data['label'] == g_node_data['label']:
                similarity += 0.5
        
        # Position similarity (if available)
        if 'position' in s_node_data and 'position' in g_node_data:
            dist = np.linalg.norm(
                np.array(s_node_data['position']) - np.array(g_node_data['position'])
            )
            similarity += np.exp(-dist / 10.0) * 0.3
        
        # Size similarity
        if 'size' in s_node_data and 'size' in g_node_data:
            size_ratio = min(s_node_data['size'], g_node_data['size']) / \
                        max(s_node_data['size'], g_node_data['size'])
            similarity += size_ratio * 0.2
        
        return similarity
    
    def _compute_edge_similarity(self, s_edge_data: Dict, g_edge_data: Dict) -> float:
        """Compute edge similarity"""
        # Simple weight similarity
        s_weight = s_edge_data.get('weight', 1.0)
        g_weight = g_edge_data.get('weight', 1.0)
        
        return np.exp(-abs(s_weight - g_weight))
    
    def _update_node_to_edge_messages(self, scene_graph: nx.Graph, goal_graph: nx.Graph):
        """Update messages from nodes to edges"""
        for s_edge in scene_graph.edges():
            s1, s2 = s_edge
            
            for g_edge in goal_graph.edges():
                g1, g2 = g_edge
                
                # Message from s1 to edge
                msg_s1 = self.node_beliefs[s1].get(g1, 0.0)
                
                # Message from s2 to edge
                msg_s2 = self.node_beliefs[s2].get(g2, 0.0)
                
                # Alternative matching
                msg_s1_alt = self.node_beliefs[s1].get(g2, 0.0)
                msg_s2_alt = self.node_beliefs[s2].get(g1, 0.0)
                
                # Combine messages
                self.messages[(s_edge, g_edge)] = max(
                    msg_s1 * msg_s2,
                    msg_s1_alt * msg_s2_alt
                )
    
    def _update_edge_to_node_messages(self, scene_graph: nx.Graph, goal_graph: nx.Graph):
        """Update messages from edges to nodes"""
        for s_node in scene_graph.nodes():
            # Get incident edges
            incident_edges = list(scene_graph.edges(s_node))
            
            for g_node in goal_graph.nodes():
                # Aggregate messages from incident edges
                total_message = 1.0
                
                for s_edge in incident_edges:
                    edge_message = 0.0
                    
                    for g_edge in goal_graph.edges(g_node):
                        if (s_edge, g_edge) in self.messages:
                            edge_message += self.messages[(s_edge, g_edge)]
                    
                    total_message *= (1 + edge_message)
                
                # Store message
                self.messages[(s_node, g_node)] = total_message
    
    def _update_node_beliefs(self, scene_graph: nx.Graph, goal_graph: nx.Graph):
        """Update node beliefs based on messages"""
        for s_node in scene_graph.nodes():
            for g_node in goal_graph.nodes():
                # Prior belief
                prior = self._compute_node_similarity(
                    scene_graph.nodes[s_node],
                    goal_graph.nodes[g_node]
                )
                
                # Incoming messages
                if (s_node, g_node) in self.messages:
                    message = self.messages[(s_node, g_node)]
                else:
                    message = 1.0
                
                # Update belief
                self.node_beliefs[s_node][g_node] = prior * message
        
        # Normalize
        for s_node in self.node_beliefs:
            total = sum(self.node_beliefs[s_node].values())
            if total > 0:
                for g_node in self.node_beliefs[s_node]:
                    self.node_beliefs[s_node][g_node] /= total
    
    def _update_edge_beliefs(self, scene_graph: nx.Graph, goal_graph: nx.Graph):
        """Update edge beliefs"""
        for s_edge in scene_graph.edges():
            for g_edge in goal_graph.edges():
                if (s_edge, g_edge) in self.messages:
                    self.edge_beliefs[s_edge][g_edge] = self.messages[(s_edge, g_edge)]
    
    def _check_convergence(self, old_beliefs: Dict) -> bool:
        """Check if beliefs have converged"""
        total_change = 0.0
        
        for s_node in self.node_beliefs:
            for g_node in self.node_beliefs[s_node]:
                old_val = old_beliefs.get(s_node, {}).get(g_node, 0.0)
                new_val = self.node_beliefs[s_node][g_node]
                total_change += abs(new_val - old_val)
        
        return total_change < self.convergence_threshold
    
    def _extract_map_assignment(self) -> Dict:
        """Extract maximum a posteriori assignment"""
        assignment = {}
        
        # For each scene node, find best matching goal node
        for s_node in self.node_beliefs:
            if self.node_beliefs[s_node]:
                best_match = max(self.node_beliefs[s_node].items(), 
                               key=lambda x: x[1])
                if best_match[1] > 0.3:  # Confidence threshold
                    assignment[s_node] = best_match[0]
        
        return assignment


# ============================================================================
# COMPONENT 5: GRAPH NEURAL OPERATORS WITHOUT TRAINING (GNO-TF)
# ============================================================================

class TrainingFreeGraphOperator:
    """
    Graph Neural Operators using physics-inspired operators
    No learned parameters - uses mathematical physics operators
    """
    
    def __init__(self):
        self.operators = {
            'diffusion': self.heat_diffusion_operator,
            'wave': self.wave_propagation_operator,
            'schrodinger': self.quantum_walk_operator,
            'advection': self.advection_operator
        }
        
    def apply_operator(self, 
                       graph: nx.Graph,
                       features: np.ndarray,
                       operator_type: str = 'diffusion',
                       time: float = 1.0) -> np.ndarray:
        """Apply graph operator without any trained weights"""
        if operator_type not in self.operators:
            raise ValueError(f"Unknown operator: {operator_type}")
        
        return self.operators[operator_type](graph, features, time)
    
    def heat_diffusion_operator(self, graph: nx.Graph, features: np.ndarray, t: float) -> np.ndarray:
        """Heat equation on graphs - models information diffusion"""
        L = nx.laplacian_matrix(graph).toarray()
        
        # Heat kernel: exp(-tL)
        heat_kernel = expm(-t * L)
        
        # Apply to features
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        diffused = heat_kernel @ features
        
        return diffused
    
    def wave_propagation_operator(self, graph: nx.Graph, features: np.ndarray, t: float) -> np.ndarray:
        """Wave equation for oscillatory patterns"""
        L = nx.laplacian_matrix(graph).toarray()
        
        # Wave operator: cos(sqrt(L) * t)
        eigenvalues, eigenvectors = np.linalg.eig(L)
        
        # Ensure non-negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Compute cos(sqrt(λ) * t) for each eigenvalue
        wave_eigenvalues = np.cos(np.sqrt(eigenvalues) * t)
        
        # Reconstruct operator
        wave_operator = eigenvectors @ np.diag(wave_eigenvalues) @ eigenvectors.T
        
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        return np.real(wave_operator @ features)
    
    def quantum_walk_operator(self, graph: nx.Graph, features: np.ndarray, t: float) -> np.ndarray:
        """Quantum walk for exploration - complex-valued"""
        L = nx.laplacian_matrix(graph).toarray()
        
        # Hamiltonian from Laplacian
        H = L  # Simple choice
        
        # Quantum evolution: exp(-iHt)
        evolution = expm(-1j * H * t)
        
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        # Apply and take magnitude
        quantum_features = evolution @ features.astype(complex)
        
        return np.abs(quantum_features)
    
    def advection_operator(self, graph: nx.Graph, features: np.ndarray, t: float) -> np.ndarray:
        """Advection operator for directional flow"""
        A = nx.adjacency_matrix(graph).toarray()
        
        # Normalize for stability
        row_sums = A.sum(axis=1)
        row_sums[row_sums == 0] = 1
        A_normalized = A / row_sums[:, np.newaxis]
        
        # Advection: (I + tA)
        advection = np.eye(len(graph)) + t * A_normalized
        
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        return advection @ features
    
    def combine_operators(self,
                         graph: nx.Graph,
                         features: np.ndarray,
                         operators: List[str],
                         weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine multiple operators"""
        if weights is None:
            weights = [1.0 / len(operators)] * len(operators)
        
        combined = np.zeros_like(features)
        
        for op, weight in zip(operators, weights):
            result = self.apply_operator(graph, features, op)
            combined += weight * result
        
        return combined


# ============================================================================
# COMPONENT 6: TOPOLOGICAL SIGNATURES FOR LOOP CLOSURE (Already implemented)
# ============================================================================
# [Previous TSLC implementation remains the same]


# ============================================================================
# INTEGRATED NAVIGATION SYSTEM
# ============================================================================

class EnhancedGraphNavigationSystem:
    """
    Complete integration of all 6 novel components for training-free navigation
    """
    
    def __init__(self, args):
        self.args = args
        
        # Initialize all components
        self.temporal_memory = TemporalGraphMemory()
        self.counterfactual_reasoner = CounterfactualGraphReasoner()
        self.hierarchical_graph = HierarchicalTopologicalGraph()
        self.probabilistic_matcher = ProbabilisticGraphMatcher()
        self.graph_operator = TrainingFreeGraphOperator()
        # TSLC already imported from previous implementation
        
        # State tracking
        self.current_graph = nx.Graph()
        self.trajectory = []
        self.exploration_history = []
        
    def process_observation(self, observation: np.ndarray, semantic_map: np.ndarray) -> Dict:
        """Process new observation through all components"""
        
        # 1. Build hierarchical representation
        self.hierarchical_graph.build_hierarchy(observation, semantic_map)
        
        # 2. Update temporal memory
        current_graph_dict = self._graph_to_dict(self.current_graph)
        self.temporal_memory.add_snapshot(current_graph_dict)
        
        # 3. Apply graph operators for feature propagation
        if len(self.current_graph) > 0:
            features = self._extract_node_features(self.current_graph)
            
            # Diffusion for smoothing
            diffused = self.graph_operator.apply_operator(
                self.current_graph, features, 'diffusion', t=0.5
            )
            
            # Wave for pattern detection
            wave = self.graph_operator.apply_operator(
                self.current_graph, features, 'wave', t=1.0
            )
            
            # Combine
            enhanced_features = 0.7 * diffused + 0.3 * wave
            self._update_node_features(self.current_graph, enhanced_features)
        
        return {
            'hierarchical_graph': self.hierarchical_graph,
            'temporal_context': self.temporal_memory.graph_snapshots[-5:],
            'enhanced_features': enhanced_features if len(self.current_graph) > 0 else None
        }
    
    def match_with_goal(self, goal_graph: nx.Graph) -> Dict:
        """Match current scene with goal using probabilistic matching"""
        
        # Get appropriate level from hierarchy
        scene_graph = self.hierarchical_graph.query_at_level(HierarchicalLevel.OBJECT)
        
        if scene_graph is None:
            scene_graph = self.current_graph
        
        # Probabilistic matching
        match_result = self.probabilistic_matcher.match_with_belief_propagation(
            scene_graph, goal_graph
        )
        
        return match_result
    
    def generate_exploration_hypotheses(self, goal_graph: nx.Graph) -> List[Dict]:
        """Generate counterfactual hypotheses for exploration"""
        
        # Get frontier nodes
        frontier_nodes = self._identify_frontiers()
        
        # Generate counterfactuals
        scene_dict = self._graph_to_dict(self.current_graph)
        goal_dict = self._graph_to_dict(goal_graph)
        
        counterfactuals = self.counterfactual_reasoner.generate_counterfactuals(
            scene_dict, goal_dict, frontier_nodes
        )
        
        # Predict future state
        future_prediction = self.temporal_memory.predict_future_state(steps_ahead=5)
        
        return {
            'counterfactuals': counterfactuals,
            'future_prediction': future_prediction,
            'exploration_targets': self._rank_exploration_targets(counterfactuals)
        }
    
    def update_trajectory(self, pose: Tuple[float, float, float]):
        """Update trajectory for loop closure detection"""
        self.trajectory.append(pose)
        
        # Check for loops using topological signatures
        # (Using TSLC implementation from before)
    
    def _graph_to_dict(self, graph: nx.Graph) -> Dict:
        """Convert networkx graph to dictionary"""
        return {
            'nodes': [{'label': n, **graph.nodes[n]} for n in graph.nodes()],
            'edges': [{'source': e[0], 'target': e[1], **graph.edges[e]} 
                     for e in graph.edges()]
        }
    
    def _identify_frontiers(self) -> List[Dict]:
        """Identify frontier nodes for exploration"""
        frontiers = []
        
        for node in self.current_graph.nodes():
            # Simple heuristic: nodes with few connections are frontiers
            if self.current_graph.degree(node) < 3:
                frontiers.append({
                    'label': node,
                    'position': self.current_graph.nodes[node].get('position', [0, 0, 0])
                })
        
        return frontiers
    
    def _extract_node_features(self, graph: nx.Graph) -> np.ndarray:
        """Extract feature vector for each node"""
        n_nodes = len(graph)
        features = np.zeros((n_nodes, 1))  # Simple 1D features
        
        for i, node in enumerate(graph.nodes()):
            # Use degree as simple feature
            features[i] = graph.degree(node)
        
        return features
    
    def _update_node_features(self, graph: nx.Graph, features: np.ndarray):
        """Update node features in graph"""
        for i, node in enumerate(graph.nodes()):
            if i < len(features):
                graph.nodes[node]['enhanced_feature'] = features[i]
    
    def _rank_exploration_targets(self, counterfactuals: List[Dict]) -> List[Dict]:
        """Rank exploration targets based on counterfactual confidence"""
        targets = []
        
        for cf in counterfactuals:
            if cf['confidence'] > 0.3:
                for node in cf.get('predicted_nodes', []):
                    if 'position' in node:
                        targets.append({
                            'position': node['position'],
                            'confidence': cf['confidence'],
                            'hypothesis': cf['hypothesis']
                        })
        
        # Sort by confidence
        targets.sort(key=lambda x: x['confidence'], reverse=True)
        
        return targets[:5]  # Top 5 targets