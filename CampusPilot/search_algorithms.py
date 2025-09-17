"""
Search Algorithms for Campus Navigation

Implements BFS, DFS, UCS (Uniform Cost Search), and A* algorithms
with proper distance calculations and performance metrics.
"""

import heapq
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import math


def calculate_walking_time(distance_m: float, speed_ms: float = 1.4) -> float:
    """
    Calculate walking time in seconds based on distance and walking speed.
    
    Args:
        distance_m: Distance in meters
        speed_ms: Walking speed in m/s (default: 1.4 m/s)
    
    Returns:
        Walking time in seconds
    """
    return distance_m / speed_ms


def calculate_path_distance(path: List[str], graph: Dict[str, Dict[str, Dict[str, Any]]]) -> float:
    """
    Calculate total distance for a path by summing edge weights.
    
    Args:
        path: List of node names representing the path
        graph: Graph adjacency list
    
    Returns:
        Total distance in meters
    """
    if len(path) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        if next_node in graph.get(current_node, {}):
            total_distance += graph[current_node][next_node]['weight']
        else:
            # If direct edge doesn't exist, return infinite distance
            return float('inf')
    
    return total_distance


def heuristic_distance(node: str, goal: str, coords: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculate heuristic distance (straight-line) between two nodes.
    
    Args:
        node: Current node
        goal: Goal node
        coords: Dictionary mapping nodes to (lat, lon) coordinates
    
    Returns:
        Euclidean distance in meters (approximation)
    """
    if node not in coords or goal not in coords:
        return 0.0
    
    lat1, lon1 = coords[node]
    lat2, lon2 = coords[goal]
    
    # Convert to approximate meters using simple formula
    # This is a rough approximation suitable for small distances
    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1
    
    # Approximate conversion: 1 degree ≈ 111,000 meters
    lat_meters = lat_diff * 111000
    lon_meters = lon_diff * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
    
    return math.sqrt(lat_meters**2 + lon_meters**2)


def bfs(graph: Dict[str, Dict[str, Dict[str, Any]]], start: str, goal: str, 
        coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Breadth-First Search implementation.
    
    Args:
        graph: Graph adjacency list
        start: Starting node
        goal: Goal node
        coords: Node coordinates
    
    Returns:
        Dictionary with path, distance, time, and exploration stats
    """
    if start not in graph or goal not in graph:
        return {
            'path': [],
            'distance_m': float('inf'),
            'walking_time_s': float('inf'),
            'nodes_explored': [],
            'nodes_explored_count': 0
        }
    
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    nodes_explored = [start]
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            
            # Calculate metrics
            distance = calculate_path_distance(path, graph)
            walking_time = calculate_walking_time(distance)
            
            return {
                'path': path,
                'distance_m': distance,
                'walking_time_s': walking_time,
                'nodes_explored': nodes_explored,
                'nodes_explored_count': len(nodes_explored)
            }
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                nodes_explored.append(neighbor)
    
    # No path found
    return {
        'path': [],
        'distance_m': float('inf'),
        'walking_time_s': float('inf'),
        'nodes_explored': nodes_explored,
        'nodes_explored_count': len(nodes_explored)
    }


def dfs(graph: Dict[str, Dict[str, Dict[str, Any]]], start: str, goal: str,
        coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Depth-First Search implementation.
    
    Args:
        graph: Graph adjacency list
        start: Starting node
        goal: Goal node
        coords: Node coordinates
    
    Returns:
        Dictionary with path, distance, time, and exploration stats
    """
    if start not in graph or goal not in graph:
        return {
            'path': [],
            'distance_m': float('inf'),
            'walking_time_s': float('inf'),
            'nodes_explored': [],
            'nodes_explored_count': 0
        }
    
    stack = [start]
    visited = set([start])
    parent = {start: None}
    nodes_explored = [start]
    
    while stack:
        current = stack.pop()
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            
            # Calculate metrics
            distance = calculate_path_distance(path, graph)
            walking_time = calculate_walking_time(distance)
            
            return {
                'path': path,
                'distance_m': distance,
                'walking_time_s': walking_time,
                'nodes_explored': nodes_explored,
                'nodes_explored_count': len(nodes_explored)
            }
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
                nodes_explored.append(neighbor)
    
    # No path found
    return {
        'path': [],
        'distance_m': float('inf'),
        'walking_time_s': float('inf'),
        'nodes_explored': nodes_explored,
        'nodes_explored_count': len(nodes_explored)
    }


def uniform_cost_search(graph: Dict[str, Dict[str, Dict[str, Any]]], start: str, goal: str,
                       coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Uniform Cost Search (Dijkstra's) implementation.
    
    Args:
        graph: Graph adjacency list
        start: Starting node
        goal: Goal node
        coords: Node coordinates
    
    Returns:
        Dictionary with path, distance, time, and exploration stats
    """
    if start not in graph or goal not in graph:
        return {
            'path': [],
            'distance_m': float('inf'),
            'walking_time_s': float('inf'),
            'nodes_explored': [],
            'nodes_explored_count': 0
        }
    
    # Priority queue: (cost, node)
    pq = [(0, start)]
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    nodes_explored = []
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        nodes_explored.append(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            
            # Calculate metrics
            distance = calculate_path_distance(path, graph)
            walking_time = calculate_walking_time(distance)
            
            return {
                'path': path,
                'distance_m': distance,
                'walking_time_s': walking_time,
                'nodes_explored': nodes_explored,
                'nodes_explored_count': len(nodes_explored)
            }
        
        for neighbor in graph[current]:
            edge_weight = graph[current][neighbor]['weight']
            new_cost = current_cost + edge_weight
            
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
    
    # No path found
    return {
        'path': [],
        'distance_m': float('inf'),
        'walking_time_s': float('inf'),
        'nodes_explored': nodes_explored,
        'nodes_explored_count': len(nodes_explored)
    }


def a_star(graph: Dict[str, Dict[str, Dict[str, Any]]], start: str, goal: str,
           coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    A* search implementation.
    
    Args:
        graph: Graph adjacency list
        start: Starting node
        goal: Goal node
        coords: Node coordinates
    
    Returns:
        Dictionary with path, distance, time, and exploration stats
    """
    if start not in graph or goal not in graph:
        return {
            'path': [],
            'distance_m': float('inf'),
            'walking_time_s': float('inf'),
            'nodes_explored': [],
            'nodes_explored_count': 0
        }
    
    # Priority queue: (f_score, node)
    pq = [(heuristic_distance(start, goal, coords), start)]
    visited = set()
    parent = {start: None}
    g_score = {start: 0}
    nodes_explored = []
    
    while pq:
        current_f, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        nodes_explored.append(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            
            # Calculate metrics
            distance = calculate_path_distance(path, graph)
            walking_time = calculate_walking_time(distance)
            
            return {
                'path': path,
                'distance_m': distance,
                'walking_time_s': walking_time,
                'nodes_explored': nodes_explored,
                'nodes_explored_count': len(nodes_explored)
            }
        
        for neighbor in graph[current]:
            edge_weight = graph[current][neighbor]['weight']
            tentative_g = g_score[current] + edge_weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_distance(neighbor, goal, coords)
                heapq.heappush(pq, (f_score, neighbor))
    
    # No path found
    return {
        'path': [],
        'distance_m': float('inf'),
        'walking_time_s': float('inf'),
        'nodes_explored': nodes_explored,
        'nodes_explored_count': len(nodes_explored)
    }


def get_k_shortest_paths(graph: Dict[str, Dict[str, Dict[str, Any]]], start: str, goal: str,
                        coords: Dict[str, Tuple[float, float]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Get k shortest paths between start and goal using UCS as base algorithm.
    
    Args:
        graph: Graph adjacency list
        start: Starting node
        goal: Goal node
        coords: Node coordinates
        k: Number of paths to find
    
    Returns:
        List of path dictionaries sorted by distance
    """
    try:
        import networkx as nx
        
        # Convert our graph to NetworkX graph
        G = nx.Graph()
        for node, neighbors in graph.items():
            for neighbor, data in neighbors.items():
                G.add_edge(node, neighbor, weight=data['weight'])
        
        # Get k shortest paths
        try:
            paths_generator = nx.shortest_simple_paths(G, start, goal, weight='weight')
            k_paths = []
            
            for i, path in enumerate(paths_generator):
                if i >= k:
                    break
                
                distance = calculate_path_distance(path, graph)
                walking_time = calculate_walking_time(distance)
                
                k_paths.append({
                    'path': path,
                    'distance_m': distance,
                    'walking_time_s': walking_time,
                    'nodes_explored': path,  # For k-shortest, explored nodes = path
                    'nodes_explored_count': len(path)
                })
            
            return k_paths
            
        except nx.NetworkXNoPath:
            return []
        
    except ImportError:
        # Fallback: just return the single UCS result
        ucs_result = uniform_cost_search(graph, start, goal, coords)
        return [ucs_result] if ucs_result['path'] else []


# Algorithm registry for easy access
ALGORITHMS = {
    'bfs': bfs,
    'dfs': dfs,
    'ucs': uniform_cost_search,
    'a_star': a_star
}


def run_algorithm(algorithm_name: str, graph: Dict[str, Dict[str, Dict[str, Any]]], 
                 start: str, goal: str, coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Run a specific algorithm by name.
    
    Args:
        algorithm_name: Name of algorithm ('bfs', 'dfs', 'ucs', 'a_star')
        graph: Graph adjacency list
        start: Starting node
        goal: Goal node
        coords: Node coordinates
    
    Returns:
        Algorithm result dictionary
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return ALGORITHMS[algorithm_name](graph, start, goal, coords)


if __name__ == "__main__":
    # Test with a simple graph
    test_graph = {
        'A': {'B': {'weight': 100}, 'C': {'weight': 150}},
        'B': {'A': {'weight': 100}, 'D': {'weight': 200}},
        'C': {'A': {'weight': 150}, 'D': {'weight': 100}},
        'D': {'B': {'weight': 200}, 'C': {'weight': 100}}
    }
    
    test_coords = {
        'A': (0, 0),
        'B': (0, 1),
        'C': (1, 0),
        'D': (1, 1)
    }
    
    print("Testing search algorithms...")
    for alg_name in ALGORITHMS:
        result = run_algorithm(alg_name, test_graph, 'A', 'D', test_coords)
        print(f"{alg_name.upper()}: Path = {result['path']}, Distance = {result['distance_m']:.1f}m")