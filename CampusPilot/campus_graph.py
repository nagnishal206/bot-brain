"""
Campus Graph Builder and KML Parser

This module provides functionality to:
1. Parse KML files to extract campus landmarks and paths
2. Build graph structures with accurate distance calculations using Haversine formula
3. Provide fallback graph with all required campus locations
"""

import os
from typing import Dict, Tuple, Any, Optional
from geopy.distance import geodesic
import networkx as nx
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance in meters between two points on Earth
    using the geodesic method from geopy which is more accurate than simple Haversine.
    
    Args:
        coord1: (latitude, longitude) of first point
        coord2: (latitude, longitude) of second point
    
    Returns:
        Distance in meters
    """
    return geodesic(coord1, coord2).meters


def parse_kml_file(kml_path: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """
    Parse KML file to extract Points (landmarks) and LineStrings (paths).
    
    Args:
        kml_path: Path to the KML file
        
    Returns:
        Tuple of (coordinates_dict, graph_data)
        coordinates_dict maps node names to (lat, lon)
        graph_data contains extracted path information
    """
    coords = {}
    paths = []
    
    try:
        # Parse KML using XML parser
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # Define KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Find all placemarks
        placemarks = root.findall('.//kml:Placemark', ns)
        
        for placemark in placemarks:
            name_elem = placemark.find('kml:name', ns)
            if name_elem is not None:
                name = name_elem.text
                
                # Look for Point coordinates
                point = placemark.find('.//kml:Point/kml:coordinates', ns)
                if point is not None:
                    coords_text = point.text.strip()
                    if coords_text:
                        # Parse coordinates (format: lon,lat,alt)
                        parts = coords_text.split(',')
                        if len(parts) >= 2:
                            lon, lat = float(parts[0]), float(parts[1])
                            coords[name] = (lat, lon)
                
                # Look for LineString coordinates (paths)
                linestring = placemark.find('.//kml:LineString/kml:coordinates', ns)
                if linestring is not None:
                    coords_text = linestring.text.strip()
                    if coords_text:
                        # Parse path coordinates
                        path_coords = []
                        for coord_set in coords_text.split():
                            parts = coord_set.split(',')
                            if len(parts) >= 2:
                                lon, lat = float(parts[0]), float(parts[1])
                                path_coords.append((lat, lon))
                        if path_coords:
                            paths.append({
                                'name': name,
                                'coords': path_coords
                            })
        
        print(f"Successfully parsed KML file: {len(coords)} points, {len(paths)} paths")
        return coords, {'paths': paths}
        
    except Exception as e:
        print(f"Error parsing KML file: {e}")
        return {}, {'paths': []}


def create_fallback_graph() -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Tuple[float, float]]]:
    """
    Create a fallback graph with all required campus landmarks.
    Uses approximate coordinates and distances for demonstration purposes.
    
    Returns:
        Tuple of (graph, coordinates)
        graph: adjacency list representation
        coordinates: mapping of node names to (lat, lon)
    """
    # Approximate coordinates for Chanakya University (using realistic coordinates for the region)
    # Note: These are fallback coordinates and may not reflect actual positions
    fallback_coords = {
        'Entry Gate': (13.2201, 77.7540),
        'Exit Gate': (13.2201, 77.7550),
        'Main Gate': (13.2195, 77.7545),
        'Admin Block': (13.2210, 77.7548),
        'Faculty Block': (13.2215, 77.7552),
        'Library': (13.2220, 77.7545),
        'Hostel A': (13.2245, 77.7590),
        'Hostel B': (13.2240, 77.7585),
        'Food Court 1': (13.2251, 77.7573),
        'Food Court 2': (13.2255, 77.7575),
        'Food Court': (13.2253, 77.7574),  # General food court
        'Sports Complex': (13.2285, 77.7571),
        'Cricket Ground': (13.2289, 77.7571),
        'Volleyball Court': (13.2287, 77.7576),
        'Tennis Court': (13.2285, 77.7574),
        'Football Ground': (13.2281, 77.7564),
        'Auditorium': (13.2225, 77.7555),
        'Engineering Block': (13.2230, 77.7560),
        'Management Block': (13.2235, 77.7565),
        'Biometric Entry': (13.2205, 77.7542),
        'Flag Post': (13.2217, 77.7549),
        'Check Post 1': (13.2213, 77.7551),
        'Check Post 2': (13.2208, 77.7547),
        'Acad 1': (13.2222, 77.7554),
        'Acad 2': (13.2224, 77.7559),
        'Bridge': (13.2250, 77.7580)
    }
    
    # Create connections between related nodes to ensure the graph is connected
    connections = [
        # Main entrance area
        ('Entry Gate', 'Exit Gate'),
        ('Entry Gate', 'Main Gate'),
        ('Exit Gate', 'Main Gate'),
        ('Main Gate', 'Biometric Entry'),
        ('Biometric Entry', 'Check Post 1'),
        ('Check Post 1', 'Check Post 2'),
        ('Check Post 2', 'Flag Post'),
        
        # Academic area
        ('Main Gate', 'Admin Block'),
        ('Admin Block', 'Faculty Block'),
        ('Faculty Block', 'Library'),
        ('Library', 'Auditorium'),
        ('Auditorium', 'Engineering Block'),
        ('Engineering Block', 'Management Block'),
        ('Flag Post', 'Acad 1'),
        ('Acad 1', 'Acad 2'),
        ('Acad 2', 'Library'),
        
        # Hostel area
        ('Management Block', 'Bridge'),
        ('Bridge', 'Hostel A'),
        ('Bridge', 'Hostel B'),
        ('Hostel A', 'Hostel B'),
        ('Hostel A', 'Food Court 1'),
        ('Hostel B', 'Food Court 2'),
        ('Food Court 1', 'Food Court 2'),
        ('Food Court 1', 'Food Court'),
        ('Food Court 2', 'Food Court'),
        
        # Sports area
        ('Food Court', 'Sports Complex'),
        ('Sports Complex', 'Cricket Ground'),
        ('Cricket Ground', 'Football Ground'),
        ('Football Ground', 'Tennis Court'),
        ('Tennis Court', 'Volleyball Court'),
        ('Volleyball Court', 'Sports Complex'),
        
        # Additional connections for better connectivity
        ('Library', 'Bridge'),
        ('Engineering Block', 'Sports Complex'),
        ('Faculty Block', 'Acad 1'),
    ]
    
    # Build the graph with calculated distances
    graph = {}
    for node in fallback_coords:
        graph[node] = {}
    
    for node1, node2 in connections:
        if node1 in fallback_coords and node2 in fallback_coords:
            coord1 = fallback_coords[node1]
            coord2 = fallback_coords[node2]
            distance = haversine_distance(coord1, coord2)
            
            # Add bidirectional edges
            graph[node1][node2] = {
                'weight': distance,
                'coords': coord2
            }
            graph[node2][node1] = {
                'weight': distance,
                'coords': coord1
            }
    
    print(f"Created fallback graph with {len(fallback_coords)} nodes and {len(connections)} connections")
    return graph, fallback_coords


def build_graph_from_kml(coords: Dict[str, Tuple[float, float]], 
                        kml_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build a graph from KML extracted data with improved path accuracy.
    
    Args:
        coords: Dictionary mapping node names to (lat, lon)
        kml_data: Additional data extracted from KML (paths, etc.)
        
    Returns:
        Graph as adjacency list
    """
    graph = {}
    
    # Initialize all nodes
    for node in coords:
        graph[node] = {}
    
    # Process paths from KML data if available
    paths = kml_data.get('paths', [])
    connections_made = 0
    
    print(f"Processing {len(paths)} paths from KML data...")
    
    # Use explicit KML paths to create accurate connections
    if paths:
        for path_info in paths:
            path_name = path_info['name']
            path_coords = path_info['coords']
            
            if len(path_coords) < 2:
                continue
                
            print(f"Processing path '{path_name}' with {len(path_coords)} waypoints")
            
            # Find the best matching start and end nodes for this path
            start_coord = path_coords[0]
            end_coord = path_coords[-1]
            
            start_node = find_closest_node(start_coord, coords, max_distance=100)
            end_node = find_closest_node(end_coord, coords, max_distance=100)
            
            if start_node and end_node and start_node != end_node:
                # Calculate total path distance following the actual KML path
                total_distance = 0
                for i in range(len(path_coords) - 1):
                    segment_distance = haversine_distance(path_coords[i], path_coords[i + 1])
                    total_distance += segment_distance
                
                # Create bidirectional connection with accurate distance
                graph[start_node][end_node] = {
                    'weight': total_distance,
                    'coords': coords[end_node],
                    'path_name': path_name
                }
                graph[end_node][start_node] = {
                    'weight': total_distance,
                    'coords': coords[start_node],
                    'path_name': path_name
                }
                connections_made += 1
                print(f"  Connected {start_node} ↔ {end_node} (distance: {total_distance:.1f}m)")
                
                # Also connect intermediate waypoints to nearby nodes if they exist
                for waypoint_coord in path_coords[1:-1]:  # Skip start and end
                    nearby_node = find_closest_node(waypoint_coord, coords, max_distance=50)
                    if nearby_node:
                        # Connect waypoint to path endpoints if not already connected
                        waypoint_to_start = sum(haversine_distance(path_coords[i], path_coords[i+1]) 
                                              for i in range(path_coords.index(waypoint_coord)))
                        waypoint_to_end = total_distance - waypoint_to_start
                        
                        if nearby_node not in graph[start_node]:
                            graph[start_node][nearby_node] = {
                                'weight': waypoint_to_start,
                                'coords': coords[nearby_node],
                                'path_name': path_name + ' (partial)'
                            }
                            graph[nearby_node][start_node] = {
                                'weight': waypoint_to_start,
                                'coords': coords[start_node],
                                'path_name': path_name + ' (partial)'
                            }
                        
                        if nearby_node not in graph[end_node]:
                            graph[end_node][nearby_node] = {
                                'weight': waypoint_to_end,
                                'coords': coords[nearby_node],
                                'path_name': path_name + ' (partial)'
                            }
                            graph[nearby_node][end_node] = {
                                'weight': waypoint_to_end,
                                'coords': coords[start_node],
                                'path_name': path_name + ' (partial)'
                            }
    
    print(f"Created {connections_made} connections from KML paths")
    
    # Add proximity-based connections for remaining unconnected nodes
    nodes = list(coords.keys())
    proximity_connections = 0
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            # Skip if already connected via KML paths
            if node2 in graph[node1]:
                continue
                
            coord1 = coords[node1]
            coord2 = coords[node2]
            distance = haversine_distance(coord1, coord2)
            
            # Connect nodes that are within reasonable walking distance
            # Use shorter distance for KML-based connections since we have more accurate data
            if distance < 300:  # Reduced from 500m for better accuracy
                graph[node1][node2] = {
                    'weight': distance,
                    'coords': coord2,
                    'path_name': 'proximity'
                }
                graph[node2][node1] = {
                    'weight': distance,
                    'coords': coord1,
                    'path_name': 'proximity'
                }
                proximity_connections += 1
    
    print(f"Added {proximity_connections} proximity-based connections")
    return graph


def find_closest_node(target_coord: Tuple[float, float], 
                     coords: Dict[str, Tuple[float, float]],
                     max_distance: float = 100) -> Optional[str]:
    """
    Find the closest node to a target coordinate within max_distance.
    
    Args:
        target_coord: Target (lat, lon) coordinate
        coords: Dictionary of node coordinates
        max_distance: Maximum distance in meters to consider
        
    Returns:
        Closest node name or None if no node within max_distance
    """
    min_distance = float('inf')
    closest_node = None
    
    for node, node_coord in coords.items():
        distance = haversine_distance(target_coord, node_coord)
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            closest_node = node
    
    return closest_node


def find_closest_nodes(coord1: Tuple[float, float], coord2: Tuple[float, float], 
                      coords: Dict[str, Tuple[float, float]]) -> list:
    """
    Find the closest nodes to given coordinates.
    
    Args:
        coord1, coord2: Coordinates to match
        coords: Dictionary of node coordinates
        
    Returns:
        List of closest node names
    """
    def find_closest(target_coord):
        min_distance = float('inf')
        closest_node = None
        for node, node_coord in coords.items():
            distance = haversine_distance(target_coord, node_coord)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        return closest_node
    
    return [find_closest(coord1), find_closest(coord2)]


def load_graph(kml_path: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Tuple[float, float]]]:
    """
    Load graph from KML file or create fallback graph.
    
    Args:
        kml_path: Path to KML file (default: 'campus.kml')
        
    Returns:
        Tuple of (graph, coordinates)
    """
    if kml_path is None:
        kml_path = 'campus.kml'
    
    # Try to load from KML file
    if os.path.exists(kml_path):
        print(f"Loading graph from KML file: {kml_path}")
        coords, kml_data = parse_kml_file(kml_path)
        
        if coords:
            # Build graph from KML data
            graph = build_graph_from_kml(coords, kml_data)
            
            # If the graph is too sparse, enhance with fallback connections
            fallback_graph, fallback_coords = create_fallback_graph()
            
            # Merge with fallback data for any missing nodes
            for node in fallback_coords:
                if node not in coords:
                    coords[node] = fallback_coords[node]
                    graph[node] = fallback_graph.get(node, {})
            
            # Ensure all nodes from KML are connected
            ensure_connectivity(graph, coords)
            
            return graph, coords
    
    # Fallback to default graph
    print("KML file not found or empty, using fallback graph")
    return create_fallback_graph()


def ensure_connectivity(graph: Dict[str, Dict[str, Dict[str, Any]]], 
                       coords: Dict[str, Tuple[float, float]]) -> None:
    """
    Ensure the graph is connected by adding edges between isolated components.
    
    Args:
        graph: Graph to modify
        coords: Node coordinates
    """
    # Use NetworkX to check connectivity
    nx_graph = nx.Graph()
    
    for node in graph:
        nx_graph.add_node(node)
    
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            nx_graph.add_edge(node, neighbor)
    
    # If not connected, connect components
    if not nx.is_connected(nx_graph):
        components = list(nx.connected_components(nx_graph))
        print(f"Graph has {len(components)} disconnected components, connecting them...")
        
        # Connect components by finding closest nodes between them
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            min_distance = float('inf')
            best_pair = None
            
            for node1 in comp1:
                for node2 in comp2:
                    if node1 in coords and node2 in coords:
                        distance = haversine_distance(coords[node1], coords[node2])
                        if distance < min_distance:
                            min_distance = distance
                            best_pair = (node1, node2)
            
            if best_pair:
                node1, node2 = best_pair
                graph[node1][node2] = {
                    'weight': min_distance,
                    'coords': coords[node2]
                }
                graph[node2][node1] = {
                    'weight': min_distance,
                    'coords': coords[node1]
                }


if __name__ == "__main__":
    # Test the graph loading functionality
    graph, coords = load_graph()
    print(f"Loaded graph with {len(graph)} nodes")
    print("Available nodes:", sorted(list(graph.keys())))
    
    # Print some sample connections
    print("\nSample connections:")
    for node in list(graph.keys())[:5]:
        neighbors = list(graph[node].keys())
        if neighbors:
            print(f"{node}: {neighbors[:3]}" + ("..." if len(neighbors) > 3 else ""))