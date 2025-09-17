# BotBrain - Campus Navigation Assistant

## Overview

BotBrain is a comprehensive campus navigation assistant for Chanakya University that provides intelligent pathfinding between campus locations. The system offers both a command-line interface and a web application, implementing multiple search algorithms (BFS, DFS, Uniform Cost Search, and A*) to find optimal routes between campus landmarks.

The application can parse KML files containing campus geography data or fall back to a predefined campus graph with essential locations like hostels, academic buildings, sports facilities, and dining areas. It calculates accurate walking distances using the Haversine formula and provides walking time estimates, interactive maps, and performance comparisons between different pathfinding algorithms.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components

**Graph Data Structure**: The system uses an adjacency list representation where each node (campus location) connects to neighboring nodes with weighted edges representing walking distances in meters. Coordinates are stored as latitude/longitude pairs for accurate distance calculations and map visualization.

**Multi-Algorithm Search Engine**: Implements four distinct pathfinding algorithms:
- BFS (Breadth-First Search) for shortest path by number of edges
- DFS (Depth-First Search) for path exploration
- Uniform Cost Search (Dijkstra-style) for shortest distance-based paths
- A* with Euclidean/Haversine heuristic for optimal pathfinding

**Dual Interface Architecture**: 
- CLI component (botbrain.py) for command-line queries and algorithm comparisons
- Web application (webapp.py) using Flask for interactive navigation with visual maps

**Data Processing Pipeline**: KML file parsing using FastKML and Shapely for extracting geographic points and paths, with fallback to hardcoded campus graph when KML data is unavailable.

**Visualization Layer**: Dual visualization approach using matplotlib for static map generation and Folium for interactive web maps with route highlighting and alternative path display.

### Design Patterns

**Strategy Pattern**: Multiple search algorithms implement a common interface, allowing runtime algorithm selection and easy addition of new pathfinding methods.

**Factory Pattern**: Graph loading mechanism abstracts KML parsing vs. fallback graph creation, providing consistent graph structure regardless of data source.

**Template Pattern**: Search algorithms follow consistent result formatting with standardized metrics (path, distance, walking time, nodes explored).

### Performance Considerations

**Distance Calculation**: Uses geopy's geodesic method for accurate distance calculations between coordinates, falling back to Haversine formula when needed.

**Graph Connectivity**: Ensures full connectivity in fallback mode by connecting all nodes with reasonable distance estimates (100-400m) for demonstration purposes.

**Memory Efficiency**: Stores only essential edge weights and coordinates, avoiding redundant graph representations.

## External Dependencies

**Core Libraries**:
- NetworkX: Graph algorithms and data structures
- GeoPy: Accurate geodesic distance calculations
- Flask: Web application framework
- FastKML: KML file parsing for geographic data
- Shapely: Geometric operations and spatial data handling

**Visualization**:
- Matplotlib: Static map generation and path plotting
- Folium: Interactive web maps with route visualization
- Pandas: Data processing and CSV export for algorithm comparisons

**Utility Libraries**:
- XML ElementTree: Fallback KML parsing
- Heapq: Priority queue implementation for UCS and A*
- Collections: Deque for BFS implementation

**File Format Support**:
- KML files for campus geography (Points and LineStrings)
- CSV export for algorithm performance comparisons
- PNG output for static map visualizations

**Web Technologies**:
- Jinja2 templates for HTML rendering
- CSS for responsive web interface styling
- JavaScript integration through Folium for interactive maps