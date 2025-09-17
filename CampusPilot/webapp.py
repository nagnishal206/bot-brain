"""
Flask Web Application for BotBrain Campus Navigation Assistant

Provides web interface for campus navigation with interactive maps.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
from campus_graph import load_graph
from search_algorithms import run_algorithm, get_k_shortest_paths, ALGORITHMS
from visualizer import create_folium_map, create_comparison_visualization
import os

app = Flask(__name__)

# Load campus graph on startup
print("Loading campus graph...")
graph, coords = load_graph()
print(f"Graph loaded with {len(graph)} nodes")

# Available nodes for dropdown
available_nodes = sorted(list(graph.keys()))


@app.route('/')
def index():
    """Render the main page with navigation form."""
    return render_template('index.html', nodes=available_nodes)


@app.route('/search', methods=['POST'])
def search_route():
    """Handle route search requests."""
    try:
        # Get form data
        start_node = request.form.get('start')
        goal_node = request.form.get('goal')
        k_routes = 5  # Automatically show up to 5 alternative routes
        
        # Validate inputs
        if not start_node or not goal_node:
            return render_template('results.html', error="Please select both start and destination")
        
        if start_node not in graph or goal_node not in graph:
            return render_template('results.html', error="Invalid start or destination node")
        
        # Run all algorithms for comparison - no single algorithm selection needed
        all_results = {}
        for alg_name in ALGORITHMS:
            all_results[alg_name] = run_algorithm(alg_name, graph, start_node, goal_node, coords)
        
        # Find the best result (shortest distance)
        valid_results = {name: res for name, res in all_results.items() if res['path']}
        if valid_results:
            best_alg = min(valid_results.keys(), key=lambda x: valid_results[x]['distance_m'])
            main_result = valid_results[best_alg]
        else:
            main_result = {'path': [], 'distance_m': float('inf'), 'walking_time_s': float('inf'), 'nodes_explored_count': 0}
        
        # Get k shortest paths for alternatives
        k_paths = get_k_shortest_paths(graph, start_node, goal_node, coords, k_routes)
        
        # Use all_results as comparison_results (we already calculated them)
        comparison_results = all_results
        
        # Create interactive map
        if k_paths:
            map_obj = create_folium_map(k_paths, coords)
            map_html = map_obj._repr_html_()
        else:
            map_html = "<p>No routes found</p>"
        
        # Prepare results for template
        return render_template('results.html',
                             main_result=main_result,
                             k_paths=k_paths,
                             comparison_results=comparison_results,
                             best_algorithm=best_alg if valid_results else 'None',
                             start_node=start_node,
                             goal_node=goal_node,
                             map_html=map_html)
        
    except Exception as e:
        return render_template('results.html', error=f"Error processing request: {str(e)}")


@app.route('/download_csv')
def download_csv():
    """Download comparison results as CSV."""
    try:
        start_node = request.args.get('start')
        goal_node = request.args.get('goal')
        
        if not start_node or not goal_node:
            return "Missing parameters", 400
        
        # Run all algorithms
        results = []
        for alg_name in ALGORITHMS:
            result = run_algorithm(alg_name, graph, start_node, goal_node, coords)
            results.append({
                'query': f"{start_node} → {goal_node}",
                'algorithm': alg_name.upper(),
                'distance_m': result['distance_m'],
                'walking_time_s': result['walking_time_s'],
                'nodes_explored_count': result['nodes_explored_count'],
                'path': ' → '.join(result['path'])
            })
        
        # Create DataFrame and CSV
        df = pd.DataFrame(results)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Create response
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'navigation_comparison_{start_node}_{goal_node}.csv'
        )
        
    except Exception as e:
        return f"Error generating CSV: {str(e)}", 500


@app.route('/api/nodes')
def api_nodes():
    """API endpoint to get available nodes."""
    return jsonify(available_nodes)


@app.route('/api/search')
def api_search():
    """API endpoint for route search."""
    try:
        start_node = request.args.get('start')
        goal_node = request.args.get('goal')
        algorithm = request.args.get('algorithm', 'a_star')
        
        if not start_node or not goal_node:
            return jsonify({'error': 'Missing start or goal parameter'}), 400
        
        if algorithm not in ALGORITHMS:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        result = run_algorithm(algorithm, graph, start_node, goal_node, coords)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting BotBrain Flask application...")
    print(f"Available algorithms: {list(ALGORITHMS.keys())}")
    print(f"Available nodes: {len(available_nodes)}")
    
    # Configure Flask to listen on all hosts for Replit
    app.run(host='0.0.0.0', port=5000, debug=True)