#!/usr/bin/env python3
"""
BotBrain CLI - Campus Navigation Assistant

Command-line interface for campus navigation queries.
"""

import argparse
import sys
from campus_graph import load_graph
from search_algorithms import run_algorithm, get_k_shortest_paths, ALGORITHMS
from visualizer import save_static_map


def format_time(seconds):
    """Format time in seconds to readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"


def print_result(result, algorithm_name):
    """Print algorithm result in a formatted way."""
    print(f"\n=== {algorithm_name.upper()} Results ===")
    
    if not result['path']:
        print("❌ No path found!")
        return
    
    print(f"✅ Path found!")
    print(f"📍 Route: {' → '.join(result['path'])}")
    print(f"📏 Distance: {result['distance_m']:.1f} meters")
    print(f"⏱️  Walking time: {format_time(result['walking_time_s'])}")
    print(f"🔍 Nodes explored: {result['nodes_explored_count']}")


def compare_algorithms(graph, coords, start, goal):
    """Run all algorithms and compare results."""
    print(f"\n🧠 BotBrain - Comparing all algorithms for: {start} → {goal}")
    print("=" * 60)
    
    results = {}
    
    for alg_name in ALGORITHMS:
        print(f"Running {alg_name.upper()}...")
        result = run_algorithm(alg_name, graph, start, goal, coords)
        results[alg_name] = result
        print_result(result, alg_name)
    
    # Find best results
    valid_results = {name: res for name, res in results.items() if res['path']}
    
    if valid_results:
        best_distance = min(res['distance_m'] for res in valid_results.values())
        best_nodes = min(res['nodes_explored_count'] for res in valid_results.values())
        
        print("\n🏆 SUMMARY:")
        print("-" * 40)
        
        # Find algorithms with best distance
        optimal_algos = [name for name, res in valid_results.items() 
                        if res['distance_m'] == best_distance]
        print(f"📏 Shortest distance ({best_distance:.1f}m): {', '.join(optimal_algos).upper()}")
        
        # Find algorithms with fewest explored nodes
        efficient_algos = [name for name, res in valid_results.items() 
                          if res['nodes_explored_count'] == best_nodes]
        print(f"⚡ Most efficient ({best_nodes} nodes): {', '.join(efficient_algos).upper()}")
    
    return results


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="BotBrain - Campus Navigation Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -s "Main Gate" -g "Library"
  %(prog)s -s "Hostel A" -g "Food Court" -a a_star
  %(prog)s -s "Entry Gate" -g "Sports Complex" --compare
  %(prog)s --list-nodes
        """
    )
    
    parser.add_argument('-s', '--start', help='Starting location')
    parser.add_argument('-g', '--goal', help='Destination location')
    parser.add_argument('-a', '--algorithm', choices=list(ALGORITHMS.keys()),
                       default='a_star', help='Search algorithm to use (default: a_star)')
    parser.add_argument('-k', '--k-routes', type=int, default=3,
                       help='Number of alternative routes to find (default: 3)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all algorithms')
    parser.add_argument('--list-nodes', action='store_true',
                       help='List all available campus locations')
    parser.add_argument('--save-map', help='Save route visualization to file')
    
    args = parser.parse_args()
    
    # Load campus graph
    print("🗺️  Loading campus graph...")
    graph, coords = load_graph()
    print(f"✅ Graph loaded with {len(graph)} locations")
    
    # List nodes if requested
    if args.list_nodes:
        print("\n📍 Available campus locations:")
        print("-" * 40)
        for i, node in enumerate(sorted(graph.keys()), 1):
            print(f"{i:2d}. {node}")
        return
    
    # Validate inputs
    if not args.start or not args.goal:
        print("❌ Error: Please specify both start (-s) and goal (-g) locations")
        print("Use --list-nodes to see available locations")
        sys.exit(1)
    
    if args.start not in graph:
        print(f"❌ Error: '{args.start}' is not a valid location")
        print("Use --list-nodes to see available locations")
        sys.exit(1)
    
    if args.goal not in graph:
        print(f"❌ Error: '{args.goal}' is not a valid location")
        print("Use --list-nodes to see available locations")
        sys.exit(1)
    
    # Run comparison or single algorithm
    if args.compare:
        results = compare_algorithms(graph, coords, args.start, args.goal)
        
        # Save visualization if requested
        if args.save_map:
            print(f"\n💾 Saving route visualization to {args.save_map}...")
            paths = [res['path'] for res in results.values() if res['path']]
            if paths:
                save_static_map(paths, graph, coords, args.save_map, 
                               f"Campus Navigation: {args.start} → {args.goal}")
                print(f"✅ Map saved as {args.save_map}")
    else:
        # Run single algorithm
        result = run_algorithm(args.algorithm, graph, args.start, args.goal, coords)
        print_result(result, args.algorithm)
        
        # Get alternative routes
        if result['path'] and args.k_routes > 1:
            print(f"\n🛤️  Finding {args.k_routes} alternative routes...")
            k_paths = get_k_shortest_paths(graph, args.start, args.goal, coords, args.k_routes)
            
            if len(k_paths) > 1:
                print(f"Found {len(k_paths)} routes:")
                for i, path_info in enumerate(k_paths):
                    if i == 0:
                        continue  # Skip first (already shown)
                    print(f"  Route {i+1}: {' → '.join(path_info['path'])} "
                          f"({path_info['distance_m']:.1f}m)")
            else:
                print("No alternative routes found")
        
        # Save visualization if requested
        if args.save_map and result['path']:
            print(f"\n💾 Saving route visualization to {args.save_map}...")
            save_static_map([result['path']], graph, coords, args.save_map,
                           f"Campus Navigation: {args.start} → {args.goal}")
            print(f"✅ Map saved as {args.save_map}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)