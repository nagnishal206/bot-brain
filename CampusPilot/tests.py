"""
Tests and benchmarks for BotBrain Campus Navigation Assistant

Runs sample queries and generates comparison CSV files.
"""

import pandas as pd
from campus_graph import load_graph
from search_algorithms import run_algorithm, ALGORITHMS
import time


def run_test_queries():
    """
    Run predefined test queries and generate comparison CSV.
    """
    print("🧪 Running BotBrain test queries...")
    
    # Load graph
    graph, coords = load_graph()
    
    # Define test queries
    test_queries = [
        ("Main Gate", "Library"),
        ("Hostel A", "Admin Block"), 
        ("Sports Complex", "Faculty Block"),
        ("Entry Gate", "Food Court"),
        ("Engineering Block", "Cricket Ground"),
        ("Flag Post", "Auditorium")
    ]
    
    results = []
    
    print(f"Running {len(test_queries)} queries with {len(ALGORITHMS)} algorithms...")
    
    for start, goal in test_queries:
        query_name = f"{start} → {goal}"
        print(f"\n📍 Testing: {query_name}")
        
        # Skip if nodes don't exist
        if start not in graph or goal not in graph:
            print(f"⚠️  Skipping {query_name} - nodes not found in graph")
            continue
        
        for alg_name in ALGORITHMS:
            print(f"  Running {alg_name.upper()}...", end=" ")
            
            # Time the algorithm
            start_time = time.time()
            result = run_algorithm(alg_name, graph, start, goal, coords)
            execution_time = time.time() - start_time
            
            # Record results
            results.append({
                'query': query_name,
                'algorithm': alg_name.upper(),
                'distance_m': result['distance_m'],
                'walking_time_s': result['walking_time_s'],
                'nodes_explored_count': result['nodes_explored_count'],
                'path': ' → '.join(result['path']) if result['path'] else 'No path found',
                'path_length': len(result['path']) if result['path'] else 0,
                'execution_time_s': execution_time,
                'success': bool(result['path'])
            })
            
            status = "✅" if result['path'] else "❌"
            print(f"{status} ({execution_time*1000:.1f}ms)")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_filename = 'comparison.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved to {csv_filename}")
    
    # Print summary statistics
    print_summary_stats(df)
    
    return df


def print_summary_stats(df):
    """Print summary statistics from test results."""
    print("\n📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    # Success rate by algorithm
    print("\n🎯 Success Rate by Algorithm:")
    success_rates = df.groupby('algorithm')['success'].mean() * 100
    for alg, rate in success_rates.items():
        print(f"  {alg}: {rate:.1f}%")
    
    # Average performance for successful queries
    successful = df[df['success'] == True]
    if not successful.empty:
        print("\n⚡ Average Performance (Successful Queries Only):")
        
        avg_stats = successful.groupby('algorithm').agg({
            'distance_m': 'mean',
            'walking_time_s': 'mean', 
            'nodes_explored_count': 'mean',
            'execution_time_s': 'mean'
        })
        
        for alg in avg_stats.index:
            stats = avg_stats.loc[alg]
            print(f"  {alg}:")
            print(f"    Average distance: {stats['distance_m']:.1f}m")
            print(f"    Average time: {stats['walking_time_s']/60:.1f} min")
            print(f"    Average nodes explored: {stats['nodes_explored_count']:.1f}")
            print(f"    Average execution time: {stats['execution_time_s']*1000:.1f}ms")
    
    # Best algorithm by metric
    if not successful.empty:
        print("\n🏆 Best Algorithm by Metric:")
        
        # Best distance (lowest)
        best_distance = successful.groupby('algorithm')['distance_m'].mean().idxmin()
        print(f"  Shortest paths: {best_distance}")
        
        # Most efficient (fewest nodes)
        best_efficiency = successful.groupby('algorithm')['nodes_explored_count'].mean().idxmin()
        print(f"  Most efficient: {best_efficiency}")
        
        # Fastest execution
        best_speed = successful.groupby('algorithm')['execution_time_s'].mean().idxmin()
        print(f"  Fastest execution: {best_speed}")


def test_graph_connectivity():
    """Test graph connectivity and properties."""
    print("\n🔍 Testing graph connectivity...")
    
    graph, coords = load_graph()
    
    print(f"Graph has {len(graph)} nodes")
    print(f"Coordinates available for {len(coords)} nodes")
    
    # Count edges
    total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"Graph has {total_edges} bidirectional edges")
    
    # Check if all nodes have coordinates
    nodes_without_coords = set(graph.keys()) - set(coords.keys())
    if nodes_without_coords:
        print(f"⚠️  Nodes without coordinates: {nodes_without_coords}")
    else:
        print("✅ All nodes have coordinates")
    
    # Test basic connectivity (simple check)
    isolated_nodes = [node for node, neighbors in graph.items() if not neighbors]
    if isolated_nodes:
        print(f"⚠️  Isolated nodes found: {isolated_nodes}")
    else:
        print("✅ No isolated nodes found")


def run_performance_benchmark():
    """Run performance benchmark on different graph sizes."""
    print("\n⚡ Running performance benchmark...")
    
    graph, coords = load_graph()
    
    # Sample different start-goal pairs
    nodes = list(graph.keys())
    if len(nodes) < 2:
        print("❌ Not enough nodes for benchmark")
        return
    
    # Test pairs at different distances
    test_pairs = [
        (nodes[0], nodes[len(nodes)//4]),
        (nodes[0], nodes[len(nodes)//2]),
        (nodes[0], nodes[-1])
    ]
    
    benchmark_results = []
    
    for start, goal in test_pairs:
        if start not in graph or goal not in graph:
            continue
            
        print(f"Benchmarking: {start} → {goal}")
        
        for alg_name in ALGORITHMS:
            times = []
            
            # Run multiple times for average
            for _ in range(5):
                start_time = time.time()
                result = run_algorithm(alg_name, graph, start, goal, coords)
                execution_time = time.time() - start_time
                times.append(execution_time)
            
            avg_time = sum(times) / len(times)
            benchmark_results.append({
                'pair': f"{start} → {goal}",
                'algorithm': alg_name.upper(),
                'avg_execution_time_ms': avg_time * 1000,
                'nodes_explored': result['nodes_explored_count']
            })
    
    # Save benchmark results
    if benchmark_results:
        benchmark_df = pd.DataFrame(benchmark_results)
        benchmark_df.to_csv('benchmark_results.csv', index=False)
        print("💾 Benchmark results saved to benchmark_results.csv")
        
        # Print fastest algorithm
        fastest = benchmark_df.groupby('algorithm')['avg_execution_time_ms'].mean().idxmin()
        print(f"🏃 Fastest algorithm overall: {fastest}")


if __name__ == "__main__":
    print("🧠 BotBrain Testing Suite")
    print("=" * 40)
    
    try:
        # Test graph loading and connectivity
        test_graph_connectivity()
        
        # Run main test queries
        run_test_queries()
        
        # Run performance benchmark
        run_performance_benchmark()
        
        print("\n✅ All tests completed successfully!")
        print("📊 Check comparison.csv and benchmark_results.csv for detailed results")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()