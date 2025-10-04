#!/usr/bin/env python
"""
Test script for evolutionary algorithms
"""

import sys
import os
import django

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evolutionary_optimizer.settings')
django.setup()

from algorithms.optimization_functions import OPTIMIZATION_FUNCTIONS
from algorithms.evolutionary_algorithms import ALGORITHMS


def test_algorithm(algorithm_key, function_key, iterations=50):
    """Test a specific algorithm on a specific function"""
    print(f"\nTesting {ALGORITHMS[algorithm_key]['name']} on {OPTIMIZATION_FUNCTIONS[function_key]['name']}")
    print("-" * 80)

    # Get algorithm and function
    algorithm_info = ALGORITHMS[algorithm_key]
    function_info = OPTIMIZATION_FUNCTIONS[function_key]

    # Set up bounds for 2D optimization
    bounds = [function_info['domain'], function_info['domain']]

    # Initialize algorithm
    algorithm_class = algorithm_info['class']
    if algorithm_key == 'ga':
        algorithm = algorithm_class(
            objective_func=function_info['function'],
            bounds=bounds,
            generations=iterations,
            population_size=30
        )
    elif algorithm_key == 'pso':
        algorithm = algorithm_class(
            objective_func=function_info['function'],
            bounds=bounds,
            iterations=iterations,
            swarm_size=30
        )
    elif algorithm_key == 'aco':
        algorithm = algorithm_class(
            objective_func=function_info['function'],
            bounds=bounds,
            iterations=iterations,
            num_ants=30
        )
    elif algorithm_key == 'tlbo':
        algorithm = algorithm_class(
            objective_func=function_info['function'],
            bounds=bounds,
            iterations=iterations,
            population_size=30
        )
    elif algorithm_key == 'ts':
        algorithm = algorithm_class(
            objective_func=function_info['function'],
            bounds=bounds,
            iterations=iterations
        )

    try:
        # Run optimization
        history = algorithm.optimize()

        # Get results
        final_result = history[-1]
        initial_result = history[0]

        print(f"Algorithm completed successfully!")
        print(f"Results:")
        print(f"   Initial fitness: {initial_result['best_fitness']:.6f}")
        print(f"   Final fitness: {final_result['best_fitness']:.6f}")
        print(f"   Best solution: ({final_result['best_solution'][0]:.4f}, {final_result['best_solution'][1]:.4f})")
        print(f"   Global optimum: ({function_info['global_optimum'][0]}, {function_info['global_optimum'][1]})")
        print(f"   Global minimum: {function_info['global_minimum']}")

        # Calculate improvement
        improvement = initial_result['best_fitness'] - final_result['best_fitness']
        improvement_percent = (improvement / initial_result['best_fitness']) * 100 if initial_result['best_fitness'] != 0 else 0
        print(f"   Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")

        return True

    except Exception as e:
        print(f"Algorithm failed with error: {str(e)}")
        return False


def main():
    """Run tests for all algorithms on all functions"""
    print("Evolutionary Algorithms Test Suite")
    print("=" * 80)

    # Test configurations
    test_configs = [
        ('ga', 'sphere'),
        ('pso', 'sphere'),
        ('aco', 'sphere'),
        ('tlbo', 'sphere'),
        ('ts', 'sphere'),
        ('ga', 'rosenbrock'),
        ('pso', 'rosenbrock'),
    ]

    successful_tests = 0
    total_tests = len(test_configs)

    for algorithm_key, function_key in test_configs:
        success = test_algorithm(algorithm_key, function_key, iterations=30)
        if success:
            successful_tests += 1

    print("\n" + "=" * 80)
    print(f"Test Summary: {successful_tests}/{total_tests} tests passed")

    if successful_tests == total_tests:
        print("All tests passed! The application is ready to use.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)