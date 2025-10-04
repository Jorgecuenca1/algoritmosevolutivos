"""
Flight Optimization Problem
===========================
Multi-objective optimization for selecting the best flight based on:
- Price (minimize)
- Duration (minimize)
- Stopovers/Escalas (minimize)
- Comfort (maximize)
"""

import numpy as np
import json


class FlightOptimizationProblem:
    """
    Optimizes flight selection based on multiple criteria
    """

    def __init__(self, flights_data, weights=None):
        """
        Parameters:
        -----------
        flights_data : list of dicts
            Each dict contains: {'name': str, 'price': float, 'duration': float,
                                'stopovers': int, 'comfort': float}
        weights : dict
            Importance weights {'w_price': float, 'w_duration': float,
                               'w_stopovers': float, 'w_comfort': float}
        """
        self.flights = flights_data
        self.n_flights = len(flights_data)

        # Default weights (can be customized by user)
        if weights is None:
            self.weights = {
                'w_price': 0.35,     # 35% importance to price
                'w_duration': 0.35,  # 35% importance to duration
                'w_stopovers': 0.10, # 10% importance to stopovers
                'w_comfort': 0.20    # 20% importance to comfort
            }
        else:
            self.weights = weights

        # Extract and normalize data
        self.prices = np.array([f['price'] for f in flights_data])
        self.durations = np.array([f['duration'] for f in flights_data])
        self.stopovers = np.array([f['stopovers'] for f in flights_data])
        self.comforts = np.array([f['comfort'] for f in flights_data])

        # Normalization factors (max values for each criterion)
        self.max_price = np.max(self.prices) if np.max(self.prices) > 0 else 1
        self.max_duration = np.max(self.durations) if np.max(self.durations) > 0 else 1
        self.max_stopovers = np.max(self.stopovers) if np.max(self.stopovers) > 0 else 1
        self.max_comfort = 10.0  # Comfort scale 1-10

    def evaluate(self, solution):
        """
        Evaluate the fitness of a solution (flight index)

        For continuous optimization algorithms, solution is a float array.
        We convert it to a flight index.

        Lower value = better flight
        """
        # Convert continuous solution to discrete flight index
        # Take the first dimension and map to [0, n_flights-1]
        if isinstance(solution, (list, np.ndarray)):
            # For continuous algorithms, solution is in search space [0, n_flights]
            # We need to map it to flight indices [0, n_flights-1]
            if len(solution) > 0:
                # Use first dimension, ensure it's within bounds
                idx = int(abs(solution[0])) % self.n_flights
            else:
                idx = 0
        else:
            idx = int(solution) % self.n_flights

        # Get flight attributes
        price = self.prices[idx]
        duration = self.durations[idx]
        stopovers = self.stopovers[idx]
        comfort = self.comforts[idx]

        # Calculate normalized cost function
        # F(vuelo) = w1*(price/max_price) + w2*(duration/max_duration) +
        #            w3*(stopovers/max_stopovers) - w4*(comfort/10)

        cost = (
            self.weights['w_price'] * (price / self.max_price) +
            self.weights['w_duration'] * (duration / self.max_duration) +
            self.weights['w_stopovers'] * (stopovers / self.max_stopovers) -
            self.weights['w_comfort'] * (comfort / self.max_comfort)
        )

        return cost

    def evaluate_all_flights(self):
        """
        Evaluate all flights and return sorted list

        Returns:
        --------
        list of tuples: (flight_index, flight_data, cost)
        """
        results = []

        for idx in range(self.n_flights):
            # Get flight attributes directly (no conversion needed)
            price = self.prices[idx]
            duration = self.durations[idx]
            stopovers = self.stopovers[idx]
            comfort = self.comforts[idx]

            # Calculate normalized cost function directly
            cost = (
                self.weights['w_price'] * (price / self.max_price) +
                self.weights['w_duration'] * (duration / self.max_duration) +
                self.weights['w_stopovers'] * (stopovers / self.max_stopovers) -
                self.weights['w_comfort'] * (comfort / self.max_comfort)
            )

            flight_info = self.flights[idx].copy()
            flight_info['cost'] = cost
            flight_info['index'] = idx

            results.append((idx, flight_info, cost))

        # Sort by cost (lower is better)
        results.sort(key=lambda x: x[2])

        return results

    def get_best_flight(self):
        """
        Get the best flight based on current weights

        Returns:
        --------
        dict: Best flight information with cost
        """
        results = self.evaluate_all_flights()
        return results[0][1]  # Return the flight_info dict

    def get_flight_ranking(self):
        """
        Get ranking of all flights

        Returns:
        --------
        list: Sorted list of flight information dictionaries
        """
        results = self.evaluate_all_flights()
        return [flight_info for _, flight_info, _ in results]


def create_flight_problem_from_data(flights_data, weights=None):
    """
    Factory function to create a FlightOptimizationProblem

    Parameters:
    -----------
    flights_data : list of dicts or JSON string
    weights : dict or None

    Returns:
    --------
    FlightOptimizationProblem instance
    """
    if isinstance(flights_data, str):
        flights_data = json.loads(flights_data)

    return FlightOptimizationProblem(flights_data, weights)


# Example usage and data
EXAMPLE_FLIGHTS_BOGOTA_MADRID = [
    {'name': 'Vuelo A', 'price': 850, 'duration': 12, 'stopovers': 1, 'comfort': 7},
    {'name': 'Vuelo B', 'price': 1200, 'duration': 9, 'stopovers': 0, 'comfort': 9},
    {'name': 'Vuelo C', 'price': 650, 'duration': 20, 'stopovers': 2, 'comfort': 5},
    {'name': 'Vuelo D', 'price': 900, 'duration': 14, 'stopovers': 1, 'comfort': 8},
    {'name': 'Vuelo E', 'price': 300, 'duration': 1, 'stopovers': 2, 'comfort': 10},
]


def get_flight_optimization_function(flights_data, weights=None):
    """
    Returns a function compatible with evolutionary algorithms

    Parameters:
    -----------
    flights_data : list of dicts
        Flight information
    weights : dict
        Optimization weights

    Returns:
    --------
    callable: Function that takes solution array and returns cost
    """
    problem = FlightOptimizationProblem(flights_data, weights)

    def objective_function(x):
        """
        Wrapper function for evolutionary algorithms
        x: numpy array (solution in continuous space)
        """
        return problem.evaluate(x)

    # Attach problem instance for access to results
    objective_function.problem = problem

    return objective_function


if __name__ == "__main__":
    # Example usage
    print("Flight Optimization Problem - Example")
    print("=" * 50)

    problem = FlightOptimizationProblem(EXAMPLE_FLIGHTS_BOGOTA_MADRID)

    print("\nAll flights ranked:")
    ranking = problem.get_flight_ranking()

    for i, flight in enumerate(ranking, 1):
        print(f"\n{i}. {flight['name']}")
        print(f"   Price: ${flight['price']}")
        print(f"   Duration: {flight['duration']} hrs")
        print(f"   Stopovers: {flight['stopovers']}")
        print(f"   Comfort: {flight['comfort']}/10")
        print(f"   Total Cost: {flight['cost']:.4f}")

    print("\n" + "=" * 50)
    print("Best flight:")
    best = problem.get_best_flight()
    print(f"{best['name']} - Total Cost: {best['cost']:.4f}")

    # Test with different weights
    print("\n" + "=" * 50)
    print("Testing with comfort-focused weights:")
    comfort_weights = {
        'w_price': 0.2,
        'w_duration': 0.2,
        'w_stopovers': 0.1,
        'w_comfort': 0.5  # 50% importance to comfort
    }

    problem_comfort = FlightOptimizationProblem(EXAMPLE_FLIGHTS_BOGOTA_MADRID, comfort_weights)
    best_comfort = problem_comfort.get_best_flight()
    print(f"Best flight: {best_comfort['name']} - Cost: {best_comfort['cost']:.4f}")
    print(f"Comfort: {best_comfort['comfort']}/10")
