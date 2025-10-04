import numpy as np
from itertools import permutations

def sphere_function(x):
    """
    Sphere function: f(x) = sum(x^2)
    Global minimum: f(0) = 0
    Domain: [-5, 5] for each dimension
    """
    return np.sum(x**2)

def rosenbrock_function(x):
    """
    Rosenbrock function: f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
    Global minimum: f(1, 1, ..., 1) = 0
    Domain: [-5, 5] for each dimension
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin_function(x):
    """
    Rastrigin function: f(x) = A*n + sum(x[i]^2 - A*cos(2*pi*x[i]))
    Global minimum: f(0) = 0
    Domain: [-5.12, 5.12] for each dimension
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# ===== COMPLEX OPTIMIZATION PROBLEMS =====

def ackley_function(x):
    """
    Ackley function: highly multimodal with many local minima
    Global minimum: f(0,...,0) = 0
    Domain: [-32.768, 32.768] for each dimension
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def schwefel_function(x):
    """
    Schwefel function: deceptive with global optimum far from local optima
    Global minimum: f(420.9687,...,420.9687) ≈ 0
    Domain: [-500, 500] for each dimension
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def griewank_function(x):
    """
    Griewank function: highly multimodal with many regularly distributed local minima
    Global minimum: f(0,...,0) = 0
    Domain: [-600, 600] for each dimension
    """
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1

def levy_function(x):
    """
    Levy function: multimodal with several local minima
    Global minimum: f(1,...,1) = 0
    Domain: [-10, 10] for each dimension
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def michalewicz_function(x):
    """
    Michalewicz function: highly multimodal, steep valleys and ridges
    Global minimum: varies by dimension (for 2D: f ≈ -1.8013)
    Domain: [0, π] for each dimension
    """
    m = 10
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m))

# ===== COMBINATORIAL OPTIMIZATION PROBLEMS =====

class TSPProblem:
    """Traveling Salesman Problem - find shortest route visiting all cities"""
    def __init__(self, cities=None, n_cities=10, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if cities is None:
            # Generate random cities in 2D space
            self.cities = np.random.rand(n_cities, 2) * 100
        else:
            self.cities = np.array(cities)

        self.n_cities = len(self.cities)
        self.distance_matrix = self._calculate_distances()

    def _calculate_distances(self):
        """Calculate Euclidean distance matrix between all cities"""
        n = self.n_cities
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist_matrix

    def evaluate(self, route):
        """
        Evaluate a route (sequence of city indices)
        route: array-like, permutation of [0, 1, ..., n-1]
        """
        route = np.array(route, dtype=int)
        total_distance = 0
        for i in range(len(route)):
            total_distance += self.distance_matrix[route[i], route[(i + 1) % len(route)]]
        return total_distance

    def get_best_solution_brute_force(self, max_cities=10):
        """Find optimal solution using brute force (only for small instances)"""
        if self.n_cities > max_cities:
            return None, None

        best_route = None
        best_distance = float('inf')

        for route in permutations(range(self.n_cities)):
            distance = self.evaluate(route)
            if distance < best_distance:
                best_distance = distance
                best_route = route

        return best_route, best_distance

class KnapsackProblem:
    """0/1 Knapsack Problem - maximize value without exceeding weight capacity"""
    def __init__(self, weights=None, values=None, capacity=None, n_items=20, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if weights is None or values is None:
            # Generate random items
            self.weights = np.random.randint(1, 50, n_items)
            self.values = np.random.randint(1, 100, n_items)
        else:
            self.weights = np.array(weights)
            self.values = np.array(values)

        self.n_items = len(self.weights)

        if capacity is None:
            # Set capacity to ~50% of total weight
            self.capacity = int(np.sum(self.weights) * 0.5)
        else:
            self.capacity = capacity

    def evaluate(self, solution):
        """
        Evaluate a solution (binary array indicating item selection)
        solution: array-like, binary values [0 or 1]
        Returns negative value (for minimization), or large penalty if infeasible
        """
        solution = np.array(solution, dtype=int)
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)

        if total_weight > self.capacity:
            # Penalize infeasible solutions
            penalty = (total_weight - self.capacity) * 1000
            return penalty - total_value

        # Return negative value for minimization
        return -total_value

    def repair_solution(self, solution):
        """Repair an infeasible solution by removing items"""
        solution = np.array(solution, dtype=int)
        indices = np.where(solution == 1)[0]

        # Remove items until feasible
        while np.sum(solution * self.weights) > self.capacity and len(indices) > 0:
            # Remove least valuable item per unit weight
            value_per_weight = self.values[indices] / self.weights[indices]
            remove_idx = indices[np.argmin(value_per_weight)]
            solution[remove_idx] = 0
            indices = np.where(solution == 1)[0]

        return solution

class JobSchedulingProblem:
    """Job Shop Scheduling Problem - minimize makespan"""
    def __init__(self, processing_times=None, n_jobs=10, n_machines=3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if processing_times is None:
            # Generate random processing times
            self.processing_times = np.random.randint(1, 20, (n_jobs, n_machines))
        else:
            self.processing_times = np.array(processing_times)

        self.n_jobs, self.n_machines = self.processing_times.shape

    def evaluate(self, schedule):
        """
        Evaluate a schedule (sequence of job indices)
        schedule: array-like, permutation of [0, 1, ..., n_jobs-1]
        Returns makespan (total completion time)
        """
        schedule = np.array(schedule, dtype=int)

        # Track completion time for each machine
        machine_times = np.zeros(self.n_machines)
        # Track when each job finishes its previous operation
        job_times = np.zeros(self.n_jobs)

        for job in schedule:
            for machine in range(self.n_machines):
                # Job can start on this machine only after:
                # 1. Previous operation of this job is complete
                # 2. This machine is free
                start_time = max(job_times[job], machine_times[machine])
                processing_time = self.processing_times[job, machine]
                finish_time = start_time + processing_time

                job_times[job] = finish_time
                machine_times[machine] = finish_time

        # Makespan is the maximum completion time
        return np.max(machine_times)

class VehicleRoutingProblem:
    """Capacitated Vehicle Routing Problem (CVRP) - minimize total route distance"""
    def __init__(self, locations=None, demands=None, n_locations=15, vehicle_capacity=100,
                 n_vehicles=3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if locations is None:
            # First location is depot (0,0), rest are random customers
            self.locations = np.vstack([np.zeros(2), np.random.rand(n_locations - 1, 2) * 100])
        else:
            self.locations = np.array(locations)

        self.n_locations = len(self.locations)

        if demands is None:
            # Depot has 0 demand, customers have random demands
            self.demands = np.concatenate([np.zeros(1), np.random.randint(5, 30, self.n_locations - 1)])
        else:
            self.demands = np.array(demands)

        self.vehicle_capacity = vehicle_capacity
        self.n_vehicles = n_vehicles
        self.distance_matrix = self._calculate_distances()

    def _calculate_distances(self):
        """Calculate Euclidean distance matrix"""
        n = self.n_locations
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(self.locations[i] - self.locations[j])
        return dist_matrix

    def evaluate(self, routes):
        """
        Evaluate vehicle routes
        routes: list of lists, e.g., [[1,2,3], [4,5], [6,7,8]]
        Returns total distance + penalty for constraint violations
        """
        total_distance = 0
        penalty = 0

        for route in routes:
            if len(route) == 0:
                continue

            # Check capacity constraint
            route_demand = sum(self.demands[i] for i in route)
            if route_demand > self.vehicle_capacity:
                penalty += (route_demand - self.vehicle_capacity) * 1000

            # Calculate route distance (depot -> customers -> depot)
            route_distance = self.distance_matrix[0, route[0]]  # Depot to first customer
            for i in range(len(route) - 1):
                route_distance += self.distance_matrix[route[i], route[i + 1]]
            route_distance += self.distance_matrix[route[-1], 0]  # Last customer to depot

            total_distance += route_distance

        return total_distance + penalty

# Dictionary of available functions
OPTIMIZATION_FUNCTIONS = {
    # Basic benchmark functions
    'sphere': {
        'function': sphere_function,
        'name': 'Sphere Function',
        'description': 'f(x,y) = x² + y²',
        'domain': [-5, 5],
        'global_optimum': [0, 0],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'easy'
    },
    'rosenbrock': {
        'function': rosenbrock_function,
        'name': 'Rosenbrock Function',
        'description': 'f(x,y) = 100(y - x²)² + (1 - x)²',
        'domain': [-5, 5],
        'global_optimum': [1, 1],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'medium'
    },
    'rastrigin': {
        'function': rastrigin_function,
        'name': 'Rastrigin Function',
        'description': 'f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))',
        'domain': [-5.12, 5.12],
        'global_optimum': [0, 0],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'hard'
    },
    # Advanced benchmark functions
    'ackley': {
        'function': ackley_function,
        'name': 'Ackley Function',
        'description': 'Highly multimodal with many local minima',
        'domain': [-32.768, 32.768],
        'global_optimum': [0, 0],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'hard'
    },
    'schwefel': {
        'function': schwefel_function,
        'name': 'Schwefel Function',
        'description': 'Deceptive function with distant global optimum',
        'domain': [-500, 500],
        'global_optimum': [420.9687, 420.9687],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'very_hard'
    },
    'griewank': {
        'function': griewank_function,
        'name': 'Griewank Function',
        'description': 'Many regularly distributed local minima',
        'domain': [-600, 600],
        'global_optimum': [0, 0],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'hard'
    },
    'levy': {
        'function': levy_function,
        'name': 'Levy Function',
        'description': 'Multimodal with several local minima',
        'domain': [-10, 10],
        'global_optimum': [1, 1],
        'global_minimum': 0,
        'type': 'continuous',
        'difficulty': 'medium'
    },
    'michalewicz': {
        'function': michalewicz_function,
        'name': 'Michalewicz Function',
        'description': 'Steep valleys and ridges',
        'domain': [0, np.pi],
        'global_optimum': None,  # Varies by dimension
        'global_minimum': -1.8013,  # For 2D
        'type': 'continuous',
        'difficulty': 'very_hard'
    },
}