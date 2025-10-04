"""
Flight Optimization using Evolutionary Algorithms
==================================================
Implements Genetic Algorithm (GA) for flight selection optimization
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from .flight_optimizer import FlightOptimizationProblem


class GeneticAlgorithmFlights:
    """
    Genetic Algorithm for Flight Optimization

    Chromosome encoding: Each individual represents a flight selection
    - For discrete problems: integer index [0, n_flights-1]
    - For multi-city trips: array of flight indices
    """

    def __init__(self,
                 problem: FlightOptimizationProblem,
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 elitism: int = 2):
        """
        Initialize Genetic Algorithm

        Parameters:
        -----------
        problem : FlightOptimizationProblem
            The flight optimization problem
        population_size : int
            Number of individuals in population
        generations : int
            Number of generations to evolve
        crossover_rate : float
            Probability of crossover [0,1]
        mutation_rate : float
            Probability of mutation [0,1]
        elitism : int
            Number of best individuals to preserve
        """
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.n_flights = problem.n_flights

        # History tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_solutions': [],
            'diversity': []
        }

    def create_individual(self) -> int:
        """Create a random individual (flight index)"""
        return random.randint(0, self.n_flights - 1)

    def create_population(self) -> List[int]:
        """Create initial random population"""
        return [self.create_individual() for _ in range(self.population_size)]

    def evaluate_fitness(self, individual: int) -> float:
        """
        Evaluate fitness of an individual

        Returns negative cost (we want to minimize cost, so maximize -cost)
        """
        # Create solution array for the problem
        solution = np.array([individual])
        cost = self.problem.evaluate(solution)
        # Return negative cost (GA maximizes fitness)
        return -cost

    def tournament_selection(self, population: List[int],
                            fitness_values: List[float],
                            tournament_size: int = 3) -> int:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]

    def crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """
        Crossover operation for discrete flight selection

        For single flight selection, we use a weighted average approach
        that occasionally selects alternate flights
        """
        if random.random() < self.crossover_rate:
            # 70% chance to inherit from better parent
            # 30% chance to explore nearby flights
            if random.random() < 0.7:
                offspring1 = parent1 if random.random() < 0.5 else parent2
                offspring2 = parent2 if random.random() < 0.5 else parent1
            else:
                # Explore nearby flights
                offspring1 = (parent1 + random.randint(-1, 1)) % self.n_flights
                offspring2 = (parent2 + random.randint(-1, 1)) % self.n_flights
        else:
            offspring1, offspring2 = parent1, parent2

        return offspring1, offspring2

    def mutate(self, individual: int) -> int:
        """
        Mutation operation

        Randomly change to a different flight
        """
        if random.random() < self.mutation_rate:
            # Random mutation: select any other flight
            return random.randint(0, self.n_flights - 1)
        return individual

    def calculate_diversity(self, population: List[int]) -> float:
        """Calculate population diversity (unique individuals / total)"""
        return len(set(population)) / len(population)

    def optimize(self) -> Dict:
        """
        Run Genetic Algorithm optimization

        Returns:
        --------
        dict: Optimization results with history
        """
        # Initialize population
        population = self.create_population()

        # Initialize standard history format for Django view compatibility
        standard_history = []

        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_values = [self.evaluate_fitness(ind) for ind in population]

            # Track statistics
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            best_idx = fitness_values.index(best_fitness)
            best_solution = population[best_idx]
            diversity = self.calculate_diversity(population)

            self.history['best_fitness'].append(-best_fitness)  # Store as cost
            self.history['avg_fitness'].append(-avg_fitness)
            self.history['best_solutions'].append(best_solution)
            self.history['diversity'].append(diversity)

            # Add to standard history format (for Django view)
            standard_history.append({
                'iteration': generation,
                'best_solution': [float(best_solution), 0.0],  # Format as 2D array
                'best_fitness': float(-best_fitness),
                'avg_fitness': float(-avg_fitness)
            })

            # Elitism: preserve best individuals
            elite_indices = sorted(range(len(fitness_values)),
                                 key=lambda i: fitness_values[i],
                                 reverse=True)[:self.elitism]
            elite = [population[i] for i in elite_indices]

            # Create new population
            new_population = elite.copy()

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)

                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)

                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                new_population.extend([offspring1, offspring2])

            # Trim to population size
            population = new_population[:self.population_size]

        # Final evaluation
        final_fitness = [self.evaluate_fitness(ind) for ind in population]
        best_idx = final_fitness.index(max(final_fitness))
        best_solution = population[best_idx]
        best_cost = -final_fitness[best_idx]

        # Get flight information
        best_flight = self.problem.flights[best_solution]

        return {
            'best_solution': best_solution,
            'best_cost': best_cost,
            'best_flight': best_flight,
            'history': standard_history,  # Use standard format for Django view
            'detailed_history': self.history,  # Keep detailed history
            'final_population': population,
            'final_fitness': [-f for f in final_fitness]
        }


class ParticleSwarmFlights:
    """
    Particle Swarm Optimization for Flight Selection

    Adapts continuous PSO to discrete flight selection problem
    """

    def __init__(self,
                 problem: FlightOptimizationProblem,
                 n_particles: int = 30,
                 iterations: int = 100,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize PSO for flights

        Parameters:
        -----------
        problem : FlightOptimizationProblem
        n_particles : int
            Number of particles
        iterations : int
            Number of iterations
        w : float
            Inertia weight
        c1 : float
            Cognitive parameter
        c2 : float
            Social parameter
        """
        self.problem = problem
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_flights = problem.n_flights

        # History
        self.history = {
            'best_cost': [],
            'avg_cost': [],
            'best_solutions': []
        }

    def evaluate(self, particle_position: float) -> float:
        """Evaluate particle (convert continuous to discrete)"""
        flight_idx = int(abs(particle_position) % self.n_flights)
        solution = np.array([flight_idx])
        return self.problem.evaluate(solution)

    def optimize(self) -> Dict:
        """Run PSO optimization"""
        # Initialize particles in continuous space [0, n_flights]
        positions = np.random.uniform(0, self.n_flights, self.n_particles)
        velocities = np.random.uniform(-1, 1, self.n_particles)

        # Personal best
        pbest_positions = positions.copy()
        pbest_costs = np.array([self.evaluate(p) for p in positions])

        # Global best
        gbest_idx = np.argmin(pbest_costs)
        gbest_position = pbest_positions[gbest_idx]
        gbest_cost = pbest_costs[gbest_idx]

        # Initialize standard history format
        standard_history = []

        # Optimization loop
        for iteration in range(self.iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                               self.c2 * r2 * (gbest_position - positions[i]))

                # Update position
                positions[i] += velocities[i]
                positions[i] = positions[i] % self.n_flights  # Keep in bounds

                # Evaluate
                cost = self.evaluate(positions[i])

                # Update personal best
                if cost < pbest_costs[i]:
                    pbest_costs[i] = cost
                    pbest_positions[i] = positions[i]

                # Update global best
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_position = positions[i]

            # Track history
            avg_cost = np.mean(pbest_costs)
            self.history['best_cost'].append(gbest_cost)
            self.history['avg_cost'].append(avg_cost)
            self.history['best_solutions'].append(int(gbest_position % self.n_flights))

            # Add to standard history format
            standard_history.append({
                'iteration': iteration,
                'best_solution': [float(gbest_position), 0.0],
                'best_fitness': float(gbest_cost),
                'avg_fitness': float(avg_cost)
            })

        # Get best flight
        best_flight_idx = int(gbest_position % self.n_flights)
        best_flight = self.problem.flights[best_flight_idx]

        return {
            'best_solution': best_flight_idx,
            'best_cost': gbest_cost,
            'best_flight': best_flight,
            'history': standard_history,  # Use standard format for Django view
            'detailed_history': self.history
        }


# Example usage
if __name__ == "__main__":
    from .flight_optimizer import EXAMPLE_FLIGHTS_BOGOTA_MADRID

    print("=" * 60)
    print("FLIGHT OPTIMIZATION USING EVOLUTIONARY ALGORITHMS")
    print("=" * 60)

    # Create problem
    problem = FlightOptimizationProblem(EXAMPLE_FLIGHTS_BOGOTA_MADRID)

    print("\n📊 Available Flights:")
    for i, flight in enumerate(EXAMPLE_FLIGHTS_BOGOTA_MADRID):
        print(f"{i+1}. {flight['name']}: ${flight['price']}, "
              f"{flight['duration']}h, {flight['stopovers']} stops, "
              f"comfort {flight['comfort']}/10")

    # Run Genetic Algorithm
    print("\n" + "=" * 60)
    print("🧬 GENETIC ALGORITHM")
    print("=" * 60)

    ga = GeneticAlgorithmFlights(problem, population_size=50, generations=100)
    ga_result = ga.optimize()

    print(f"\n✅ Best Flight Found: {ga_result['best_flight']['name']}")
    print(f"   Cost Score: {ga_result['best_cost']:.4f}")
    print(f"   Price: ${ga_result['best_flight']['price']}")
    print(f"   Duration: {ga_result['best_flight']['duration']}h")
    print(f"   Stopovers: {ga_result['best_flight']['stopovers']}")
    print(f"   Comfort: {ga_result['best_flight']['comfort']}/10")

    print(f"\n📈 Convergence:")
    print(f"   Initial Best Cost: {ga_result['history']['best_fitness'][0]:.4f}")
    print(f"   Final Best Cost: {ga_result['history']['best_fitness'][-1]:.4f}")
    print(f"   Improvement: {ga_result['history']['best_fitness'][0] - ga_result['history']['best_fitness'][-1]:.4f}")

    # Run Particle Swarm Optimization
    print("\n" + "=" * 60)
    print("🌊 PARTICLE SWARM OPTIMIZATION")
    print("=" * 60)

    pso = ParticleSwarmFlights(problem, n_particles=30, iterations=100)
    pso_result = pso.optimize()

    print(f"\n✅ Best Flight Found: {pso_result['best_flight']['name']}")
    print(f"   Cost Score: {pso_result['best_cost']:.4f}")
    print(f"   Price: ${pso_result['best_flight']['price']}")
    print(f"   Duration: {pso_result['best_flight']['duration']}h")
    print(f"   Stopovers: {pso_result['best_flight']['stopovers']}")
    print(f"   Comfort: {pso_result['best_flight']['comfort']}/10")

    print(f"\n📈 Convergence:")
    print(f"   Initial Best Cost: {pso_result['history']['best_cost'][0]:.4f}")
    print(f"   Final Best Cost: {pso_result['history']['best_cost'][-1]:.4f}")
    print(f"   Improvement: {pso_result['history']['best_cost'][0] - pso_result['history']['best_cost'][-1]:.4f}")

    print("\n" + "=" * 60)
