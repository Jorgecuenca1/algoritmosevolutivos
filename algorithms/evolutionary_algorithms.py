import numpy as np
import random
from typing import List, Tuple, Callable, Dict, Any
from .quantum_adaptive_evolution import QuantumAdaptiveEvolution

class GeneticAlgorithm:
    """Genetic Algorithm implementation"""

    def __init__(self, objective_func: Callable, bounds: List[Tuple[float, float]],
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dimensions = len(bounds)

    def initialize_population(self):
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for lower, upper in self.bounds:
                individual.append(random.uniform(lower, upper))
            population.append(individual)
        return np.array(population)

    def selection(self, population, fitness):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return np.array(selected)

    def crossover(self, parent1, parent2):
        """Single point crossover"""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.dimensions - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutation(self, individual):
        """Gaussian mutation"""
        mutated = individual.copy()
        for i in range(self.dimensions):
            if random.random() < self.mutation_rate:
                lower, upper = self.bounds[i]
                mutation_strength = (upper - lower) * 0.1
                mutated[i] += random.gauss(0, mutation_strength)
                mutated[i] = np.clip(mutated[i], lower, upper)
        return mutated

    def optimize(self):
        """Run the genetic algorithm"""
        history = []
        population = self.initialize_population()

        for generation in range(self.generations):
            # Evaluate fitness
            fitness = [self.objective_func(ind) for ind in population]

            # Track best solution
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx].copy()

            history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'best_solution': best_solution.tolist(),
                'avg_fitness': np.mean(fitness)
            })

            # Selection
            selected = self.selection(population, fitness)

            # Crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected[i], selected[i+1] if i+1 < self.population_size else selected[0]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])

            population = np.array(new_population[:self.population_size])

        return history

class ParticleSwarmOptimization:
    """Particle Swarm Optimization implementation"""

    def __init__(self, objective_func: Callable, bounds: List[Tuple[float, float]],
                 swarm_size: int = 30, iterations: int = 100,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.objective_func = objective_func
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        self.dimensions = len(bounds)

    def initialize_swarm(self):
        """Initialize particle positions and velocities"""
        positions = []
        velocities = []

        for _ in range(self.swarm_size):
            position = []
            velocity = []
            for lower, upper in self.bounds:
                position.append(random.uniform(lower, upper))
                velocity.append(random.uniform(-abs(upper-lower)/10, abs(upper-lower)/10))
            positions.append(position)
            velocities.append(velocity)

        return np.array(positions), np.array(velocities)

    def optimize(self):
        """Run particle swarm optimization"""
        history = []
        positions, velocities = self.initialize_swarm()

        # Initialize personal best
        personal_best_positions = positions.copy()
        personal_best_fitness = [self.objective_func(pos) for pos in positions]

        # Initialize global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]

        for iteration in range(self.iterations):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                               self.c2 * r2 * (global_best_position - positions[i]))

                # Update position
                positions[i] += velocities[i]

                # Apply bounds
                for j in range(self.dimensions):
                    lower, upper = self.bounds[j]
                    positions[i][j] = np.clip(positions[i][j], lower, upper)

                # Evaluate fitness
                fitness = self.objective_func(positions[i])

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i].copy()

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[i].copy()

            # Calculate average fitness
            current_fitness = [self.objective_func(pos) for pos in positions]
            avg_fitness = np.mean(current_fitness)

            history.append({
                'generation': iteration,
                'best_fitness': global_best_fitness,
                'best_solution': global_best_position.tolist(),
                'avg_fitness': avg_fitness
            })

        return history

class AntColonyOptimization:
    """Ant Colony Optimization for continuous optimization"""

    def __init__(self, objective_func: Callable, bounds: List[Tuple[float, float]],
                 num_ants: int = 30, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta   # heuristic importance
        self.rho = rho     # evaporation rate
        self.dimensions = len(bounds)
        self.grid_size = 20  # discretization for continuous space

    def discretize_space(self):
        """Create discrete grid for pheromone deposition"""
        grids = []
        for lower, upper in self.bounds:
            grid = np.linspace(lower, upper, self.grid_size)
            grids.append(grid)
        return grids

    def get_grid_index(self, position, grids):
        """Get grid indices for a continuous position"""
        indices = []
        for i, pos in enumerate(position):
            grid = grids[i]
            idx = np.argmin(np.abs(grid - pos))
            indices.append(idx)
        return tuple(indices)

    def optimize(self):
        """Run ant colony optimization"""
        history = []
        grids = self.discretize_space()

        # Initialize pheromone matrix
        pheromone_shape = [self.grid_size] * self.dimensions
        pheromones = np.ones(pheromone_shape)

        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.iterations):
            solutions = []
            fitness_values = []

            # Generate solutions for each ant
            for ant in range(self.num_ants):
                solution = []
                for dim in range(self.dimensions):
                    lower, upper = self.bounds[dim]
                    # Random exploration with some bias towards pheromone
                    if random.random() < 0.7:  # exploration
                        value = random.uniform(lower, upper)
                    else:  # exploitation based on pheromones
                        grid_probs = np.sum(pheromones, axis=tuple(i for i in range(self.dimensions) if i != dim))
                        grid_probs = grid_probs / np.sum(grid_probs)
                        chosen_idx = np.random.choice(self.grid_size, p=grid_probs)
                        value = grids[dim][chosen_idx] + random.gauss(0, (upper-lower)/20)
                        value = np.clip(value, lower, upper)
                    solution.append(value)

                solutions.append(solution)
                fitness = self.objective_func(np.array(solution))
                fitness_values.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()

            # Update pheromones
            pheromones *= (1 - self.rho)  # evaporation

            # Deposit pheromones (simplified for continuous space)
            for sol, fitness in zip(solutions, fitness_values):
                if fitness < np.percentile(fitness_values, 50):  # only good solutions
                    grid_idx = self.get_grid_index(sol, grids)
                    pheromone_deposit = 1.0 / (1.0 + fitness)
                    pheromones[grid_idx] += pheromone_deposit

            avg_fitness = np.mean(fitness_values)
            history.append({
                'generation': iteration,
                'best_fitness': best_fitness,
                'best_solution': best_solution,
                'avg_fitness': avg_fitness
            })

        return history

class TeachingLearningBasedOptimization:
    """Teaching-Learning-Based Optimization implementation"""

    def __init__(self, objective_func: Callable, bounds: List[Tuple[float, float]],
                 population_size: int = 30, iterations: int = 100):
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.iterations = iterations
        self.dimensions = len(bounds)

    def initialize_population(self):
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for lower, upper in self.bounds:
                individual.append(random.uniform(lower, upper))
            population.append(individual)
        return np.array(population)

    def teaching_phase(self, population, teacher, mean):
        """Teaching phase of TLBO"""
        new_population = []
        teaching_factor = random.choice([1, 2])

        for student in population:
            # Generate new student
            new_student = student + random.random() * (teacher - teaching_factor * mean)

            # Apply bounds
            for i in range(self.dimensions):
                lower, upper = self.bounds[i]
                new_student[i] = np.clip(new_student[i], lower, upper)

            # Keep better solution
            if self.objective_func(new_student) < self.objective_func(student):
                new_population.append(new_student)
            else:
                new_population.append(student)

        return np.array(new_population)

    def learning_phase(self, population):
        """Learning phase of TLBO"""
        new_population = []

        for i, student in enumerate(population):
            # Select random different student
            j = random.choice([idx for idx in range(len(population)) if idx != i])
            other_student = population[j]

            # Generate new student
            if self.objective_func(student) < self.objective_func(other_student):
                new_student = student + random.random() * (student - other_student)
            else:
                new_student = student + random.random() * (other_student - student)

            # Apply bounds
            for k in range(self.dimensions):
                lower, upper = self.bounds[k]
                new_student[k] = np.clip(new_student[k], lower, upper)

            # Keep better solution
            if self.objective_func(new_student) < self.objective_func(student):
                new_population.append(new_student)
            else:
                new_population.append(student)

        return np.array(new_population)

    def optimize(self):
        """Run TLBO algorithm"""
        history = []
        population = self.initialize_population()

        for iteration in range(self.iterations):
            # Evaluate population
            fitness_values = [self.objective_func(ind) for ind in population]

            # Find teacher (best individual)
            teacher_idx = np.argmin(fitness_values)
            teacher = population[teacher_idx].copy()
            best_fitness = fitness_values[teacher_idx]

            # Calculate mean
            mean = np.mean(population, axis=0)

            # Teaching phase
            population = self.teaching_phase(population, teacher, mean)

            # Learning phase
            population = self.learning_phase(population)

            # Update fitness
            fitness_values = [self.objective_func(ind) for ind in population]
            avg_fitness = np.mean(fitness_values)

            history.append({
                'generation': iteration,
                'best_fitness': best_fitness,
                'best_solution': teacher.tolist(),
                'avg_fitness': avg_fitness
            })

        return history

class TabuSearch:
    """Tabu Search implementation"""

    def __init__(self, objective_func: Callable, bounds: List[Tuple[float, float]],
                 iterations: int = 100, tabu_list_size: int = 20, step_size: float = 0.1):
        self.objective_func = objective_func
        self.bounds = bounds
        self.iterations = iterations
        self.tabu_list_size = tabu_list_size
        self.step_size = step_size
        self.dimensions = len(bounds)
        self.tabu_list = []

    def generate_neighbors(self, solution):
        """Generate neighbor solutions"""
        neighbors = []
        for i in range(self.dimensions):
            for direction in [-1, 1]:
                neighbor = solution.copy()
                step = direction * self.step_size * (self.bounds[i][1] - self.bounds[i][0])
                neighbor[i] += step

                # Apply bounds
                lower, upper = self.bounds[i]
                neighbor[i] = np.clip(neighbor[i], lower, upper)
                neighbors.append(neighbor)

        return neighbors

    def is_tabu(self, solution):
        """Check if solution is in tabu list"""
        for tabu_sol in self.tabu_list:
            if np.allclose(solution, tabu_sol, atol=0.01):
                return True
        return False

    def update_tabu_list(self, solution):
        """Update tabu list"""
        self.tabu_list.append(solution.copy())
        if len(self.tabu_list) > self.tabu_list_size:
            self.tabu_list.pop(0)

    def optimize(self):
        """Run tabu search"""
        history = []

        # Initialize random solution
        current_solution = []
        for lower, upper in self.bounds:
            current_solution.append(random.uniform(lower, upper))
        current_solution = np.array(current_solution)

        best_solution = current_solution.copy()
        best_fitness = self.objective_func(current_solution)

        for iteration in range(self.iterations):
            neighbors = self.generate_neighbors(current_solution)

            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_fitness = float('inf')

            for neighbor in neighbors:
                if not self.is_tabu(neighbor):
                    fitness = self.objective_func(neighbor)
                    if fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = fitness

            # If no non-tabu neighbor found, accept best neighbor anyway
            if best_neighbor is None:
                neighbors_fitness = [(n, self.objective_func(n)) for n in neighbors]
                best_neighbor, best_neighbor_fitness = min(neighbors_fitness, key=lambda x: x[1])

            # Move to best neighbor
            current_solution = best_neighbor
            self.update_tabu_list(current_solution)

            # Update best solution
            if best_neighbor_fitness < best_fitness:
                best_fitness = best_neighbor_fitness
                best_solution = current_solution.copy()

            history.append({
                'generation': iteration,
                'best_fitness': best_fitness,
                'best_solution': best_solution.tolist(),
                'avg_fitness': best_neighbor_fitness  # Current solution fitness
            })

        return history

# Algorithm registry
ALGORITHMS = {
    'qaea': {
        'class': QuantumAdaptiveEvolution,
        'name': 'Quantum Adaptive Evolution (QAEA)',
        'description': 'Revolutionary hybrid algorithm combining DE, quantum operators, adaptive control, and elite archive'
    },
    'ga': {
        'class': GeneticAlgorithm,
        'name': 'Genetic Algorithm',
        'description': 'Evolutionary algorithm based on natural selection'
    },
    'pso': {
        'class': ParticleSwarmOptimization,
        'name': 'Particle Swarm Optimization',
        'description': 'Swarm intelligence algorithm inspired by bird flocking'
    },
    'aco': {
        'class': AntColonyOptimization,
        'name': 'Ant Colony Optimization',
        'description': 'Metaheuristic inspired by ant foraging behavior'
    },
    'tlbo': {
        'class': TeachingLearningBasedOptimization,
        'name': 'Teaching-Learning-Based Optimization',
        'description': 'Algorithm inspired by teaching-learning process'
    },
    'ts': {
        'class': TabuSearch,
        'name': 'Tabu Search',
        'description': 'Local search with memory to avoid cycling'
    }
}