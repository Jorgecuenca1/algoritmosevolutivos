"""
Meta-Heuristic Generator Module
================================
This module automatically generates new optimization algorithms using
Genetic Programming and component-based algorithm construction.
"""

import numpy as np
import random
from copy import deepcopy
import json


class AlgorithmComponent:
    """Base class for algorithm components (building blocks)"""

    def __init__(self, name, component_type, parameters=None):
        self.name = name
        self.type = component_type
        self.parameters = parameters or {}

    def execute(self, population, fitness_values, **kwargs):
        """Execute the component operation"""
        raise NotImplementedError

    def to_dict(self):
        return {
            'name': self.name,
            'type': self.type,
            'parameters': self.parameters
        }


class SelectionComponent(AlgorithmComponent):
    """Selection operator component"""

    def __init__(self, method='tournament', **params):
        super().__init__(f"Selection_{method}", "selection", params)
        self.method = method

    def execute(self, population, fitness_values, n_select=None, **kwargs):
        n_select = n_select or len(population) // 2

        if self.method == 'tournament':
            return self._tournament_selection(population, fitness_values, n_select)
        elif self.method == 'roulette':
            return self._roulette_selection(population, fitness_values, n_select)
        elif self.method == 'rank':
            return self._rank_selection(population, fitness_values, n_select)
        elif self.method == 'best':
            return self._best_selection(population, fitness_values, n_select)

    def _tournament_selection(self, population, fitness_values, n_select):
        tournament_size = self.parameters.get('tournament_size', 3)
        selected = []

        for _ in range(n_select):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return selected

    def _roulette_selection(self, population, fitness_values, n_select):
        # Convert to maximization (inverse fitness)
        max_fitness = max(fitness_values)
        weights = [max_fitness - f + 1 for f in fitness_values]
        total = sum(weights)
        probabilities = [w / total for w in weights]

        selected_idx = np.random.choice(len(population), n_select, p=probabilities)
        return [population[i].copy() for i in selected_idx]

    def _rank_selection(self, population, fitness_values, n_select):
        sorted_idx = np.argsort(fitness_values)
        ranks = np.arange(len(population), 0, -1)
        probabilities = ranks / sum(ranks)

        selected_idx = np.random.choice(len(population), n_select, p=probabilities)
        return [population[sorted_idx[i]].copy() for i in selected_idx]

    def _best_selection(self, population, fitness_values, n_select):
        sorted_idx = np.argsort(fitness_values)
        return [population[i].copy() for i in sorted_idx[:n_select]]


class CrossoverComponent(AlgorithmComponent):
    """Crossover/recombination operator component"""

    def __init__(self, method='single_point', **params):
        super().__init__(f"Crossover_{method}", "crossover", params)
        self.method = method

    def execute(self, parents, **kwargs):
        if len(parents) < 2:
            return parents

        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]

            if self.method == 'single_point':
                child1, child2 = self._single_point_crossover(parent1, parent2)
            elif self.method == 'uniform':
                child1, child2 = self._uniform_crossover(parent1, parent2)
            elif self.method == 'arithmetic':
                child1, child2 = self._arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspring.extend([child1, child2])

        return offspring

    def _single_point_crossover(self, parent1, parent2):
        point = len(parent1) // 2
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _uniform_crossover(self, parent1, parent2):
        mask = np.random.rand(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _arithmetic_crossover(self, parent1, parent2):
        alpha = self.parameters.get('alpha', 0.5)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2


class MutationComponent(AlgorithmComponent):
    """Mutation operator component"""

    def __init__(self, method='gaussian', **params):
        super().__init__(f"Mutation_{method}", "mutation", params)
        self.method = method

    def execute(self, population, bounds, **kwargs):
        mutation_rate = self.parameters.get('rate', 0.1)
        mutated = []

        for individual in population:
            if np.random.rand() < mutation_rate:
                if self.method == 'gaussian':
                    mutated_ind = self._gaussian_mutation(individual, bounds)
                elif self.method == 'uniform':
                    mutated_ind = self._uniform_mutation(individual, bounds)
                elif self.method == 'boundary':
                    mutated_ind = self._boundary_mutation(individual, bounds)
                else:
                    mutated_ind = individual.copy()

                mutated.append(mutated_ind)
            else:
                mutated.append(individual.copy())

        return mutated

    def _gaussian_mutation(self, individual, bounds):
        sigma = self.parameters.get('sigma', 0.1)
        mutated = individual + np.random.normal(0, sigma, len(individual))
        return np.clip(mutated, bounds[0], bounds[1])

    def _uniform_mutation(self, individual, bounds):
        idx = np.random.randint(0, len(individual))
        mutated = individual.copy()
        mutated[idx] = np.random.uniform(bounds[0], bounds[1])
        return mutated

    def _boundary_mutation(self, individual, bounds):
        idx = np.random.randint(0, len(individual))
        mutated = individual.copy()
        mutated[idx] = random.choice([bounds[0], bounds[1]])
        return mutated


class LocalSearchComponent(AlgorithmComponent):
    """Local search/improvement component"""

    def __init__(self, method='hill_climbing', **params):
        super().__init__(f"LocalSearch_{method}", "local_search", params)
        self.method = method

    def execute(self, population, fitness_values, objective_func, bounds, **kwargs):
        improved = []
        n_improve = self.parameters.get('n_improve', min(5, len(population)))

        # Apply local search to best individuals
        sorted_idx = np.argsort(fitness_values)[:n_improve]

        for i in range(len(population)):
            if i in sorted_idx:
                if self.method == 'hill_climbing':
                    improved_ind = self._hill_climbing(population[i], objective_func, bounds)
                else:
                    improved_ind = population[i].copy()
                improved.append(improved_ind)
            else:
                improved.append(population[i].copy())

        return improved

    def _hill_climbing(self, individual, objective_func, bounds):
        best = individual.copy()
        best_fitness = objective_func(best)
        step_size = self.parameters.get('step_size', 0.1)
        max_iterations = self.parameters.get('max_iterations', 10)

        for _ in range(max_iterations):
            # Try random neighbor
            neighbor = best + np.random.normal(0, step_size, len(best))
            neighbor = np.clip(neighbor, bounds[0], bounds[1])

            neighbor_fitness = objective_func(neighbor)
            if neighbor_fitness < best_fitness:
                best = neighbor
                best_fitness = neighbor_fitness

        return best


class ReplacementComponent(AlgorithmComponent):
    """Replacement/survival selection component"""

    def __init__(self, method='generational', **params):
        super().__init__(f"Replacement_{method}", "replacement", params)
        self.method = method

    def execute(self, population, offspring, fitness_pop, fitness_off, **kwargs):
        if self.method == 'generational':
            return offspring, fitness_off
        elif self.method == 'elitist':
            return self._elitist_replacement(population, offspring, fitness_pop, fitness_off)
        elif self.method == 'steady_state':
            return self._steady_state_replacement(population, offspring, fitness_pop, fitness_off)

    def _elitist_replacement(self, population, offspring, fitness_pop, fitness_off):
        n_elites = self.parameters.get('n_elites', 2)

        # Get best from current population
        elite_idx = np.argsort(fitness_pop)[:n_elites]
        elites = [population[i] for i in elite_idx]
        elite_fitness = [fitness_pop[i] for i in elite_idx]

        # Replace worst offspring with elites
        new_pop = offspring[:-n_elites] + elites
        new_fitness = fitness_off[:-n_elites] + elite_fitness

        return new_pop, new_fitness

    def _steady_state_replacement(self, population, offspring, fitness_pop, fitness_off):
        # Combine and select best
        combined = population + offspring
        combined_fitness = fitness_pop + fitness_off

        sorted_idx = np.argsort(combined_fitness)[:len(population)]
        new_pop = [combined[i] for i in sorted_idx]
        new_fitness = [combined_fitness[i] for i in sorted_idx]

        return new_pop, new_fitness


class GeneratedAlgorithm:
    """
    A complete algorithm generated from components
    """

    def __init__(self, components=None, population_size=50):
        self.components = components or []
        self.population_size = population_size
        self.name = "GeneratedAlgorithm"
        self.generation = 0

    def add_component(self, component):
        """Add a component to the algorithm"""
        self.components.append(component)

    def optimize(self, objective_func, bounds, iterations=100, dimensions=2):
        """
        Execute the generated algorithm

        Returns history of best fitness values
        """
        # Initialize population
        population = [
            np.random.uniform(bounds[0], bounds[1], dimensions)
            for _ in range(self.population_size)
        ]

        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_solution': None,
            'iterations': []
        }

        for iteration in range(iterations):
            # Evaluate fitness
            fitness_values = [objective_func(ind) for ind in population]

            # Track history
            best_idx = np.argmin(fitness_values)
            history['best_fitness'].append(fitness_values[best_idx])
            history['avg_fitness'].append(np.mean(fitness_values))
            history['iterations'].append(iteration)

            if iteration == iterations - 1:
                history['best_solution'] = population[best_idx]

            # Execute algorithm components in sequence
            current_pop = population
            current_fitness = fitness_values

            for component in self.components:
                if component.type == 'selection':
                    selected = component.execute(current_pop, current_fitness)
                    current_pop = selected

                elif component.type == 'crossover':
                    offspring = component.execute(current_pop)
                    current_pop = offspring

                elif component.type == 'mutation':
                    mutated = component.execute(current_pop, bounds)
                    current_pop = mutated

                elif component.type == 'local_search':
                    # Re-evaluate if needed
                    if len(current_fitness) != len(current_pop):
                        current_fitness = [objective_func(ind) for ind in current_pop]

                    improved = component.execute(
                        current_pop, current_fitness, objective_func, bounds
                    )
                    current_pop = improved

                elif component.type == 'replacement':
                    # Evaluate offspring
                    offspring_fitness = [objective_func(ind) for ind in current_pop]
                    new_pop, new_fitness = component.execute(
                        population, current_pop, fitness_values, offspring_fitness
                    )
                    population = new_pop
                    break  # Replacement is typically the last step

            # If no replacement component, just use the current population
            if not any(c.type == 'replacement' for c in self.components):
                population = current_pop

        return history

    def to_dict(self):
        """Export algorithm structure"""
        return {
            'name': self.name,
            'population_size': self.population_size,
            'components': [comp.to_dict() for comp in self.components]
        }

    def get_code(self):
        """Generate Python code for this algorithm"""
        code = f"class {self.name}:\n"
        code += f"    def __init__(self, objective_func, bounds, iterations=100, population_size={self.population_size}):\n"
        code += "        self.objective_func = objective_func\n"
        code += "        self.bounds = bounds\n"
        code += "        self.iterations = iterations\n"
        code += f"        self.population_size = population_size\n\n"
        code += "    def optimize(self):\n"
        code += "        # Algorithm implementation\n"

        for i, comp in enumerate(self.components):
            code += f"        # Step {i+1}: {comp.name}\n"

        return code


class MetaHeuristicGenerator:
    """
    Generates new meta-heuristic algorithms using genetic programming
    """

    def __init__(self):
        self.component_library = self._build_component_library()
        self.algorithm_pool = []

    def _build_component_library(self):
        """Build library of available components"""
        return {
            'selection': [
                SelectionComponent('tournament', tournament_size=3),
                SelectionComponent('roulette'),
                SelectionComponent('rank'),
                SelectionComponent('best'),
            ],
            'crossover': [
                CrossoverComponent('single_point'),
                CrossoverComponent('uniform'),
                CrossoverComponent('arithmetic', alpha=0.5),
            ],
            'mutation': [
                MutationComponent('gaussian', rate=0.1, sigma=0.1),
                MutationComponent('gaussian', rate=0.2, sigma=0.2),
                MutationComponent('uniform', rate=0.15),
                MutationComponent('boundary', rate=0.05),
            ],
            'local_search': [
                LocalSearchComponent('hill_climbing', n_improve=3, step_size=0.1, max_iterations=10),
            ],
            'replacement': [
                ReplacementComponent('generational'),
                ReplacementComponent('elitist', n_elites=2),
                ReplacementComponent('steady_state'),
            ]
        }

    def generate_random_algorithm(self, name=None):
        """Generate a random algorithm from components"""
        algorithm = GeneratedAlgorithm(population_size=random.choice([30, 50, 100]))

        if name:
            algorithm.name = name
        else:
            algorithm.name = f"Generated_Algorithm_{len(self.algorithm_pool)}"

        # Standard evolutionary algorithm structure
        # 1. Selection
        selection = random.choice(self.component_library['selection'])
        algorithm.add_component(deepcopy(selection))

        # 2. Crossover (optional, 70% chance)
        if random.random() < 0.7:
            crossover = random.choice(self.component_library['crossover'])
            algorithm.add_component(deepcopy(crossover))

        # 3. Mutation
        mutation = random.choice(self.component_library['mutation'])
        algorithm.add_component(deepcopy(mutation))

        # 4. Local search (optional, 30% chance)
        if random.random() < 0.3:
            local_search = random.choice(self.component_library['local_search'])
            algorithm.add_component(deepcopy(local_search))

        # 5. Replacement
        replacement = random.choice(self.component_library['replacement'])
        algorithm.add_component(deepcopy(replacement))

        return algorithm

    def evolve_algorithms(self, objective_func, bounds, n_algorithms=10,
                         n_generations=5, test_iterations=50):
        """
        Use genetic programming to evolve better algorithms

        This creates a population of algorithms and evolves them to find
        better performing algorithm configurations
        """
        # Generate initial population of algorithms
        population = [
            self.generate_random_algorithm(f"Evolved_Gen0_Alg{i}")
            for i in range(n_algorithms)
        ]

        best_algorithm = None
        best_performance = float('inf')

        for generation in range(n_generations):
            # Evaluate each algorithm
            performances = []

            for algo in population:
                try:
                    history = algo.optimize(objective_func, bounds, iterations=test_iterations)
                    # Performance = final best fitness
                    performance = history['best_fitness'][-1]
                    performances.append(performance)

                    if performance < best_performance:
                        best_performance = performance
                        best_algorithm = deepcopy(algo)

                except Exception as e:
                    # If algorithm fails, give it worst performance
                    performances.append(float('inf'))

            # Select best algorithms
            sorted_idx = np.argsort(performances)
            survivors = [population[i] for i in sorted_idx[:n_algorithms // 2]]

            # Generate offspring through variation
            offspring = []
            for i in range(n_algorithms // 2):
                parent = random.choice(survivors)
                child = self._mutate_algorithm(parent, generation)
                offspring.append(child)

            population = survivors + offspring

        self.algorithm_pool.append(best_algorithm)
        return best_algorithm

    def _mutate_algorithm(self, algorithm, generation):
        """Create a variation of an algorithm by changing components"""
        new_algo = GeneratedAlgorithm(population_size=algorithm.population_size)
        new_algo.name = f"Evolved_Gen{generation+1}_Mutant"

        for component in algorithm.components:
            # 70% chance to keep component, 30% chance to replace
            if random.random() < 0.7:
                new_algo.add_component(deepcopy(component))
            else:
                # Replace with random component of same type
                new_component = random.choice(self.component_library[component.type])
                new_algo.add_component(deepcopy(new_component))

        # 20% chance to add an extra component
        if random.random() < 0.2 and len(new_algo.components) < 6:
            comp_type = random.choice(['local_search', 'mutation'])
            new_component = random.choice(self.component_library[comp_type])
            # Insert before replacement
            new_algo.components.insert(-1, deepcopy(new_component))

        return new_algo

    def create_custom_algorithm(self, component_spec):
        """
        Create algorithm from user specification

        component_spec: list of dicts like:
        [
            {'type': 'selection', 'method': 'tournament', 'params': {...}},
            {'type': 'crossover', 'method': 'uniform'},
            ...
        ]
        """
        algorithm = GeneratedAlgorithm()

        for spec in component_spec:
            comp_type = spec['type']
            method = spec.get('method', None)
            params = spec.get('params', {})

            if comp_type == 'selection':
                component = SelectionComponent(method, **params)
            elif comp_type == 'crossover':
                component = CrossoverComponent(method, **params)
            elif comp_type == 'mutation':
                component = MutationComponent(method, **params)
            elif comp_type == 'local_search':
                component = LocalSearchComponent(method, **params)
            elif comp_type == 'replacement':
                component = ReplacementComponent(method, **params)
            else:
                continue

            algorithm.add_component(component)

        return algorithm
