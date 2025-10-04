"""
Hyper-Heuristic Module
======================
This module implements intelligent algorithm selection and recommendation
based on problem characteristics using machine learning and heuristic rules.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json


class ProblemAnalyzer:
    """Analyzes problem characteristics to help select appropriate algorithms"""

    def __init__(self):
        self.features = {}

    def analyze_continuous_function(self, objective_func, bounds, n_samples=100):
        """
        Analyze characteristics of a continuous optimization function
        Returns feature vector describing the problem
        """
        # Sample random points in the search space
        dim = 2  # For now, focus on 2D problems
        lower, upper = bounds[0], bounds[1]

        samples = []
        values = []

        for _ in range(n_samples):
            point = np.random.uniform(lower, upper, dim)
            value = objective_func(point)
            samples.append(point)
            values.append(value)

        samples = np.array(samples)
        values = np.array(values)

        # Extract features
        features = {
            # Statistical features
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'cv': np.std(values) / (np.abs(np.mean(values)) + 1e-10),  # Coefficient of variation

            # Landscape features
            'ruggedness': self._calculate_ruggedness(samples, values),
            'modality': self._estimate_modality(samples, values),
            'gradient_variance': self._calculate_gradient_variance(samples, values),

            # Search space features
            'dimension': dim,
            'domain_size': upper - lower,

            # Other features
            'symmetry': self._estimate_symmetry(samples, values),
        }

        self.features = features
        return features

    def _calculate_ruggedness(self, samples, values):
        """Calculate landscape ruggedness (autocorrelation of fitness)"""
        if len(values) < 2:
            return 0

        # Calculate differences between nearby points
        diffs = []
        for i in range(len(samples) - 1):
            for j in range(i + 1, min(i + 10, len(samples))):
                dist = np.linalg.norm(samples[i] - samples[j])
                if dist < 1.0:  # Consider only nearby points
                    value_diff = abs(values[i] - values[j])
                    diffs.append(value_diff)

        return np.std(diffs) if diffs else 0

    def _estimate_modality(self, samples, values):
        """Estimate number of modes (local optima) in the landscape"""
        # Simple heuristic: count peaks in sorted values
        sorted_values = np.sort(values)
        peaks = 0

        for i in range(1, len(sorted_values) - 1):
            if sorted_values[i] < sorted_values[i-1] and sorted_values[i] < sorted_values[i+1]:
                peaks += 1

        return peaks

    def _calculate_gradient_variance(self, samples, values):
        """Calculate variance of approximate gradients"""
        if len(samples) < 2:
            return 0

        gradients = []
        for i in range(len(samples) - 1):
            for j in range(i + 1, min(i + 5, len(samples))):
                delta_x = samples[j] - samples[i]
                delta_f = values[j] - values[i]
                dist = np.linalg.norm(delta_x)
                if dist > 1e-6:
                    gradient = delta_f / dist
                    gradients.append(gradient)

        return np.var(gradients) if gradients else 0

    def _estimate_symmetry(self, samples, values):
        """Estimate symmetry of the function around the origin"""
        # Compare values of points and their negatives
        symmetry_score = 0
        count = 0

        for i, point in enumerate(samples):
            # Find closest point to -point
            neg_point = -point
            distances = np.linalg.norm(samples - neg_point, axis=1)
            closest_idx = np.argmin(distances)

            if distances[closest_idx] < 1.0:
                diff = abs(values[i] - values[closest_idx])
                symmetry_score += diff
                count += 1

        return symmetry_score / count if count > 0 else 0


class AlgorithmSelector:
    """Selects the most appropriate algorithm for a given problem"""

    def __init__(self):
        self.analyzer = ProblemAnalyzer()
        self.knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Build rule-based knowledge for algorithm selection"""
        return {
            'rules': [
                {
                    'condition': lambda f: f['modality'] < 5 and f['ruggedness'] < 10,
                    'algorithm': 'ga',
                    'reason': 'Low modality and smooth landscape favor Genetic Algorithm'
                },
                {
                    'condition': lambda f: f['ruggedness'] < 5,
                    'algorithm': 'pso',
                    'reason': 'Smooth landscape is ideal for Particle Swarm Optimization'
                },
                {
                    'condition': lambda f: f['modality'] > 10,
                    'algorithm': 'aco',
                    'reason': 'High modality suggests multiple optima, ACO explores well'
                },
                {
                    'condition': lambda f: f['cv'] < 0.5,
                    'algorithm': 'tlbo',
                    'reason': 'Low variance suggests TLBO can efficiently exploit the landscape'
                },
                {
                    'condition': lambda f: f['dimension'] <= 2,
                    'algorithm': 'ts',
                    'reason': 'Low dimension allows Tabu Search to be very effective'
                },
            ],
            'algorithm_characteristics': {
                'ga': {
                    'good_for': ['multimodal', 'discrete', 'medium_dimension'],
                    'poor_for': ['smooth', 'continuous'],
                    'exploration': 0.7,
                    'exploitation': 0.6
                },
                'pso': {
                    'good_for': ['smooth', 'continuous', 'unimodal'],
                    'poor_for': ['highly_multimodal', 'discrete'],
                    'exploration': 0.8,
                    'exploitation': 0.7
                },
                'aco': {
                    'good_for': ['combinatorial', 'discrete', 'multimodal'],
                    'poor_for': ['continuous', 'smooth'],
                    'exploration': 0.9,
                    'exploitation': 0.5
                },
                'tlbo': {
                    'good_for': ['continuous', 'constrained'],
                    'poor_for': ['discrete'],
                    'exploration': 0.6,
                    'exploitation': 0.8
                },
                'ts': {
                    'good_for': ['small_dimension', 'local_search'],
                    'poor_for': ['high_dimension'],
                    'exploration': 0.5,
                    'exploitation': 0.9
                }
            }
        }

    def recommend_algorithm(self, objective_func=None, bounds=None, problem_type='continuous',
                           features=None):
        """
        Recommend the best algorithm for the given problem

        Parameters:
        -----------
        objective_func : callable, optional
            The objective function to analyze
        bounds : list, optional
            Search space bounds
        problem_type : str
            Type of problem ('continuous' or 'combinatorial')
        features : dict, optional
            Pre-computed problem features

        Returns:
        --------
        dict : Recommendation with algorithm, confidence, and reasoning
        """
        if features is None and objective_func is not None and bounds is not None:
            features = self.analyzer.analyze_continuous_function(objective_func, bounds)

        if features is None:
            # Default recommendation
            return {
                'algorithm': 'ga',
                'confidence': 0.5,
                'reason': 'Default recommendation: Genetic Algorithm is versatile',
                'alternatives': ['pso', 'aco']
            }

        # Apply rule-based system
        recommendations = []

        for rule in self.knowledge_base['rules']:
            try:
                if rule['condition'](features):
                    recommendations.append({
                        'algorithm': rule['algorithm'],
                        'reason': rule['reason'],
                        'confidence': 0.8
                    })
            except:
                continue

        if recommendations:
            # Return the first matching rule
            best = recommendations[0]
            alternatives = [r['algorithm'] for r in recommendations[1:3]]
            best['alternatives'] = alternatives if alternatives else ['ga', 'pso']
            return best

        # Fallback recommendation based on problem type
        if problem_type == 'combinatorial':
            return {
                'algorithm': 'aco',
                'confidence': 0.6,
                'reason': 'ACO generally performs well on combinatorial problems',
                'alternatives': ['ga', 'ts']
            }
        else:
            return {
                'algorithm': 'pso',
                'confidence': 0.6,
                'reason': 'PSO is a good general-purpose optimizer for continuous problems',
                'alternatives': ['ga', 'tlbo']
            }

    def rank_algorithms(self, objective_func=None, bounds=None, features=None):
        """
        Rank all available algorithms for the given problem

        Returns:
        --------
        list : Ranked list of algorithms with scores
        """
        if features is None and objective_func is not None:
            features = self.analyzer.analyze_continuous_function(objective_func, bounds)

        scores = {}
        algos = ['ga', 'pso', 'aco', 'tlbo', 'ts']

        for algo in algos:
            score = self._calculate_algorithm_score(algo, features)
            scores[algo] = score

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                'algorithm': algo,
                'score': score,
                'characteristics': self.knowledge_base['algorithm_characteristics'].get(algo, {})
            }
            for algo, score in ranked
        ]

    def _calculate_algorithm_score(self, algorithm, features):
        """Calculate suitability score for an algorithm given problem features"""
        if features is None:
            return 0.5

        char = self.knowledge_base['algorithm_characteristics'].get(algorithm, {})

        score = 0.5  # Base score

        # Adjust based on modality
        if features.get('modality', 0) > 10:
            if 'multimodal' in char.get('good_for', []):
                score += 0.2
            if 'smooth' in char.get('good_for', []):
                score -= 0.2

        # Adjust based on ruggedness
        if features.get('ruggedness', 0) < 5:
            if 'smooth' in char.get('good_for', []):
                score += 0.2

        # Adjust based on dimension
        if features.get('dimension', 2) <= 2:
            if 'small_dimension' in char.get('good_for', []):
                score += 0.15

        return max(0, min(1, score))


class HyperHeuristic:
    """
    Main hyper-heuristic class that coordinates algorithm selection,
    adaptation, and ensemble methods
    """

    def __init__(self):
        self.selector = AlgorithmSelector()
        self.performance_history = {}

    def select_algorithm(self, problem_data):
        """
        Select the best algorithm for a given problem

        Parameters:
        -----------
        problem_data : dict
            Dictionary containing problem information:
            - 'type': 'continuous' or 'combinatorial'
            - 'function': objective function (optional)
            - 'bounds': search space bounds (optional)
            - 'features': pre-computed features (optional)

        Returns:
        --------
        dict : Algorithm recommendation
        """
        return self.selector.recommend_algorithm(
            objective_func=problem_data.get('function'),
            bounds=problem_data.get('bounds'),
            problem_type=problem_data.get('type', 'continuous'),
            features=problem_data.get('features')
        )

    def rank_algorithms(self, problem_data):
        """Rank all algorithms for the given problem"""
        return self.selector.rank_algorithms(
            objective_func=problem_data.get('function'),
            bounds=problem_data.get('bounds'),
            features=problem_data.get('features')
        )

    def update_performance(self, algorithm, problem_type, performance):
        """Update performance history for learning"""
        key = f"{algorithm}_{problem_type}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        self.performance_history[key].append(performance)

    def get_best_performing_algorithm(self, problem_type):
        """Get historically best performing algorithm for a problem type"""
        best_algo = None
        best_avg = float('inf')

        for key, performances in self.performance_history.items():
            if problem_type in key:
                avg = np.mean(performances)
                if avg < best_avg:
                    best_avg = avg
                    best_algo = key.split('_')[0]

        return best_algo if best_algo else 'ga'
