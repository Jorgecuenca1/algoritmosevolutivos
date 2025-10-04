from django.db import models
from django.contrib.auth.models import User


class CustomProblem(models.Model):
    """Store user-defined optimization problems"""
    PROBLEM_TYPES = [
        ('continuous', 'Continuous Optimization'),
        ('combinatorial', 'Combinatorial Optimization'),
    ]

    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    problem_type = models.CharField(max_length=20, choices=PROBLEM_TYPES)
    function_code = models.TextField(help_text="Python function code")
    bounds_lower = models.FloatField(null=True, blank=True)
    bounds_upper = models.FloatField(null=True, blank=True)
    dimensions = models.IntegerField(default=2)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_public = models.BooleanField(default=False)
    usage_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class GeneratedAlgorithm(models.Model):
    """Store auto-generated algorithms"""
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    components = models.JSONField(help_text="Algorithm components and parameters")
    population_size = models.IntegerField(default=50)
    performance_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_evolved = models.BooleanField(default=False, help_text="Was this algorithm evolved?")
    generation = models.IntegerField(default=0, help_text="Generation number if evolved")
    parent_algorithm = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-performance_score', '-created_at']

    def __str__(self):
        return self.name


class OptimizationRun(models.Model):
    """Store optimization run results"""
    ALGORITHM_CHOICES = [
        ('ga', 'Genetic Algorithm'),
        ('pso', 'Particle Swarm Optimization'),
        ('aco', 'Ant Colony Optimization'),
        ('tlbo', 'Teaching-Learning-Based Optimization'),
        ('ts', 'Tabu Search'),
        ('generated', 'Generated Algorithm'),
    ]

    algorithm = models.CharField(max_length=20, choices=ALGORITHM_CHOICES)
    generated_algorithm = models.ForeignKey(GeneratedAlgorithm, on_delete=models.SET_NULL,
                                           null=True, blank=True)
    problem_name = models.CharField(max_length=200)
    custom_problem = models.ForeignKey(CustomProblem, on_delete=models.SET_NULL,
                                      null=True, blank=True)
    iterations = models.IntegerField(default=100)
    population_size = models.IntegerField(default=50)
    best_fitness = models.FloatField()
    best_solution = models.JSONField()
    convergence_history = models.JSONField()
    execution_time = models.FloatField(help_text="Execution time in seconds")
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.algorithm} on {self.problem_name} - {self.created_at}"


class AlgorithmRecommendation(models.Model):
    """Store algorithm recommendations from hyper-heuristic"""
    problem_name = models.CharField(max_length=200)
    custom_problem = models.ForeignKey(CustomProblem, on_delete=models.CASCADE,
                                      null=True, blank=True)
    recommended_algorithm = models.CharField(max_length=20)
    confidence = models.FloatField()
    reason = models.TextField()
    problem_features = models.JSONField(help_text="Extracted problem features")
    alternative_algorithms = models.JSONField(help_text="List of alternative algorithms")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Recommendation for {self.problem_name}: {self.recommended_algorithm}"
