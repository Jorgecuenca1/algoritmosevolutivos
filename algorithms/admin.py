from django.contrib import admin
from .models import CustomProblem, GeneratedAlgorithm, OptimizationRun, AlgorithmRecommendation


@admin.register(CustomProblem)
class CustomProblemAdmin(admin.ModelAdmin):
    list_display = ['name', 'problem_type', 'dimensions', 'created_by', 'created_at', 'is_public', 'usage_count']
    list_filter = ['problem_type', 'is_public', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'usage_count']


@admin.register(GeneratedAlgorithm)
class GeneratedAlgorithmAdmin(admin.ModelAdmin):
    list_display = ['name', 'population_size', 'performance_score', 'is_evolved', 'generation', 'created_at']
    list_filter = ['is_evolved', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']


@admin.register(OptimizationRun)
class OptimizationRunAdmin(admin.ModelAdmin):
    list_display = ['algorithm', 'problem_name', 'best_fitness', 'execution_time', 'created_at', 'user']
    list_filter = ['algorithm', 'created_at']
    search_fields = ['problem_name']
    readonly_fields = ['created_at']


@admin.register(AlgorithmRecommendation)
class AlgorithmRecommendationAdmin(admin.ModelAdmin):
    list_display = ['problem_name', 'recommended_algorithm', 'confidence', 'created_at']
    list_filter = ['recommended_algorithm', 'created_at']
    search_fields = ['problem_name', 'reason']
    readonly_fields = ['created_at']
