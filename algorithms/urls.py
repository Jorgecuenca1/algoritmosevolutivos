from django.urls import path
from . import views

app_name = 'algorithms'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('optimize/', views.optimize, name='optimize'),
    path('results/<str:session_id>/', views.results, name='results'),

    # Hyper-heuristic
    path('hyper-heuristic/', views.hyper_heuristic_page, name='hyper_heuristic'),
    path('api/recommend-algorithm/', views.recommend_algorithm, name='recommend_algorithm'),

    # Meta-heuristic generator
    path('meta-heuristic/', views.meta_heuristic_page, name='meta_heuristic'),
    path('api/generate-algorithm/', views.generate_algorithm, name='generate_algorithm'),
    path('api/evolve-algorithms/', views.evolve_algorithms, name='evolve_algorithms'),
    path('api/test-generated/<int:algorithm_id>/', views.test_generated_algorithm, name='test_generated_algorithm'),

    # Custom problems
    path('custom-problems/', views.custom_problems_page, name='custom_problems'),
    path('api/create-problem/', views.create_custom_problem, name='create_custom_problem'),
    path('api/test-problem/<int:problem_id>/', views.test_custom_problem, name='test_custom_problem'),
    path('api/template/<str:template_name>/', views.get_problem_template, name='get_problem_template'),

    # Flight Optimizer
    path('flight-optimizer/', views.flight_optimizer_page, name='flight_optimizer'),
    path('api/optimize-flights/', views.optimize_flights, name='optimize_flights'),
    path('flight-results/<str:session_id>/', views.flight_results, name='flight_results'),
]