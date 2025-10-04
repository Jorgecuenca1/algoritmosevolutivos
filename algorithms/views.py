import json
import uuid
import time
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.contrib import messages
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.offline import plot
import plotly.io as pio

from .optimization_functions import OPTIMIZATION_FUNCTIONS
from .evolutionary_algorithms import ALGORITHMS
from .hyper_heuristic import HyperHeuristic
from .meta_heuristic_generator import MetaHeuristicGenerator
from .custom_problem_loader import ProblemLibrary, ProblemValidator, ProblemTemplateGenerator
from .models import CustomProblem, GeneratedAlgorithm, OptimizationRun, AlgorithmRecommendation
from .flight_optimizer import FlightOptimizationProblem, get_flight_optimization_function
from .flight_evolutionary import GeneticAlgorithmFlights, ParticleSwarmFlights


def index(request):
    """Main page with algorithm selection"""
    context = {
        'algorithms': ALGORITHMS,
        'functions': OPTIMIZATION_FUNCTIONS,
    }
    return render(request, 'algorithms/index.html', context)


@csrf_exempt
def optimize(request):
    """Run optimization algorithm"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            algorithm_key = data.get('algorithm')
            function_key = data.get('function')
            iterations = int(data.get('iterations', 100))

            # Get algorithm and function
            algorithm_info = ALGORITHMS[algorithm_key]
            function_info = OPTIMIZATION_FUNCTIONS[function_key]

            # Set up bounds for 2D optimization
            bounds = [function_info['domain'], function_info['domain']]

            # Initialize algorithm
            algorithm_class = algorithm_info['class']
            if algorithm_key == 'qaea':
                algorithm = algorithm_class(
                    objective_func=function_info['function'],
                    bounds=bounds,
                    iterations=iterations,
                    population_size=50
                )
            elif algorithm_key == 'ga':
                algorithm = algorithm_class(
                    objective_func=function_info['function'],
                    bounds=bounds,
                    generations=iterations
                )
            elif algorithm_key == 'pso':
                algorithm = algorithm_class(
                    objective_func=function_info['function'],
                    bounds=bounds,
                    iterations=iterations
                )
            elif algorithm_key == 'aco':
                algorithm = algorithm_class(
                    objective_func=function_info['function'],
                    bounds=bounds,
                    iterations=iterations
                )
            elif algorithm_key == 'tlbo':
                algorithm = algorithm_class(
                    objective_func=function_info['function'],
                    bounds=bounds,
                    iterations=iterations
                )
            elif algorithm_key == 'ts':
                algorithm = algorithm_class(
                    objective_func=function_info['function'],
                    bounds=bounds,
                    iterations=iterations
                )

            # Run optimization
            history = algorithm.optimize()

            # Generate session ID and store results
            session_id = str(uuid.uuid4())
            results = {
                'algorithm': algorithm_info,
                'function': function_info,
                'history': history,
                'session_id': session_id
            }

            # Store in cache (expires in 1 hour)
            cache.set(f'optimization_{session_id}', results, 3600)

            return JsonResponse({
                'success': True,
                'session_id': session_id,
                'best_solution': history[-1]['best_solution'],
                'best_fitness': history[-1]['best_fitness']
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


def results(request, session_id):
    """Display optimization results"""
    results = cache.get(f'optimization_{session_id}')

    if not results:
        return render(request, 'algorithms/error.html', {
            'error': 'Results not found or expired'
        })

    history = results['history']
    algorithm_info = results['algorithm']
    function_info = results['function']

    # Create convergence plot
    # Support both 'generation' and 'iteration' keys
    generations = [h.get('generation', h.get('iteration', i)) for i, h in enumerate(history)]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]

    trace1 = go.Scatter(
        x=generations,
        y=best_fitness,
        mode='lines',
        name='Best Fitness',
        line=dict(color='blue')
    )

    trace2 = go.Scatter(
        x=generations,
        y=avg_fitness,
        mode='lines',
        name='Average Fitness',
        line=dict(color='red', dash='dash')
    )

    layout = go.Layout(
        title=f'{algorithm_info["name"]} - {function_info["name"]} Convergence',
        xaxis=dict(title='Iteration/Generation'),
        yaxis=dict(title='Fitness Value'),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    convergence_plot = plot(fig, output_type='div', include_plotlyjs=False)

    # Create solution trajectory plot (for 2D functions)
    solution_x = [h['best_solution'][0] for h in history]
    solution_y = [h['best_solution'][1] for h in history]

    trace_trajectory = go.Scatter(
        x=solution_x,
        y=solution_y,
        mode='lines+markers',
        name='Solution Trajectory',
        line=dict(color='green'),
        marker=dict(size=4)
    )

    # Add global optimum point if available
    traces_data = [trace_trajectory]
    optimum = function_info.get('global_optimum')
    if optimum is not None and len(optimum) >= 2:
        trace_optimum = go.Scatter(
            x=[optimum[0]],
            y=[optimum[1]],
            mode='markers',
            name='Global Optimum',
            marker=dict(color='red', size=10, symbol='star')
        )
        traces_data.append(trace_optimum)

    layout_trajectory = go.Layout(
        title=f'{algorithm_info["name"]} - Solution Trajectory',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        hovermode='closest'
    )

    fig_trajectory = go.Figure(data=traces_data, layout=layout_trajectory)
    trajectory_plot = plot(fig_trajectory, output_type='div', include_plotlyjs=False)

    # Calculate statistics
    final_result = history[-1]
    improvement = history[0]['best_fitness'] - final_result['best_fitness']
    improvement_percent = (improvement / history[0]['best_fitness']) * 100 if history[0]['best_fitness'] != 0 else 0

    context = {
        'results': results,
        'convergence_plot': convergence_plot,
        'trajectory_plot': trajectory_plot,
        'final_result': final_result,
        'improvement': improvement,
        'improvement_percent': improvement_percent,
        'total_iterations': len(history),
        'algorithm_info': algorithm_info,
        'function_info': function_info,
    }

    return render(request, 'algorithms/results.html', context)


# ========== HYPER-HEURISTIC VIEWS ==========

def hyper_heuristic_page(request):
    """Main page for hyper-heuristic functionality"""
    context = {
        'functions': OPTIMIZATION_FUNCTIONS,
        'custom_problems': CustomProblem.objects.all(),
    }
    return render(request, 'algorithms/hyper_heuristic.html', context)


@csrf_exempt
def recommend_algorithm(request):
    """Get algorithm recommendation for a problem"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            function_key = data.get('function')
            custom_problem_id = data.get('custom_problem_id')

            # Initialize hyper-heuristic
            hh = HyperHeuristic()

            # Prepare problem data
            problem_data = {}

            if custom_problem_id:
                # Load custom problem
                custom_problem = CustomProblem.objects.get(id=custom_problem_id)
                problem_data = {
                    'type': custom_problem.problem_type,
                    'bounds': [custom_problem.bounds_lower, custom_problem.bounds_upper]
                }
            elif function_key:
                # Use built-in function
                function_info = OPTIMIZATION_FUNCTIONS[function_key]
                problem_data = {
                    'type': function_info.get('type', 'continuous'),
                    'function': function_info['function'],
                    'bounds': function_info['domain']
                }

            # Get recommendation
            recommendation = hh.select_algorithm(problem_data)
            rankings = hh.rank_algorithms(problem_data)

            # Save recommendation to database
            AlgorithmRecommendation.objects.create(
                problem_name=function_key or f"Custom_{custom_problem_id}",
                custom_problem_id=custom_problem_id,
                recommended_algorithm=recommendation['algorithm'],
                confidence=recommendation['confidence'],
                reason=recommendation['reason'],
                problem_features={},
                alternative_algorithms=recommendation.get('alternatives', [])
            )

            return JsonResponse({
                'success': True,
                'recommendation': recommendation,
                'rankings': rankings
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


# ========== META-HEURISTIC GENERATOR VIEWS ==========

def meta_heuristic_page(request):
    """Main page for meta-heuristic generator"""
    generated_algorithms = GeneratedAlgorithm.objects.all()
    context = {
        'generated_algorithms': generated_algorithms,
        'functions': OPTIMIZATION_FUNCTIONS,
    }
    return render(request, 'algorithms/meta_heuristic.html', context)


@csrf_exempt
def generate_algorithm(request):
    """Generate a new random algorithm"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            algorithm_name = data.get('name', f'Generated_{int(time.time())}')

            # Initialize generator
            generator = MetaHeuristicGenerator()

            # Generate random algorithm
            algorithm = generator.generate_random_algorithm(algorithm_name)

            # Save to database
            db_algorithm = GeneratedAlgorithm.objects.create(
                name=algorithm_name,
                description="Randomly generated algorithm",
                components=algorithm.to_dict()['components'],
                population_size=algorithm.population_size,
                is_evolved=False,
                generation=0
            )

            return JsonResponse({
                'success': True,
                'algorithm_id': db_algorithm.id,
                'algorithm': algorithm.to_dict(),
                'code': algorithm.get_code()
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


@csrf_exempt
def evolve_algorithms(request):
    """Evolve algorithms using genetic programming"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            function_key = data.get('function', 'sphere')
            n_algorithms = int(data.get('n_algorithms', 5))
            n_generations = int(data.get('n_generations', 3))
            test_iterations = int(data.get('test_iterations', 30))

            # Get function
            function_info = OPTIMIZATION_FUNCTIONS[function_key]
            bounds = function_info['domain']

            # Initialize generator
            generator = MetaHeuristicGenerator()

            # Evolve algorithms
            best_algorithm = generator.evolve_algorithms(
                objective_func=function_info['function'],
                bounds=bounds,
                n_algorithms=n_algorithms,
                n_generations=n_generations,
                test_iterations=test_iterations
            )

            # Save to database
            db_algorithm = GeneratedAlgorithm.objects.create(
                name=best_algorithm.name,
                description=f"Evolved algorithm using GP on {function_info['name']}",
                components=best_algorithm.to_dict()['components'],
                population_size=best_algorithm.population_size,
                is_evolved=True,
                generation=n_generations
            )

            return JsonResponse({
                'success': True,
                'algorithm_id': db_algorithm.id,
                'algorithm': best_algorithm.to_dict(),
                'code': best_algorithm.get_code()
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


@csrf_exempt
def test_generated_algorithm(request, algorithm_id):
    """Test a generated algorithm on a function"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            function_key = data.get('function', 'sphere')
            iterations = int(data.get('iterations', 100))

            # Get algorithm from database
            db_algorithm = GeneratedAlgorithm.objects.get(id=algorithm_id)

            # Get function
            function_info = OPTIMIZATION_FUNCTIONS[function_key]
            bounds = function_info['domain']

            # Reconstruct algorithm from components
            generator = MetaHeuristicGenerator()
            algorithm = generator.create_custom_algorithm(db_algorithm.components)

            # Run optimization
            start_time = time.time()
            history = algorithm.optimize(
                objective_func=function_info['function'],
                bounds=bounds,
                iterations=iterations
            )
            execution_time = time.time() - start_time

            # Update performance score
            db_algorithm.performance_score = history['best_fitness'][-1]
            db_algorithm.save()

            # Save run to database
            OptimizationRun.objects.create(
                algorithm='generated',
                generated_algorithm=db_algorithm,
                problem_name=function_key,
                iterations=iterations,
                population_size=algorithm.population_size,
                best_fitness=history['best_fitness'][-1],
                best_solution=history['best_solution'].tolist(),
                convergence_history=history,
                execution_time=execution_time
            )

            # Generate session ID and store results
            session_id = str(uuid.uuid4())
            results = {
                'algorithm': {
                    'name': db_algorithm.name,
                    'description': db_algorithm.description
                },
                'function': function_info,
                'history': [
                    {
                        'generation': i,
                        'best_fitness': history['best_fitness'][i],
                        'avg_fitness': history['avg_fitness'][i],
                        'best_solution': history['best_solution'].tolist()
                    }
                    for i in range(len(history['best_fitness']))
                ],
                'session_id': session_id
            }

            cache.set(f'optimization_{session_id}', results, 3600)

            return JsonResponse({
                'success': True,
                'session_id': session_id,
                'best_solution': history['best_solution'].tolist(),
                'best_fitness': history['best_fitness'][-1],
                'execution_time': execution_time
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


# ========== CUSTOM PROBLEM VIEWS ==========

def custom_problems_page(request):
    """Main page for custom problems"""
    problems = CustomProblem.objects.all()
    templates = ProblemTemplateGenerator.get_all_templates()

    context = {
        'problems': problems,
        'templates': templates,
    }
    return render(request, 'algorithms/custom_problems.html', context)


@csrf_exempt
def create_custom_problem(request):
    """Create a new custom problem"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            name = data.get('name')
            description = data.get('description', '')
            problem_type = data.get('problem_type', 'continuous')
            function_code = data.get('function_code')
            bounds_lower = float(data.get('bounds_lower', -10))
            bounds_upper = float(data.get('bounds_upper', 10))
            dimensions = int(data.get('dimensions', 2))

            # Validate function
            is_valid, msg, func = ProblemValidator.validate_python_function(function_code)

            if not is_valid:
                return JsonResponse({
                    'success': False,
                    'error': f'Function validation failed: {msg}'
                })

            # Create problem
            problem = CustomProblem.objects.create(
                name=name,
                description=description,
                problem_type=problem_type,
                function_code=function_code,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                dimensions=dimensions,
                is_public=True
            )

            return JsonResponse({
                'success': True,
                'problem_id': problem.id,
                'message': f'Problem "{name}" created successfully'
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


@csrf_exempt
def test_custom_problem(request, problem_id):
    """Test optimization on a custom problem"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            algorithm_key = data.get('algorithm', 'ga')
            iterations = int(data.get('iterations', 100))

            # Get problem
            problem = CustomProblem.objects.get(id=problem_id)

            # Validate and compile function
            is_valid, msg, objective_func = ProblemValidator.validate_python_function(
                problem.function_code
            )

            if not is_valid:
                return JsonResponse({
                    'success': False,
                    'error': f'Function validation failed: {msg}'
                })

            # Get algorithm
            algorithm_info = ALGORITHMS[algorithm_key]
            # Algorithms expect bounds as [[lower, upper], [lower, upper]] for each dimension
            bounds = [[problem.bounds_lower, problem.bounds_upper] for _ in range(problem.dimensions)]

            # Initialize algorithm
            algorithm_class = algorithm_info['class']
            if algorithm_key == 'ga':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    bounds=bounds,
                    generations=iterations
                )
            elif algorithm_key == 'pso':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    bounds=bounds,
                    iterations=iterations
                )
            elif algorithm_key == 'aco':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    bounds=bounds,
                    iterations=iterations
                )
            elif algorithm_key == 'tlbo':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    bounds=bounds,
                    iterations=iterations
                )
            elif algorithm_key == 'ts':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    bounds=bounds,
                    iterations=iterations
                )

            # Run optimization
            start_time = time.time()
            history = algorithm.optimize()
            execution_time = time.time() - start_time

            # Update usage count
            problem.usage_count += 1
            problem.save()

            # Save run to database
            OptimizationRun.objects.create(
                algorithm=algorithm_key,
                problem_name=problem.name,
                custom_problem=problem,
                iterations=iterations,
                best_fitness=history[-1]['best_fitness'],
                best_solution=history[-1]['best_solution'],
                convergence_history=history,
                execution_time=execution_time
            )

            # Generate session ID and store results
            session_id = str(uuid.uuid4())
            results = {
                'algorithm': algorithm_info,
                'function': {
                    'name': problem.name,
                    'description': problem.description,
                    'global_optimum': None,
                    'domain': [problem.bounds_lower, problem.bounds_upper]
                },
                'history': history,
                'session_id': session_id
            }

            cache.set(f'optimization_{session_id}', results, 3600)

            return JsonResponse({
                'success': True,
                'session_id': session_id,
                'best_solution': history[-1]['best_solution'],
                'best_fitness': history[-1]['best_fitness'],
                'execution_time': execution_time
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


def get_problem_template(request, template_name):
    """Get a problem template"""
    templates = ProblemTemplateGenerator.get_all_templates()

    if template_name in templates:
        return JsonResponse({
            'success': True,
            'template': templates[template_name]
        })

    return JsonResponse({
        'success': False,
        'error': 'Template not found'
    })


# ========== FLIGHT OPTIMIZER VIEWS ==========

def flight_optimizer_page(request):
    """Main page for flight optimization"""
    context = {
        'algorithms': ALGORITHMS,
    }
    return render(request, 'algorithms/flight_optimizer.html', context)


@csrf_exempt
def optimize_flights(request):
    """Optimize flight selection based on user preferences"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Parse flight data
            flights_data = data.get('flights', [])
            if not flights_data:
                return JsonResponse({
                    'success': False,
                    'error': 'No flight data provided'
                })

            # Parse weights
            weights = {
                'w_price': float(data.get('w_price', 0.4)),
                'w_duration': float(data.get('w_duration', 0.3)),
                'w_stopovers': float(data.get('w_stopovers', 0.2)),
                'w_comfort': float(data.get('w_comfort', 0.1))
            }

            # Get algorithm parameters
            algorithm_key = data.get('algorithm', 'ga')
            iterations = int(data.get('iterations', 50))

            # Get algorithm info for all cases
            algorithm_info = ALGORITHMS[algorithm_key]

            # Create flight optimization problem
            flight_problem = FlightOptimizationProblem(flights_data, weights)

            # Track which algorithm implementation is used
            algorithm_implementation = "unknown"

            # Run specialized evolutionary algorithm for flights
            start_time = time.time()

            if algorithm_key == 'ga':
                # Use specialized Genetic Algorithm for flights
                algorithm_implementation = "GeneticAlgorithmFlights (specialized)"
                algorithm = GeneticAlgorithmFlights(
                    problem=flight_problem,
                    population_size=50,
                    generations=iterations,
                    crossover_rate=0.8,
                    mutation_rate=0.2,
                    elitism=2
                )
                result = algorithm.optimize()
                best_flight_idx = result['best_solution']
                best_flight = result['best_flight']
                history = result['history']

            elif algorithm_key == 'pso':
                # Use specialized PSO for flights
                algorithm_implementation = "ParticleSwarmFlights (specialized)"
                algorithm = ParticleSwarmFlights(
                    problem=flight_problem,
                    n_particles=30,
                    iterations=iterations,
                    w=0.7,
                    c1=1.5,
                    c2=1.5
                )
                result = algorithm.optimize()
                best_flight_idx = result['best_solution']
                best_flight = result['best_flight']
                history = result['history']

            else:
                # Fallback to generic algorithms (for ACO, TLBO, TS)
                algorithm_implementation = f"{algorithm_key.upper()} (generic 2D algorithm)"
                objective_func = get_flight_optimization_function(flights_data, weights)
                n_flights = len(flights_data)
                bounds = [[0, n_flights], [0, n_flights]]

                algorithm_class = algorithm_info['class']

                if algorithm_key == 'aco':
                    algorithm = algorithm_class(objective_func=objective_func, bounds=bounds, iterations=iterations)
                elif algorithm_key == 'tlbo':
                    algorithm = algorithm_class(objective_func=objective_func, bounds=bounds, iterations=iterations)
                elif algorithm_key == 'ts':
                    algorithm = algorithm_class(objective_func=objective_func, bounds=bounds, iterations=iterations)

                history = algorithm.optimize()
                best_solution = history[-1]['best_solution']
                best_flight_idx = int(abs(best_solution[0]) * len(flights_data)) % len(flights_data)
                best_flight = flights_data[best_flight_idx]

            execution_time = time.time() - start_time

            # Get ranking of all flights
            ranking = flight_problem.get_flight_ranking()

            # Generate session ID and store results
            session_id = str(uuid.uuid4())
            results = {
                'type': 'flight_optimization',
                'flights': flights_data,
                'weights': weights,
                'best_flight': best_flight,
                'best_flight_index': best_flight_idx,
                'ranking': ranking,
                'algorithm': algorithm_info,
                'execution_time': execution_time,
                'convergence_history': history
            }

            cache.set(f'flight_optimization_{session_id}', results, 3600)

            return JsonResponse({
                'success': True,
                'session_id': session_id,
                'best_flight': best_flight,
                'best_flight_index': best_flight_idx,
                'ranking': ranking,
                'execution_time': execution_time,
                'algorithm_used': algorithm_implementation  # Show which algorithm was actually used
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({'success': False, 'error': 'Method not allowed'})


def flight_results(request, session_id):
    """Display flight optimization results"""
    results = cache.get(f'flight_optimization_{session_id}')

    if not results:
        return render(request, 'algorithms/error.html', {
            'error': 'Results not found or expired'
        })

    return render(request, 'algorithms/flight_results.html', {
        'results': results,
        'session_id': session_id
    })
