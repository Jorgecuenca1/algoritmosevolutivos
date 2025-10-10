"""
AI-Powered Metaheuristic Generator
Uses OpenAI API to analyze problems and generate custom optimization algorithms
"""

import json
import re
import numpy as np
from openai import OpenAI


class AIMetaheuristicService:
    """Service for interacting with OpenAI API to generate optimization algorithms"""

    def __init__(self, api_key, model="gpt-4o-mini"):
        """
        Initialize OpenAI client with API key

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini for cost efficiency)
                   Options: gpt-4o-mini (cheapest), gpt-3.5-turbo, gpt-4o, gpt-4
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model  # Default: gpt-4o-mini (200x cheaper than GPT-4)

    def analyze_problem(self, problem_description, constraints=None):
        """
        Analyze a problem description and extract key features

        Args:
            problem_description: Natural language description of the optimization problem
            constraints: Optional dictionary of constraints

        Returns:
            Dictionary with problem analysis
        """
        prompt = f"""
Analyze the following optimization problem and provide a structured analysis:

Problem Description:
{problem_description}

{"Constraints: " + json.dumps(constraints) if constraints else ""}

Provide your analysis in the following JSON format:
{{
    "problem_type": "continuous|discrete|combinatorial|mixed",
    "objective": "minimization|maximization",
    "dimensionality": <estimated number of variables>,
    "complexity": "easy|medium|hard",
    "landscape": "unimodal|multimodal|rugged|smooth",
    "constraints_type": "none|box|linear|nonlinear",
    "recommended_algorithm": "GA|PSO|ACO|DE|SA|TLBO|custom",
    "reasoning": "explanation of why this algorithm is recommended",
    "key_features": ["list", "of", "key", "features"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in optimization algorithms and metaheuristics. Analyze problems and recommend appropriate solution strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group())
                return {
                    'success': True,
                    'analysis': analysis
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not parse analysis response'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_algorithm_code(self, problem_description, problem_analysis, bounds, dimensions=2):
        """
        Generate Python code for a custom optimization algorithm

        Args:
            problem_description: Natural language description
            problem_analysis: Analysis from analyze_problem()
            bounds: List of [lower, upper] bounds for each dimension
            dimensions: Number of dimensions

        Returns:
            Dictionary with generated algorithm code
        """
        prompt = f"""
Generate a complete Python implementation of an optimization algorithm for the following problem:

Problem Description:
{problem_description}

Problem Analysis:
{json.dumps(problem_analysis, indent=2)}

Bounds: {bounds}
Dimensions: {dimensions}

Requirements:
1. Create a class called 'AIGeneratedAlgorithm' with an __init__ and optimize() method
2. The __init__ method should accept: objective_func, bounds, iterations=100, population_size=50
3. The optimize() method should return a list of dictionaries with history:
   [
     {{'iteration': 0, 'best_fitness': float, 'avg_fitness': float, 'best_solution': list}},
     ...
   ]
4. Use numpy for numerical operations
5. Implement proper bounds checking
6. Include docstrings explaining the algorithm
7. Make the algorithm efficient and robust
8. Base the algorithm on the recommended approach from the analysis
9. Include adaptive parameters if appropriate
10. Only import standard libraries: numpy, random, copy, math

Return ONLY the Python code, no explanations. Start with 'import' statements.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer specializing in optimization algorithms. Generate clean, efficient, well-documented code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2500
            )

            code = response.choices[0].message.content

            # Clean up code (remove markdown formatting if present)
            code = re.sub(r'^```python\s*', '', code)
            code = re.sub(r'^```\s*', '', code)
            code = re.sub(r'\s*```$', '', code)

            return {
                'success': True,
                'code': code.strip()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_conclusions(self, problem_description, algorithm_name, results, execution_time):
        """
        Generate conclusions and insights from optimization results

        Args:
            problem_description: Original problem description
            algorithm_name: Name of algorithm used
            results: Optimization results with history
            execution_time: Time taken to execute

        Returns:
            Dictionary with conclusions
        """
        history = results.get('history', [])
        if not history:
            return {
                'success': False,
                'error': 'No history data available'
            }

        final_result = history[-1]
        initial_fitness = history[0]['best_fitness']
        final_fitness = final_result['best_fitness']
        convergence_rate = (initial_fitness - final_fitness) / len(history) if len(history) > 0 else 0

        prompt = f"""
Analyze the following optimization results and provide insights:

Problem:
{problem_description}

Algorithm Used: {algorithm_name}

Results Summary:
- Initial fitness: {initial_fitness:.6f}
- Final fitness: {final_fitness:.6f}
- Total iterations: {len(history)}
- Execution time: {execution_time:.2f} seconds
- Convergence rate: {convergence_rate:.6f} per iteration
- Best solution: {final_result.get('best_solution', [])}

Full convergence history (first 5 and last 5 iterations):
First 5: {json.dumps(history[:5], indent=2) if len(history) >= 5 else json.dumps(history, indent=2)}
Last 5: {json.dumps(history[-5:], indent=2) if len(history) >= 5 else ''}

Provide a comprehensive analysis in the following JSON format:
{{
    "summary": "brief 2-3 sentence summary of results",
    "performance": "excellent|good|fair|poor",
    "convergence_quality": "fast|moderate|slow",
    "solution_quality": "optimal|near-optimal|suboptimal",
    "key_insights": [
        "insight 1",
        "insight 2",
        "insight 3"
    ],
    "recommendations": [
        "recommendation 1",
        "recommendation 2"
    ],
    "potential_improvements": [
        "improvement 1",
        "improvement 2"
    ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in optimization and algorithm performance analysis. Provide insightful, actionable conclusions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                conclusions = json.loads(json_match.group())
                return {
                    'success': True,
                    'conclusions': conclusions
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not parse conclusions response'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_objective_function(self, problem_description, bounds, dimensions):
        """
        Generate Python code for an objective function based on problem description

        Args:
            problem_description: Natural language description
            bounds: Bounds for the problem
            dimensions: Number of dimensions

        Returns:
            Dictionary with function code
        """
        prompt = f"""
Generate a Python objective function for the following optimization problem:

Problem Description:
{problem_description}

Bounds: {bounds}
Dimensions: {dimensions}

Requirements:
1. Create a function called 'objective_function' that takes a solution array/list as input
2. Return a single float value (the fitness/cost)
3. Handle edge cases and bounds violations appropriately
4. Use numpy for calculations
5. Include a docstring explaining what the function optimizes
6. The function should be for MINIMIZATION (lower is better)

Return ONLY the Python code for the function, no explanations.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in mathematical modeling and optimization. Generate clean, efficient Python code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            code = response.choices[0].message.content

            # Clean up code
            code = re.sub(r'^```python\s*', '', code)
            code = re.sub(r'^```\s*', '', code)
            code = re.sub(r'\s*```$', '', code)

            return {
                'success': True,
                'code': code.strip()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def validate_and_execute_code(code, objective_func, bounds, iterations=100, population_size=50):
    """
    Safely validate and execute generated algorithm code

    Args:
        code: Python code string to execute
        objective_func: Objective function to optimize
        bounds: Problem bounds
        iterations: Number of iterations
        population_size: Population size

    Returns:
        Dictionary with execution results or error
    """
    try:
        import random
        import copy
        import math

        # Create a safe namespace for code execution
        # Include necessary modules and built-in functions
        safe_builtins = {
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'float': float,
            'int': int,
            'bool': bool,
            'str': str,
            'True': True,
            'False': False,
            'None': None,
            'sorted': sorted,
            'reversed': reversed,
            'round': round,
            'pow': pow,
            'isinstance': isinstance,
            'type': type,
            'print': print,  # For debugging if needed
            '__import__': __import__,  # Needed for some imports
            '__build_class__': __build_class__,  # Needed to define classes
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'all': all,
            'any': any,
        }

        namespace = {
            'np': np,
            'numpy': np,
            'random': random,
            'copy': copy,
            'math': math,
            '__name__': '__main__',  # Set module name for if __name__ checks
            '__builtins__': safe_builtins
        }

        # Execute the code to define the class
        exec(code, namespace)

        # Get the generated class
        if 'AIGeneratedAlgorithm' not in namespace:
            return {
                'success': False,
                'error': 'Generated code does not define AIGeneratedAlgorithm class'
            }

        algorithm_class = namespace['AIGeneratedAlgorithm']

        # Create a wrapper for the objective function to handle different input types
        def objective_func_wrapper(x):
            """Wrapper to ensure objective function receives correct input format"""
            # Convert to numpy array if it's a list
            if isinstance(x, list):
                x = np.array(x)
            return objective_func(x)

        # Instantiate and run the algorithm with flexible parameters
        # Try different parameter combinations to match the generated __init__ signature
        try:
            # Try with all parameters
            algorithm = algorithm_class(
                objective_func=objective_func_wrapper,
                bounds=bounds,
                iterations=iterations,
                population_size=population_size
            )
        except TypeError:
            try:
                # Try without population_size
                algorithm = algorithm_class(
                    objective_func=objective_func_wrapper,
                    bounds=bounds,
                    iterations=iterations
                )
            except TypeError:
                try:
                    # Try with just required parameters
                    algorithm = algorithm_class(
                        objective_func=objective_func_wrapper,
                        bounds=bounds
                    )
                except TypeError as e:
                    return {
                        'success': False,
                        'error': f'Could not instantiate algorithm: {str(e)}'
                    }

        history = algorithm.optimize()

        if not history or len(history) == 0:
            return {
                'success': False,
                'error': 'Algorithm returned empty history'
            }

        return {
            'success': True,
            'history': history
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Execution error: {str(e)}'
        }
