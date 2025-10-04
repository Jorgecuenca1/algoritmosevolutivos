"""
Custom Problem Loader Module
=============================
Allows users to upload and define their own optimization problems
"""

import numpy as np
import json
import ast
import re
from typing import Callable, Dict, Any, List


class ProblemValidator:
    """Validates custom optimization problems"""

    @staticmethod
    def validate_python_function(code: str) -> tuple:
        """
        Validate Python function code for safety and correctness

        Returns: (is_valid, message, function_or_none)
        """
        # Check for dangerous operations
        dangerous_keywords = [
            'import os', 'import sys', 'import subprocess', '__import__',
            'exec', 'eval', 'compile', 'open(', 'file(',
            '__builtins__', 'globals(', 'locals(',
            'input(', 'raw_input('
        ]

        for keyword in dangerous_keywords:
            if keyword in code.lower():
                return False, f"Dangerous operation detected: {keyword}", None

        # Check that it's a valid function
        if not code.strip().startswith('def '):
            return False, "Code must define a function starting with 'def'", None

        # Try to parse and execute
        try:
            # Parse the code
            tree = ast.parse(code)

            # Check that it's a function definition
            if not isinstance(tree.body[0], ast.FunctionDef):
                return False, "Code must contain a function definition", None

            # Execute in restricted namespace
            namespace = {
                'np': np,
                'numpy': np,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'range': range,
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'pow': np.power,
            }

            exec(code, namespace)

            # Get the function
            func_name = tree.body[0].name
            func = namespace[func_name]

            # Test the function with sample input
            test_input = np.array([0.5, 0.5])
            try:
                result = func(test_input)
                if not isinstance(result, (int, float, np.number)):
                    return False, "Function must return a numeric value", None
            except Exception as e:
                return False, f"Function test failed: {str(e)}", None

            return True, "Function is valid", func

        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}", None
        except Exception as e:
            return False, f"Validation error: {str(e)}", None

    @staticmethod
    def validate_bounds(bounds: Any) -> tuple:
        """
        Validate bounds specification

        Returns: (is_valid, message, normalized_bounds)
        """
        try:
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lower, upper = bounds
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                    if lower < upper:
                        return True, "Bounds are valid", [float(lower), float(upper)]
                    else:
                        return False, "Lower bound must be less than upper bound", None

            return False, "Bounds must be [lower, upper]", None

        except Exception as e:
            return False, f"Invalid bounds: {str(e)}", None

    @staticmethod
    def validate_problem_json(json_data: Dict) -> tuple:
        """
        Validate a complete problem specification in JSON format

        Returns: (is_valid, message, problem_dict)
        """
        required_fields = ['name', 'type', 'function_code']

        # Check required fields
        for field in required_fields:
            if field not in json_data:
                return False, f"Missing required field: {field}", None

        # Validate problem type
        if json_data['type'] not in ['continuous', 'combinatorial']:
            return False, "Problem type must be 'continuous' or 'combinatorial'", None

        # Validate function
        is_valid, msg, func = ProblemValidator.validate_python_function(
            json_data['function_code']
        )

        if not is_valid:
            return False, f"Function validation failed: {msg}", None

        # Validate bounds for continuous problems
        if json_data['type'] == 'continuous':
            if 'bounds' not in json_data:
                return False, "Continuous problems require 'bounds'", None

            is_valid, msg, bounds = ProblemValidator.validate_bounds(json_data['bounds'])
            if not is_valid:
                return False, f"Bounds validation failed: {msg}", None

        return True, "Problem is valid", json_data


class CustomProblem:
    """Represents a custom user-defined optimization problem"""

    def __init__(self, name: str, problem_type: str, function_code: str,
                 bounds=None, dimensions=2, description="", constraints=None):
        self.name = name
        self.type = problem_type
        self.function_code = function_code
        self.bounds = bounds
        self.dimensions = dimensions
        self.description = description
        self.constraints = constraints or []
        self.objective_function = None

        # Compile the function
        self._compile_function()

    def _compile_function(self):
        """Compile the function code"""
        is_valid, msg, func = ProblemValidator.validate_python_function(self.function_code)

        if not is_valid:
            raise ValueError(f"Invalid function: {msg}")

        self.objective_function = func

    def evaluate(self, solution):
        """Evaluate a solution"""
        if self.objective_function is None:
            raise RuntimeError("Function not compiled")

        try:
            result = self.objective_function(solution)

            # Apply constraints
            penalty = 0
            for constraint in self.constraints:
                if not constraint(solution):
                    penalty += 1e6

            return result + penalty

        except Exception as e:
            raise RuntimeError(f"Error evaluating solution: {str(e)}")

    def to_dict(self):
        """Export problem as dictionary"""
        return {
            'name': self.name,
            'type': self.type,
            'function_code': self.function_code,
            'bounds': self.bounds,
            'dimensions': self.dimensions,
            'description': self.description
        }

    def to_json(self):
        """Export problem as JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class ProblemLibrary:
    """Manages a collection of custom problems"""

    def __init__(self):
        self.problems = {}
        self._load_examples()

    def _load_examples(self):
        """Load example custom problems"""
        # Example 1: Custom quadratic function
        quadratic_code = """def custom_quadratic(x):
    return x[0]**2 + 2*x[1]**2 + 3*x[0]*x[1] + 5"""

        self.add_problem(
            name="Custom Quadratic",
            problem_type="continuous",
            function_code=quadratic_code,
            bounds=[-10, 10],
            dimensions=2,
            description="Custom quadratic function with cross term"
        )

        # Example 2: Branin function
        branin_code = """def branin(x):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s"""

        self.add_problem(
            name="Branin Function",
            problem_type="continuous",
            function_code=branin_code,
            bounds=[-5, 15],
            dimensions=2,
            description="Branin function - has 3 global minima"
        )

        # Example 3: Six-Hump Camel function
        camel_code = """def six_hump_camel(x):
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2"""

        self.add_problem(
            name="Six-Hump Camel",
            problem_type="continuous",
            function_code=camel_code,
            bounds=[-3, 3],
            dimensions=2,
            description="Six-Hump Camel function - 2 global minima, 4 local minima"
        )

    def add_problem(self, name: str, problem_type: str, function_code: str,
                   bounds=None, dimensions=2, description=""):
        """Add a problem to the library"""
        try:
            problem = CustomProblem(
                name=name,
                problem_type=problem_type,
                function_code=function_code,
                bounds=bounds,
                dimensions=dimensions,
                description=description
            )

            self.problems[name] = problem
            return True, f"Problem '{name}' added successfully"

        except Exception as e:
            return False, f"Failed to add problem: {str(e)}"

    def get_problem(self, name: str):
        """Get a problem by name"""
        return self.problems.get(name)

    def list_problems(self):
        """List all available problems"""
        return [
            {
                'name': name,
                'type': prob.type,
                'dimensions': prob.dimensions,
                'description': prob.description
            }
            for name, prob in self.problems.items()
        ]

    def remove_problem(self, name: str):
        """Remove a problem from the library"""
        if name in self.problems:
            del self.problems[name]
            return True, f"Problem '{name}' removed"
        return False, f"Problem '{name}' not found"

    def import_from_json(self, json_str: str):
        """Import problem from JSON string"""
        try:
            data = json.loads(json_str)

            is_valid, msg, validated_data = ProblemValidator.validate_problem_json(data)
            if not is_valid:
                return False, msg

            return self.add_problem(
                name=data['name'],
                problem_type=data['type'],
                function_code=data['function_code'],
                bounds=data.get('bounds'),
                dimensions=data.get('dimensions', 2),
                description=data.get('description', '')
            )

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, f"Import failed: {str(e)}"

    def export_to_json(self, name: str):
        """Export problem to JSON string"""
        problem = self.get_problem(name)
        if problem:
            return True, problem.to_json()
        return False, f"Problem '{name}' not found"


class ProblemTemplateGenerator:
    """Generates templates for common problem types"""

    @staticmethod
    def generate_continuous_template(problem_name="my_function"):
        """Generate template for continuous optimization problem"""
        template = f"""def {problem_name}(x):
    '''
    Custom optimization function

    Parameters:
    -----------
    x : numpy array
        Input vector [x[0], x[1], ..., x[n-1]]

    Returns:
    --------
    float : objective value to minimize
    '''
    # Example: minimize x^2 + y^2
    return np.sum(x**2)

# You can use:
# - np.sum(), np.prod(), np.mean()
# - np.sin(), np.cos(), np.exp(), np.log(), np.sqrt()
# - Mathematical operators: +, -, *, /, **
# - x[0], x[1], ... to access variables
"""
        return template

    @staticmethod
    def generate_constrained_template(problem_name="constrained_function"):
        """Generate template for constrained optimization"""
        template = f"""def {problem_name}(x):
    '''
    Constrained optimization function

    Use large penalties for constraint violations
    '''
    # Objective function
    objective = x[0]**2 + x[1]**2

    # Constraints (add penalty if violated)
    penalty = 0

    # Example: x[0] + x[1] <= 1
    if x[0] + x[1] > 1:
        penalty += 1000 * (x[0] + x[1] - 1)**2

    # Example: x[0] >= 0
    if x[0] < 0:
        penalty += 1000 * x[0]**2

    return objective + penalty
"""
        return template

    @staticmethod
    def generate_engineering_template():
        """Generate template for engineering design problem"""
        template = """def engineering_design(x):
    '''
    Engineering design optimization

    Example: minimize weight subject to stress constraints
    '''
    # Design variables
    diameter = x[0]  # in mm
    length = x[1]    # in mm

    # Objective: minimize weight
    weight = np.pi * (diameter/2)**2 * length * 7850  # kg (steel density)

    # Constraints
    penalty = 0

    # Maximum stress constraint
    max_stress = 200  # MPa
    applied_load = 1000  # N
    cross_section = np.pi * (diameter/2)**2
    stress = applied_load / cross_section

    if stress > max_stress:
        penalty += 1e6 * (stress - max_stress)**2

    # Minimum dimension constraints
    if diameter < 5:
        penalty += 1e6 * (5 - diameter)**2
    if length < 10:
        penalty += 1e6 * (10 - length)**2

    return weight + penalty
"""
        return template

    @staticmethod
    def get_all_templates():
        """Get all available templates"""
        return {
            'continuous': ProblemTemplateGenerator.generate_continuous_template(),
            'constrained': ProblemTemplateGenerator.generate_constrained_template(),
            'engineering': ProblemTemplateGenerator.generate_engineering_template()
        }
