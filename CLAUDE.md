# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete Django web application that implements multiple evolutionary algorithms for solving optimization problems. The project was developed for the Evolutionary Algorithms course at Universidad de Buenos Aires (UBA).

**Key Features:**
- Web-based interface for algorithm selection and execution
- 5 different evolutionary algorithms (GA, PSO, ACO, TLBO, TS)
- 3 benchmark optimization functions (Sphere, Rosenbrock, Rastrigin)
- Interactive convergence plots and solution trajectory visualization
- Comprehensive academic documentation

## Development Environment

- **Python Version**: 3.13.7
- **Framework**: Django 5.2.6
- **Virtual Environment**: Located in `.venv/`
- **Dependencies**: See `requirements.txt`

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Unix/Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Django setup
python manage.py migrate
```

### Running the Application
```bash
# Start Django development server
python manage.py runserver

# Access application at http://localhost:8000
```

### Testing and Validation
```bash
# Test all algorithms
python test_algorithms.py

# Run complete demonstration
python demo_completo.py

# Django tests
python manage.py test
```

## Project Structure

```
algoritmosevolutivos/
├── algorithms/                    # Main Django app
│   ├── evolutionary_algorithms.py # Algorithm implementations
│   ├── optimization_functions.py  # Objective functions
│   ├── views.py                   # Django views
│   ├── urls.py                    # URL routing
│   └── models.py                  # Django models
├── templates/algorithms/          # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Main interface
│   ├── results.html               # Results page
│   └── error.html                 # Error page
├── evolutionary_optimizer/        # Django project settings
├── test_algorithms.py             # Algorithm testing suite
├── demo_completo.py               # Complete demonstration
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── DOCUMENTACION_ACADEMICA.md     # Academic documentation
```

## Algorithms Implementation

### Available Algorithms
1. **Genetic Algorithm (GA)**: Population-based with selection, crossover, mutation
2. **Particle Swarm Optimization (PSO)**: Swarm intelligence algorithm
3. **Ant Colony Optimization (ACO)**: Pheromone-based optimization
4. **Teaching-Learning-Based Optimization (TLBO)**: Education-inspired algorithm
5. **Tabu Search (TS)**: Memory-based local search

### Objective Functions
1. **Sphere Function**: f(x,y) = x² + y² (unimodal, easy)
2. **Rosenbrock Function**: f(x,y) = 100(y - x²)² + (1 - x)² (narrow valley)
3. **Rastrigin Function**: Multimodal with many local minima (challenging)

## Web Application Architecture

### Backend (Django)
- **Views**: Handle algorithm execution and result processing
- **Cache**: Store optimization results with session IDs
- **JSON API**: Asynchronous communication with frontend

### Frontend
- **Bootstrap 5**: Responsive design framework
- **Plotly.js**: Interactive visualization library
- **Vanilla JavaScript**: Algorithm selection and execution

### Key Files to Understand

#### `algorithms/evolutionary_algorithms.py`
Contains all algorithm implementations with standardized interfaces:
```python
class AlgorithmName:
    def __init__(self, objective_func, bounds, iterations=100, **params):
        # Algorithm initialization

    def optimize(self):
        # Returns history with convergence data
        return history
```

#### `algorithms/views.py`
Django views that handle:
- Main page rendering (`index`)
- Algorithm execution (`optimize`)
- Results visualization (`results`)

#### `templates/algorithms/index.html`
Main interface with:
- Algorithm selection cards
- Function selection cards
- Parameter configuration
- Asynchronous execution with progress feedback

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to all functions and classes
- Implement proper error handling

### Algorithm Implementation
- Each algorithm must return standardized history format
- Include both best and average fitness tracking
- Implement proper bounds checking
- Use NumPy for numerical operations

### Web Interface
- Maintain responsive design principles
- Provide clear user feedback
- Handle errors gracefully
- Include loading indicators for long operations

### Testing
- Test each algorithm individually with `test_algorithms.py`
- Verify web interface functionality
- Check visualization rendering
- Validate result accuracy

## Google Colab Compatibility

The project is designed to run in Google Colab:

```python
# Install dependencies
!pip install django plotly pandas numpy scipy matplotlib

# Setup and run
!python manage.py migrate
!python manage.py runserver 0.0.0.0:8000 &

# Optional: Create public tunnel
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"App available at: {public_url}")
```

## Common Tasks

### Adding New Algorithms
1. Implement algorithm class in `evolutionary_algorithms.py`
2. Add entry to `ALGORITHMS` dictionary
3. Update view logic in `views.py` if needed
4. Test with `test_algorithms.py`

### Adding New Objective Functions
1. Implement function in `optimization_functions.py`
2. Add entry to `OPTIMIZATION_FUNCTIONS` dictionary
3. Test with existing algorithms

### Modifying Visualizations
- Edit Plotly.js code in `views.py` (results function)
- Modify HTML templates for layout changes
- Update CSS in base template for styling

## Academic Documentation

- **README.md**: Complete project documentation
- **DOCUMENTACION_ACADEMICA.md**: Formal academic paper (3+ pages)
- **demo_completo.py**: Comprehensive demonstration script

## Performance Notes

- Algorithms run on server-side (Django backend)
- Results cached for 1 hour using Django cache framework
- Visualizations generated server-side and sent as HTML
- Session-based result storage prevents data loss