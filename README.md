# 🧬 Evolutionary Algorithms Optimizer

Una aplicación web Django que implementa múltiples algoritmos evolutivos para resolver problemas de optimización. Desarrollado para el curso de Algoritmos Evolutivos de la Universidad de Buenos Aires.

## 📋 Descripción del Proyecto

Este proyecto implementa una interfaz web interactiva que permite a los usuarios seleccionar entre diferentes algoritmos evolutivos y funciones objetivo para resolver problemas de optimización. El sistema incluye visualizaciones de convergencia y trayectorias de solución.

### 🎯 Problema Abordado

El proyecto se enfoca en la optimización de funciones matemáticas en 2D, específicamente:

1. **Función Esfera**: f(x,y) = x² + y²
2. **Función Rosenbrock**: f(x,y) = 100(y - x²)² + (1 - x)²
3. **Función Rastrigin**: f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))

### 🤖 Algoritmos Implementados

1. **Algoritmo Genético (GA)**: Basado en selección natural, cruzamiento y mutación
2. **Optimización por Enjambre de Partículas (PSO)**: Inspirado en el comportamiento de bandadas
3. **Optimización por Colonia de Hormigas (ACO)**: Basado en el comportamiento de forrajeo de hormigas
4. **Optimización Basada en Enseñanza-Aprendizaje (TLBO)**: Simula el proceso educativo
5. **Búsqueda Tabú (TS)**: Búsqueda local con memoria para evitar ciclos

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.13.7
- pip
- virtualenv (recomendado)

### Pasos de Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <repository-url>
   cd algoritmosevolutivos
   ```

2. **Crear y activar entorno virtual**:
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar base de datos**:
   ```bash
   python manage.py migrate
   ```

5. **Ejecutar servidor de desarrollo**:
   ```bash
   python manage.py runserver
   ```

6. **Acceder a la aplicación**:
   Abrir navegador en `http://localhost:8000`

## 🚀 Uso de la Aplicación

### Interfaz Principal

1. **Seleccionar Algoritmo**: Elegir entre GA, PSO, ACO, TLBO o TS
2. **Seleccionar Función Objetivo**: Elegir entre Sphere, Rosenbrock o Rastrigin
3. **Configurar Parámetros**: Establecer número de iteraciones (10-1000)
4. **Ejecutar Optimización**: Hacer click en "Run Optimization"

### Visualización de Resultados

Los resultados incluyen:

- **Gráfico de Convergencia**: Muestra la evolución del fitness a lo largo de las iteraciones
- **Trayectoria de Solución**: Visualiza el camino de la mejor solución en el espacio 2D
- **Estadísticas Detalladas**: Métricas de rendimiento y análisis comparativo
- **Insights del Algoritmo**: Explicaciones sobre el comportamiento del algoritmo

## 🧪 Pruebas

Para verificar que todos los algoritmos funcionan correctamente:

```bash
python test_algorithms.py
```

Este script ejecuta una batería de pruebas en todos los algoritmos con diferentes funciones objetivo.

## 📊 Características Técnicas

### Arquitectura del Sistema

- **Backend**: Django 5.2.6
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualizaciones**: Plotly.js
- **Almacenamiento**: SQLite (cache de resultados)

### Estructura del Proyecto

```
algoritmosevolutivos/
├── algorithms/                 # Aplicación principal
│   ├── evolutionary_algorithms.py  # Implementaciones de algoritmos
│   ├── optimization_functions.py   # Funciones objetivo
│   ├── views.py               # Vistas Django
│   └── urls.py                # URLs de la aplicación
├── templates/                 # Plantillas HTML
│   └── algorithms/
│       ├── base.html
│       ├── index.html
│       ├── results.html
│       └── error.html
├── evolutionary_optimizer/    # Configuración Django
├── test_algorithms.py         # Suite de pruebas
└── requirements.txt           # Dependencias
```

### Algoritmos - Detalles de Implementación

#### 1. Algoritmo Genético (GA)
- **Población**: 50 individuos
- **Selección**: Torneo (tamaño 3)
- **Cruzamiento**: Un punto (80%)
- **Mutación**: Gaussiana (10%)

#### 2. PSO (Particle Swarm Optimization)
- **Enjambre**: 30 partículas
- **Inercia**: w = 0.7
- **Coeficientes**: c1 = c2 = 1.5

#### 3. ACO (Ant Colony Optimization)
- **Hormigas**: 30
- **Discretización**: Grid 20x20
- **Evaporación**: ρ = 0.1

#### 4. TLBO (Teaching-Learning-Based Optimization)
- **Población**: 30 estudiantes
- **Fases**: Enseñanza y Aprendizaje
- **Factor de enseñanza**: 1 o 2 (aleatorio)

#### 5. Tabu Search
- **Lista tabú**: 20 elementos
- **Tamaño de paso**: 10% del dominio
- **Vecindario**: 4 direcciones por dimensión

## 📈 Resultados y Análisis

### Funciones de Prueba

#### Función Esfera
- **Características**: Convexa, unimodal
- **Mejor Algoritmo**: TLBO (convergencia 100%)
- **Dificultad**: Fácil

#### Función Rosenbrock
- **Características**: No convexa, valle estrecho
- **Mejor Algoritmo**: PSO (convergencia 99.99%)
- **Dificultad**: Moderada

#### Función Rastrigin
- **Características**: Multimodal, muchos mínimos locales
- **Mejor Algoritmo**: Variable según ejecución
- **Dificultad**: Difícil

### Análisis Comparativo

1. **PSO**: Excelente para convergencia rápida
2. **GA**: Buena exploración del espacio
3. **TLBO**: Muy efectivo en funciones unimodales
4. **ACO**: Buen balance exploración/explotación
5. **TS**: Efectivo para escape de mínimos locales

## 🎯 Compatibilidad con Google Colab

El proyecto está diseñado para ejecutarse fácilmente en Google Colab:

1. **Subir archivos** al entorno Colab
2. **Instalar dependencias**:
   ```python
   !pip install django plotly pandas numpy scipy matplotlib
   ```
3. **Ejecutar servidor**:
   ```python
   !python manage.py runserver 0.0.0.0:8000 &
   ```
4. **Usar túnel** para acceso público (ngrok recomendado)

## 📚 Documentación Académica

### Metodología

1. **Análisis del Problema**: Selección de funciones benchmark estándar
2. **Implementación**: Desarrollo de 5 algoritmos evolutivos diferentes
3. **Interfaz Usuario**: Aplicación web interactiva con Django
4. **Visualización**: Gráficos de convergencia y trayectorias
5. **Evaluación**: Pruebas automatizadas y análisis comparativo

### Inconvenientes Encontrados y Soluciones

#### 1. Convergencia Prematura en GA
- **Problema**: Pérdida de diversidad poblacional
- **Solución**: Ajuste de tasa de mutación y selección por torneo

#### 2. Estancamiento en PSO
- **Problema**: Partículas atrapadas en mínimos locales
- **Solución**: Parámetros de inercia dinámicos

#### 3. Discretización en ACO
- **Problema**: Adaptación a espacio continuo
- **Solución**: Grid de feromonas con interpolación

#### 4. Sensibilidad de Parámetros en TLBO
- **Problema**: Rendimiento variable según función
- **Solución**: Factor de enseñanza adaptativo

#### 5. Memoria Limitada en TS
- **Problema**: Lista tabú muy pequeña
- **Solución**: Tamaño adaptativo según progreso

## 🔧 Configuración Avanzada

### Parámetros Personalizables

Modificar en `evolutionary_algorithms.py`:

```python
# Ejemplo para GA
algorithm = GeneticAlgorithm(
    population_size=100,    # Tamaño población
    mutation_rate=0.15,     # Tasa mutación
    crossover_rate=0.85,    # Tasa cruzamiento
    generations=200         # Generaciones
)
```

### Nuevas Funciones Objetivo

Agregar en `optimization_functions.py`:

```python
def nueva_funcion(x):
    return x[0]**2 + x[1]**2  # Ejemplo

OPTIMIZATION_FUNCTIONS['nueva'] = {
    'function': nueva_funcion,
    'name': 'Nueva Función',
    'description': 'Descripción',
    'domain': [-10, 10],
    'global_optimum': [0, 0],
    'global_minimum': 0
}
```

## 📄 Licencia

Este proyecto fue desarrollado para fines académicos en la Universidad de Buenos Aires.

## 👥 Contribución

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📞 Soporte

Para preguntas o problemas:
- Crear issue en el repositorio
- Contactar al desarrollador

---

**Desarrollado con ❤️ para el curso de Algoritmos Evolutivos - UBA 2024**#   a l g o r i t m o s e v o l u t i v o s  
 