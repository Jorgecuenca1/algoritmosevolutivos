#!/usr/bin/env python
"""
Demostración completa del sistema de algoritmos evolutivos
Ejecuta todos los algoritmos en todas las funciones y genera un reporte
"""

import sys
import os
import django
import time
import matplotlib.pyplot as plt
import numpy as np

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evolutionary_optimizer.settings')
django.setup()

from algorithms.optimization_functions import OPTIMIZATION_FUNCTIONS
from algorithms.evolutionary_algorithms import ALGORITHMS


def ejecutar_demostracion_completa():
    """Ejecuta una demostración completa del sistema"""

    print("=" * 80)
    print("DEMOSTRACIÓN COMPLETA - ALGORITMOS EVOLUTIVOS")
    print("Universidad de Buenos Aires - Algoritmos Evolutivos")
    print("=" * 80)

    # Configuración de la demo
    iteraciones = 50
    ejecuciones_por_test = 3

    print(f"\nConfiguración de la demostración:")
    print(f"- Iteraciones por algoritmo: {iteraciones}")
    print(f"- Ejecuciones por test: {ejecuciones_por_test}")
    print(f"- Algoritmos: {len(ALGORITHMS)}")
    print(f"- Funciones objetivo: {len(OPTIMIZATION_FUNCTIONS)}")

    resultados_completos = {}

    # Ejecutar todos los algoritmos en todas las funciones
    for func_key, func_info in OPTIMIZATION_FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"FUNCIÓN: {func_info['name']}")
        print(f"Descripción: {func_info['description']}")
        print(f"Dominio: {func_info['domain']}")
        print(f"Óptimo global: {func_info['global_optimum']} -> {func_info['global_minimum']}")
        print(f"{'='*60}")

        resultados_completos[func_key] = {}

        for algo_key, algo_info in ALGORITHMS.items():
            print(f"\n--- {algo_info['name']} ---")

            resultados_algoritmo = []

            for ejecucion in range(ejecuciones_por_test):
                print(f"Ejecución {ejecucion + 1}/{ejecuciones_por_test}... ", end="")

                # Configurar algoritmo
                bounds = [func_info['domain'], func_info['domain']]
                algorithm_class = algo_info['class']

                if algo_key == 'ga':
                    algorithm = algorithm_class(
                        objective_func=func_info['function'],
                        bounds=bounds,
                        generations=iteraciones,
                        population_size=30
                    )
                elif algo_key == 'pso':
                    algorithm = algorithm_class(
                        objective_func=func_info['function'],
                        bounds=bounds,
                        iterations=iteraciones,
                        swarm_size=30
                    )
                elif algo_key == 'aco':
                    algorithm = algorithm_class(
                        objective_func=func_info['function'],
                        bounds=bounds,
                        iterations=iteraciones,
                        num_ants=30
                    )
                elif algo_key == 'tlbo':
                    algorithm = algorithm_class(
                        objective_func=func_info['function'],
                        bounds=bounds,
                        iterations=iteraciones,
                        population_size=30
                    )
                elif algo_key == 'ts':
                    algorithm = algorithm_class(
                        objective_func=func_info['function'],
                        bounds=bounds,
                        iterations=iteraciones
                    )

                # Ejecutar optimización
                start_time = time.time()
                history = algorithm.optimize()
                end_time = time.time()

                # Guardar resultados
                final_result = history[-1]
                initial_result = history[0]

                resultado = {
                    'fitness_final': final_result['best_fitness'],
                    'solucion_final': final_result['best_solution'],
                    'fitness_inicial': initial_result['best_fitness'],
                    'mejora': initial_result['best_fitness'] - final_result['best_fitness'],
                    'tiempo_ejecucion': end_time - start_time,
                    'historia': history
                }

                resultados_algoritmo.append(resultado)
                print(f"Fitness: {final_result['best_fitness']:.6f}")

            # Calcular estadísticas
            fitness_values = [r['fitness_final'] for r in resultados_algoritmo]
            mejoras = [r['mejora'] for r in resultados_algoritmo]
            tiempos = [r['tiempo_ejecucion'] for r in resultados_algoritmo]

            estadisticas = {
                'fitness_promedio': np.mean(fitness_values),
                'fitness_std': np.std(fitness_values),
                'fitness_mejor': np.min(fitness_values),
                'fitness_peor': np.max(fitness_values),
                'mejora_promedio': np.mean(mejoras),
                'tiempo_promedio': np.mean(tiempos),
                'resultados_individuales': resultados_algoritmo
            }

            resultados_completos[func_key][algo_key] = estadisticas

            # Mostrar resumen
            print(f"  Fitness promedio: {estadisticas['fitness_promedio']:.6f} ± {estadisticas['fitness_std']:.6f}")
            print(f"  Mejor fitness: {estadisticas['fitness_mejor']:.6f}")
            print(f"  Tiempo promedio: {estadisticas['tiempo_promedio']:.2f}s")

    # Generar reporte final
    print(f"\n{'='*80}")
    print("REPORTE FINAL - RANKING POR FUNCIÓN")
    print(f"{'='*80}")

    for func_key, func_info in OPTIMIZATION_FUNCTIONS.items():
        print(f"\n{func_info['name']}:")
        print("-" * 40)

        # Ordenar algoritmos por mejor fitness promedio
        algoritmos_ordenados = sorted(
            resultados_completos[func_key].items(),
            key=lambda x: x[1]['fitness_promedio']
        )

        for i, (algo_key, stats) in enumerate(algoritmos_ordenados, 1):
            algo_name = ALGORITHMS[algo_key]['name']
            print(f"{i}. {algo_name:30} | {stats['fitness_promedio']:.6f} ± {stats['fitness_std']:.6f}")

    # Guardar gráfico de convergencia comparativo
    crear_grafico_comparativo(resultados_completos)

    print(f"\n{'='*80}")
    print("DEMOSTRACIÓN COMPLETADA")
    print("- Todos los algoritmos ejecutados exitosamente")
    print("- Resultados guardados en resultados_demo.txt")
    print("- Gráfico comparativo guardado en convergencia_comparativa.png")
    print(f"{'='*80}")

    return resultados_completos


def crear_grafico_comparativo(resultados):
    """Crea un gráfico comparativo de convergencia"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Convergencia Comparativa de Algoritmos Evolutivos', fontsize=16)

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (func_key, func_info) in enumerate(OPTIMIZATION_FUNCTIONS.items()):
        ax = axes[i]

        for j, (algo_key, algo_info) in enumerate(ALGORITHMS.items()):
            if algo_key in resultados[func_key]:
                # Tomar la mejor ejecución
                mejor_resultado = min(
                    resultados[func_key][algo_key]['resultados_individuales'],
                    key=lambda x: x['fitness_final']
                )

                historia = mejor_resultado['historia']
                generaciones = [h['generation'] for h in historia]
                fitness_values = [h['best_fitness'] for h in historia]

                ax.plot(generaciones, fitness_values,
                       color=colors[j], label=algo_info['name'], linewidth=2)

        ax.set_title(func_info['name'])
        ax.set_xlabel('Iteración/Generación')
        ax.set_ylabel('Fitness (log scale)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergencia_comparativa.png', dpi=300, bbox_inches='tight')
    print("\nGráfico guardado como 'convergencia_comparativa.png'")


def guardar_reporte_detallado(resultados):
    """Guarda un reporte detallado en archivo de texto"""

    with open('reporte_detallado.txt', 'w', encoding='utf-8') as f:
        f.write("REPORTE DETALLADO - ALGORITMOS EVOLUTIVOS\n")
        f.write("Universidad de Buenos Aires\n")
        f.write("=" * 50 + "\n\n")

        for func_key, func_info in OPTIMIZATION_FUNCTIONS.items():
            f.write(f"FUNCIÓN: {func_info['name']}\n")
            f.write(f"Descripción: {func_info['description']}\n")
            f.write(f"Óptimo global: {func_info['global_optimum']} -> {func_info['global_minimum']}\n")
            f.write("-" * 40 + "\n")

            for algo_key, stats in resultados[func_key].items():
                algo_name = ALGORITHMS[algo_key]['name']
                f.write(f"\n{algo_name}:\n")
                f.write(f"  Fitness promedio: {stats['fitness_promedio']:.8f}\n")
                f.write(f"  Desviación estándar: {stats['fitness_std']:.8f}\n")
                f.write(f"  Mejor fitness: {stats['fitness_mejor']:.8f}\n")
                f.write(f"  Peor fitness: {stats['fitness_peor']:.8f}\n")
                f.write(f"  Mejora promedio: {stats['mejora_promedio']:.8f}\n")
                f.write(f"  Tiempo promedio: {stats['tiempo_promedio']:.4f}s\n")

            f.write("\n" + "=" * 50 + "\n\n")


if __name__ == "__main__":
    try:
        resultados = ejecutar_demostracion_completa()
        guardar_reporte_detallado(resultados)

        print("\nLa aplicación web está disponible en:")
        print("http://localhost:8000")
        print("\nPara iniciar el servidor Django:")
        print("python manage.py runserver")

    except KeyboardInterrupt:
        print("\nDemostración interrumpida por el usuario")
    except Exception as e:
        print(f"\nError durante la demostración: {str(e)}")
        sys.exit(1)