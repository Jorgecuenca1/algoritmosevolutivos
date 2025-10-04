"""Test script for evolutionary flight optimization"""

from algorithms.flight_evolutionary import GeneticAlgorithmFlights, ParticleSwarmFlights
from algorithms.flight_optimizer import EXAMPLE_FLIGHTS_BOGOTA_MADRID, FlightOptimizationProblem

# Create problem
problem = FlightOptimizationProblem(EXAMPLE_FLIGHTS_BOGOTA_MADRID)

print("=" * 60)
print("FLIGHT OPTIMIZATION USING EVOLUTIONARY ALGORITHMS")
print("=" * 60)

print("\nAvailable Flights:")
for i, flight in enumerate(EXAMPLE_FLIGHTS_BOGOTA_MADRID):
    print(f"{i+1}. {flight['name']}: ${flight['price']}, "
          f"{flight['duration']}h, {flight['stopovers']} stops, "
          f"comfort {flight['comfort']}/10")

# Run Genetic Algorithm
print("\n" + "=" * 60)
print("GENETIC ALGORITHM")
print("=" * 60)

ga = GeneticAlgorithmFlights(problem, population_size=50, generations=100)
ga_result = ga.optimize()

print(f"\nBest Flight Found: {ga_result['best_flight']['name']}")
print(f"   Cost Score: {ga_result['best_cost']:.4f}")
print(f"   Price: ${ga_result['best_flight']['price']}")
print(f"   Duration: {ga_result['best_flight']['duration']}h")
print(f"   Stopovers: {ga_result['best_flight']['stopovers']}")
print(f"   Comfort: {ga_result['best_flight']['comfort']}/10")

print(f"\nConvergence:")
print(f"   Initial Best Cost: {ga_result['history'][0]['best_fitness']:.4f}")
print(f"   Final Best Cost: {ga_result['history'][-1]['best_fitness']:.4f}")
print(f"   Improvement: {ga_result['history'][0]['best_fitness'] - ga_result['history'][-1]['best_fitness']:.4f}")

# Run Particle Swarm Optimization
print("\n" + "=" * 60)
print("PARTICLE SWARM OPTIMIZATION")
print("=" * 60)

pso = ParticleSwarmFlights(problem, n_particles=30, iterations=100)
pso_result = pso.optimize()

print(f"\nBest Flight Found: {pso_result['best_flight']['name']}")
print(f"   Cost Score: {pso_result['best_cost']:.4f}")
print(f"   Price: ${pso_result['best_flight']['price']}")
print(f"   Duration: {pso_result['best_flight']['duration']}h")
print(f"   Stopovers: {pso_result['best_flight']['stopovers']}")
print(f"   Comfort: {pso_result['best_flight']['comfort']}/10")

print(f"\nConvergence:")
print(f"   Initial Best Cost: {pso_result['history'][0]['best_fitness']:.4f}")
print(f"   Final Best Cost: {pso_result['history'][-1]['best_fitness']:.4f}")
print(f"   Improvement: {pso_result['history'][0]['best_fitness'] - pso_result['history'][-1]['best_fitness']:.4f}")

print("\n" + "=" * 60)
print("BOTH ALGORITHMS COMPLETED SUCCESSFULLY")
print("=" * 60)
