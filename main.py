import backend.genetic_algorithm as ga
import matplotlib.pyplot as plt

# just using this as test file for now, but probably useful later
if __name__ == '__main__':
    num_agents = 100
    population = ga.generate_population(num_agents=num_agents)
    best_fitness = [] 
    for n in range(1000):
        best_fitness.append(sorted(population, key=lambda x: x.fitness, reverse=True)[0].fitness)
        population = ga.evolve_population(population, num_agents=num_agents)
    plt.plot(best_fitness)
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness Over Time")
    plt.savefig("./Result Data/fitness_plot.png")  # Save the figure
    print("Plot saved as fitness_plot.png")

