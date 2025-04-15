import backend.genetic_algorithm as ga
import backend.neural_net as nn
import matplotlib.pyplot as plt
import numpy as np

# just using this as test file for now, but probably useful later
if __name__ == '__main__':
    # num_agents = 100
    # population = ga.generate_population(num_agents=num_agents)
    # best_fitness = [] 
    # for n in range(1000):
    #     best_fitness.append(sorted(population, key=lambda x: x.fitness, reverse=True)[0].fitness)
    #     population = ga.evolve_population(population, num_agents=num_agents)
    # plt.plot(best_fitness)
    # plt.xlabel("Generations")
    # plt.ylabel("Best Fitness")
    # plt.title("Best Fitness Over Time")
    # plt.savefig("./Result Data/fitness_plot.png")  # Save the figure
    # print("Plot saved as fitness_plot.png")
    # Define network architecture
    radial_distance = 6  # Creates a 5x5x5 block volume (125 blocks) + 2 inputs for yaw/pitch
    hidden_sizes = [64, 32]

    # Calculate gene size for genetic algorithm
    gene_count = nn.ControlNeuralNetwork.get_gene_size(hidden_sizes, radial_distance)
    print(f"Gene count needed: {gene_count}")

    # Generate random genes for this example
    genes = [np.random.normal(0, 1) for _ in range(gene_count)]

    # Create neural network from genes
    neural_net = nn.ControlNeuralNetwork.from_genes(genes, hidden_sizes, radial_distance)

    # Example block inputs (flattened 3D grid)
    block_inputs = [0.0] * (2 * radial_distance + 1) ** 3  # All zeros for this example
    yaw = 45.0
    pitch = 30.0

    # Get actions from network
    actions = neural_net.run_nn(block_inputs, yaw, pitch)
    print(f"Actions: {actions}")

