from backend.neural_net import ControlNeuralNetwork
from backend.genetic_algorithm import generate_population, select_population, crossover, evaluate_fitness
from backend.agent import Agent
import numpy as np

# create an agent with random genes of the known size
#     agents (list[Agent]): List of agents to evaluate

if __name__ == "__main__":
    test_agent = Agent([np.random.normal(0, 1) for _ in range(ControlNeuralNetwork.get_gene_size([64, 32], 6))])
    print(len(test_agent.genes))

