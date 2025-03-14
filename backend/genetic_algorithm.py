import numpy as np
from multiprocessing import Pool
from . import agent

'''
Genetic Algorithm Core Functions
'''

def generate_population(num_agents: int = 100) -> list:
    '''
    Creates a list of agents with random genes
    '''
    agents = [agent.Agent([np.random.normal(0, 1) for _ in range(agent.gene_size)]) for _ in range(num_agents)]
    return agents

def select_population(agents: list = [], survivor_percentage: float = 0.2) -> list:
    '''
    Returns survivors weighted towards higher fitness agents
    '''
    agent_weights = [agent.get_fitness() for agent in agents]
    agent_weights_shifted = [weight - min(agent_weights) for weight in agent_weights]
    normalized_weights = [weight / sum(agent_weights_shifted) for weight in agent_weights_shifted]
    return np.random.choice(agents, int(len(agents) * survivor_percentage), p=normalized_weights)

def crossover(parent1: agent.Agent, parent2: agent.Agent) -> agent.Agent:
    '''
    Creates a new agent by combining the genes of two parents
    '''
    # randomly select which parent the gene comes from
    new_genes = [parent1.genes[i] if np.random.uniform() < 0.5 else parent2.genes[i] for i in range(agent.gene_size)]
    return agent.Agent(new_genes)

def evaluate_fitness(agents: list) -> None:
    '''
    The fitness function that evaluates an agent's performance

    distance across course, average speed (to account for death time), course completion (high reward)
    to minimize backtracking, we could add a penalty for going backwards by saving all positions and checking if the next position is behind the last
    '''

    # temp for testing is just sum of genes
    for agent in agents:
        agent.set_fitness(sum(agent.get_genes()))

def evolve_population(population: list, num_agents: int = 100, mutation_rate: float = 0.2, mutation_strength: float = 0.2) -> list:
    '''
    Creates a new generation of agents by combining the genes of survivors
    '''
    evaluate_fitness(population)
    trimmed_population = select_population(population)
    new_population = []
    for _ in range(num_agents):
        parent1 = np.random.choice(trimmed_population)
        parent2 = np.random.choice(trimmed_population)
        new_population.append(crossover(parent1, parent2))
    for agent in new_population:
        agent.mutate(mutation_rate, mutation_strength)
    evaluate_fitness(new_population)
    return new_population
