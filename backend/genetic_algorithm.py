"""
This module contains the core functions for running the genetic algorithm
"""
import numpy as np
from random import choice, choices
from multiprocessing import Pool # this is for later when it becomes ungodly slow
from .agent import Agent, gene_size

def generate_population(num_agents: int = 100) -> list[Agent]:
    """
    Creates a population of agents with random genes

    Args:
        num_agents (int, optional): Number of agents to create. Defaults to 100.

    Returns:
        list[Agent]: List of agents
    """
    agents = [Agent([np.random.normal(0, 1) for _ in range(gene_size)]) for _ in range(num_agents)]
    return agents

def select_population(agents: list[Agent], survivor_ratio: float = 0.2) -> list[Agent]:
    """
    Selects a percentage of the population to survive based on their fitness

    Args:
        agents (list[Agent]): List of agents to select from
        survivor_ratio (float, optional): Ratio of agents to choose from population. Defaults to 0.2.

    Returns:
        list[Agent]: List of agents that survived
    """
    agent_weights = [agent.get_fitness() for agent in agents]
    agent_weights_shifted = [weight - min(agent_weights) for weight in agent_weights]
    normalized_weights = [weight / sum(agent_weights_shifted) for weight in agent_weights_shifted]
    survivors = choices(agents, weights=normalized_weights, k=int(len(agents) * survivor_ratio))

    return survivors

def crossover(parent1: Agent, parent2: Agent) -> Agent:
    """
    Creates a new agent by combining the genes of two parents

    Args:
        parent1 (Agent): First parent agent
        parent2 (Agent): Second parent agent

    Returns:
        Agent: New agent with genes from parents
    """
    # randomly select which parent the gene comes from
    new_genes = [parent1.genes[i] if np.random.uniform() < 0.5 else parent2.genes[i] for i in range(gene_size)]
    return Agent(new_genes)

def evaluate_fitness(agents: list[Agent]) -> None:
    """
    Evaluates and updates the fitness of each agent in the population

    Args:
        agents (list[Agent]): List of agents to evaluate
    """
    # temp for testing is just sum of genes
    for agent in agents:
        agent.set_fitness(sum(agent.get_genes()))

def evolve_population(population: list, num_agents: int = 100, mutation_rate: float = 0.2, mutation_strength: float = 0.2) -> list:
    """
    Evolves the population by selecting, crossing over, and mutating agents

    Args:
        population (list): List of agents to evolve
        num_agents (int, optional): Numer of agents to generate in the new population. Defaults to 100.
        mutation_rate (float, optional): Fraction of random genes that will have a mutation. Defaults to 0.2.
        mutation_strength (float, optional): Scaling factor for amount mutated genes will change by. Defaults to 0.2.

    Returns:
        list: New population of agents
    """    
    evaluate_fitness(population)
    trimmed_population = select_population(population)
    new_population: list[Agent] = []
    for _ in range(num_agents):
        parent1: Agent = choice(trimmed_population)
        parent2: Agent = choice(trimmed_population)
        new_population.append(crossover(parent1, parent2))
    for agent in new_population:
        agent.mutate(mutation_rate, mutation_strength)
    evaluate_fitness(new_population)
    return new_population
