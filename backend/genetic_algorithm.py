"""
This module contains the core functions for running the genetic algorithm
"""
import numpy as np
from random import choice, choices
from multiprocessing import Pool # this is for later when it becomes ungodly slow
from .agent import Agent, gene_size
from mc_interface.minekour.neural_net import ControlNeuralNetwork
from constants import *

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

def select_population(agents: list[Agent], survivor_ratio: float = 0.2, tournament_size: int = 3) -> list[Agent]:
    """
    Selects a percentage of the population to survive based on their fitness

    Args:
        agents (list[Agent]): List of agents to select from
        survivor_ratio (float, optional): Ratio of agents to choose from population. Defaults to 0.2.

    Returns:
        list[Agent]: List of agents that survived
    """
    num_survivors = int(len(agents) * survivor_ratio)
    survivors = []
    
    # Run tournaments until we have enough survivors
    for _ in range(num_survivors):
        # Select tournament_size random agents
        tournament_agents = np.random.choice(agents, tournament_size, replace=False)
        
        # Find the agent with highest fitness in the tournament
        winner = max(tournament_agents, key=lambda agent: agent.get_fitness())
        survivors.append(winner)
    
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
    # Create a random mask of same shape as genes
    mask = np.random.random(parent1.genes.shape) < 0.5
    
    # Vectorized operation: use mask to select genes from either parent
    new_genes = np.where(mask, parent1.genes, parent2.genes)
    
    return Agent(new_genes)

def evaluate_fitness(agents: list[Agent]) -> None:
    """
    Evaluates and updates the fitness of each agent in the population

    Args:
        agents (list[Agent]): List of agents to evaluate
    """
    # temp for testing is just sum of genes
    for agent in agents:
        # Create neural network from agent's genes
        nn = ControlNeuralNetwork.from_genes(
            agent.get_genes(), 
            hidden_layer_sizes=hidden_layer_sizes, 
            radial_distance=radial_distance
        )
        
        # Here you would evaluate the agent's performance in the environment
        # For now, using placeholder fitness calculation
        agent.set_fitness(sum(agent.get_genes()))  # Replace with actual evaluation
        '''
        SEND  STUFF TO NETWORK TO BE EXEUCTED AND GET BACK FITNESS RESULTS BEFORE SETTING
        '''

def evolve_population(population: list[Agent], num_agents: int = 100, mutation_rate: float = 0.2, mutation_strength: float = 0.2, elite_count: int = 3) -> list:
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
    print(f"Number of agents in population: {len(population)}")
    elite_agents = population[:elite_count]

    trimmed_population = select_population(population[elite_count:])
    if len(trimmed_population) < 2:
        print("Warning: Selection produced too few parents, using all non-elite agents")
        trimmed_population = population[elite_count:]  # Use all non-elite as parents
    print(f"Number of agents in trimmed population: {len(trimmed_population)}")
    
    # Create new population with vectorized operations
    new_population: list[Agent] = []
    for _ in range(num_agents):
        parent1: Agent = choice(trimmed_population)
        parent2: Agent = choice(trimmed_population)
        new_population.append(crossover(parent1, parent2))
    
    # Apply mutations
    for agent in new_population:
        agent.mutate(mutation_rate, mutation_strength)
    
    # Combine elite agents with new population
    new_population = elite_agents + new_population[:num_agents - elite_count]
    return new_population
