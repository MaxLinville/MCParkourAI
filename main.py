import json
import os
import numpy as np
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import our modules
from mc_interface.minekour.neural_net import ControlNeuralNetwork
from backend.genetic_algorithm import generate_population, evolve_population, evaluate_fitness
from backend.agent import Agent
from backend.minecraft_agents import networkCommander

# Configuration
NUM_AGENTS = 2
NUM_GENERATIONS = 5
SAVE_EVERY = 1  # Save genes every N generations
GENES_FILE = "backend/weights.json"
HIDDEN_LAYER_SIZES = [64, 32]
RADIAL_DISTANCE = 6
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.5
BATCH_SIZE = 1  # Number of agents to evaluate in parallel

def load_genes_from_file(file_path: str) -> Dict[str, List[float]]:
    """
    Load agent genes from a JSON file
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Dict[str, List[float]]: Dictionary mapping agent IDs to their genes
    """
    if not os.path.exists(file_path):
        return {}
        
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading genes file: {e}")
        return {}
        
def save_genes_to_file(agents: List[Agent], file_path: str) -> None:
    """
    Save agent genes to a JSON file
    
    Args:
        agents (List[Agent]): List of agents
        file_path (str): Path to save the JSON file
    """
    genes_dict = {f"Agent{i}": agent.get_genes() for i, agent in enumerate(agents)}
    
    try:
        with open(file_path, 'w') as f:
            json.dump(genes_dict, f, indent=2)
        print(f"Saved genes to {file_path}")
    except IOError as e:
        print(f"Error saving genes file: {e}")

def create_population(genes_dict: Dict[str, List[float]], num_agents: int) -> List[Agent]:
    """
    Create a population of agents using existing genes if available
    
    Args:
        genes_dict (Dict[str, List[float]]): Dictionary of agent genes
        num_agents (int): Number of agents in the population
        
    Returns:
        List[Agent]: List of agents
    """
    population = []
    gene_size = ControlNeuralNetwork.get_gene_size(HIDDEN_LAYER_SIZES, RADIAL_DISTANCE)
    
    for i in range(num_agents):
        agent_key = f"Agent{i}"
        if agent_key in genes_dict:
            population.append(Agent(genes_dict[agent_key]))
        else:
            # Create agent with random genes
            random_genes = [np.random.normal(0, 1) for _ in range(gene_size)]
            population.append(Agent(random_genes))
            
    return population

def genes_to_bytestring(genes: List[float]) -> bytes:
    """
    Convert genes to a bytestring for network transmission
    
    Args:
        genes (List[float]): List of gene values
        
    Returns:
        bytes: Bytestring representation
    """
    # Convert genes to a string representation and encode it
    genes_str = ";".join([str(gene) for gene in genes])
    return genes_str.encode('utf-8')

def evaluate_agents_in_minecraft(agents: List[Agent], commander: networkCommander) -> None:
    """
    Send agents to Minecraft and evaluate their fitness, using up to 32 instances in parallel
    
    Args:
        agents (List[Agent]): List of agents to evaluate
        commander (networkCommander): Network commander for Minecraft communication
    """
    # Process agents in batches of up to batch size
    for batch_start in range(0, len(agents), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(agents))
        batch_agents = agents[batch_start:batch_end]
        batch_size = len(batch_agents)
        
        print(f"Evaluating batch of {batch_size} agents (indices {batch_start}-{batch_end-1})")
        
        # Start all agents in this batch simultaneously
        for i, agent in enumerate(batch_agents):
            client_id = i  # Client ID will be 0-31 (within the batch)
            
            # Reset the environment for this client
            commander.reset(client_id)
            
            # Send genes to client
            gene_bytestring = genes_to_bytestring(agent.get_genes())
            commander.set(client_id, gene_bytestring)  # Using updated set() method
            
            # Start the simulation for this client
            commander.start(client_id)
            
        # Wait for all agents in the batch to complete
        completed_agents: set = set()
        # Continue until all agents in this batch have completed or timed out
        while len(completed_agents) < batch_size:
            dead_agents = commander.getDead()
            print(f"Dead agents: {dead_agents}")
            # Check each active client in the batch
            for dead_agent in dead_agents:
                # Skip already completed agents
                if dead_agent in completed_agents:
                    continue              

                print(f"Agent {dead_agent} has completed.")
                fitness = commander.get(dead_agent)
                commander.reset(dead_agent)

                # Check if we got a valid fitness value
                if fitness is not None and fitness != -1:
                    # Store fitness in the correct agent
                    agent_idx = batch_start + i
                    agents[agent_idx].set_fitness(fitness)
                    
                    # Mark as completed
                    completed_agents.add(dead_agent)
                    print(f"Agent {agent_idx} evaluated. Fitness: {fitness}")
            
            # Short delay before checking again
            time.sleep(1)
        print(f"Batch {batch_start // BATCH_SIZE + 1} completed. All agents evaluated.")

def main() -> None:
    # Initialize paths
    genes_path = Path(GENES_FILE)
    
    # Load existing genes if available
    genes_dict = load_genes_from_file(genes_path)
    
    # Create population
    population = create_population(genes_dict, NUM_AGENTS)
    
    # Initialize network commander for communicating with Minecraft instances
    commander = networkCommander(BATCH_SIZE)
    
    # Wait for Minecraft clients to connect
    print(f"Waiting for {BATCH_SIZE} Minecraft clients to connect...")
    commander.wait_for_clients()
    print("All clients connected!")
    
    # Generation loop
    for generation in range(NUM_GENERATIONS):
        print(f"\nGeneration {generation+1}/{NUM_GENERATIONS}")
        
        # Evaluate agents in Minecraft
        evaluate_agents_in_minecraft(population, commander)
        
        # Display results
        population.sort(key=lambda agent: agent.get_fitness(), reverse=True)
        print(f"Best fitness: {population[0].get_fitness()}")
        print(f"Average fitness: {sum(agent.get_fitness() for agent in population) / len(population)}")
        
        # Save genes periodically
        if (generation + 1) % SAVE_EVERY == 0 or generation == NUM_GENERATIONS - 1:
            save_genes_to_file(population, genes_path)
        
        # Create next generation (except for the last iteration)
        if generation < NUM_GENERATIONS - 1:
            population = evolve_population(
                population, 
                num_agents=NUM_AGENTS, 
                mutation_rate=MUTATION_RATE, 
                mutation_strength=MUTATION_STRENGTH
            )
    
    # Final save
    save_genes_to_file(population, genes_path)
    print("Genetic algorithm completed!")

if __name__ == "__main__":
    main()