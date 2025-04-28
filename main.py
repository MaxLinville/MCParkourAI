import json
import os
import numpy as np
import time
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import subprocess
from threading import Thread
import glob
import re

# Import our modules
from mc_interface.minekour.neural_net import ControlNeuralNetwork
from backend.genetic_algorithm import generate_population, evolve_population, evaluate_fitness
from backend.agent import Agent
from backend.minecraft_agents import networkCommander, start_agents
from save_figure import save_figure
from autofocus_windows import autofocus_minecraft
# Configuration
STARTING_GENERATIONS = 112  # need to replace this with getting most recent gen from file
NUM_AGENTS = 48
NUM_GENERATIONS = 100
SAVE_EVERY = 1  # Save genes every N generations
HIDDEN_LAYER_SIZES = [256, 128]
RADIAL_DISTANCE = 5
MUTATION_RATE = 0.01
MUTATION_STRENGTH = 0.1
BATCH_SIZE = 16  # Number of agents to evaluate in parallel
TEST_NAME = "updated_map"
METRICS_FILE = f"fitness_metrics/fitness_metrics_{TEST_NAME}.csv"  # New file for tracking fitness metrics
GENES_FILE = f"backend/weights_{TEST_NAME}"
PYTHON_PATH = "/mnt/c/Users/Max Linville/AppData/Local/Programs/Python/Python313/python.exe"

def save_metrics_to_csv(generation: int, best_fitness: float, avg_fitness: float, file_path: str) -> None:
    """
    Save generation metrics to a CSV file
    
    Args:
        generation (int): Current generation number
        best_fitness (float): Best fitness in the generation
        avg_fitness (float): Average fitness in the generation
        file_path (str): Path to save the CSV file
    """
    file_exists = os.path.exists(file_path)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            # Include important parameters in the header
            writer.writerow(["# Parameters:"])
            writer.writerow(["# Population Size", NUM_AGENTS])
            writer.writerow(["# Generations", NUM_GENERATIONS])
            writer.writerow(["# Mutation Rate", MUTATION_RATE])
            writer.writerow(["# Mutation Strength", MUTATION_STRENGTH])
            writer.writerow(["# Hidden Layers", HIDDEN_LAYER_SIZES])
            writer.writerow(["# Radial Distance", RADIAL_DISTANCE])
            writer.writerow(["# Batch Size", BATCH_SIZE])
            writer.writerow(["# Total Weights: ", ControlNeuralNetwork.get_gene_size(HIDDEN_LAYER_SIZES, RADIAL_DISTANCE)])
            writer.writerow([])  # Empty row for separation
            writer.writerow(["Generation", "Best Fitness", "Average Fitness"])
            
        # Write the metrics for this generation
        writer.writerow([generation + 1, best_fitness, avg_fitness])

def load_genes_from_file(file_path: str) -> Dict[str, List[float]]:
    """
    Load agent genes from the most recent weight file in the weights folder
    
    Args:
        file_path (str): Path to the weights folder
        
    Returns:
        Dict[str, List[float]]: Dictionary mapping agent IDs to their genes
    """
    # Create folder path from the file path
    folder_path = Path(file_path).with_suffix('')
    
    if not folder_path.exists():
        return {}
    
    # Find all weight files in the folder
    weight_files = glob.glob(str(folder_path) + "/*.npz")
    
    if not weight_files:
        return {}
        
    # Extract generation numbers and find the most recent
    gen_numbers = []
    for file in weight_files:
        match = re.search(r'_(\d+)\.npz$', file)
        if match:
            gen_numbers.append(int(match.group(1)))
        else:
            # For files without generation suffix
            gen_numbers.append(0)
    
    if not gen_numbers:
        return {}
        
    # Get the most recent file
    most_recent_idx = gen_numbers.index(max(gen_numbers))
    most_recent_file = weight_files[most_recent_idx]
    
    print(f"Loading weights from {most_recent_file}")
    
    try:
        # Load the compressed numpy file
        with np.load(most_recent_file) as data:
            # Convert to dictionary with numpy arrays
            return {name: data[name] for name in data.files}
    except (IOError, ValueError, EOFError) as e:
        print(f"Error loading genes file: {e}")
        return {}
        
def save_genes_to_file(agents: List[Agent], file_path: str, generation: int = None) -> None:
    """
    Save agent genes to a file in the weights folder
    
    Args:
        agents (List[Agent]): List of agents
        file_path (str): Base path for weights
        generation (int, optional): Current generation number for filename
    """
    try:
        # Create folder path from the file path
        folder_path = Path(file_path).with_suffix('')
        folder_path.mkdir(exist_ok=True)
        
        # Create dict of agent_id -> numpy array
        genes_dict = {f"Agent{i}": np.array(agent.get_genes(), dtype=np.float32) for i, agent in enumerate(agents)}
        
        if generation is not None and generation % 10 == 0:
            # Save with generation suffix for every 10th generation
            save_path = folder_path / f"weights_{generation}.npz"
        else:
            # Save latest weights
            save_path = folder_path / "weights_latest.npz"
            
        # Save as compressed npz file
        np.savez_compressed(save_path, **genes_dict)
        print(f"Saved genes to {save_path}")
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

# Adaptive mutation parameters
def get_adaptive_mutation_params(generation, max_generations):
    """Returns appropriate mutation rate and strength based on generation progress"""
    progress = generation / max_generations
    
    # Linearly decrease from initial to final values
    rate = MUTATION_RATE * np.exp(-0.008*generation) # Decreases from 1% to 0.2% after 200 generations
    strength = MUTATION_STRENGTH * np.exp(-0.008*generation) # Decreases from 0.1 to 0.02 after 200 generations
    
    return rate, strength

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
        
        thread_list: List[Thread] = []

        def set_thread_functions(client_id: int, gene_bytestring: bytes) -> None:
            """
            Thread function to set genes for a client
            
            Args:
                client_id (int): Client ID
                gene_bytestring (bytes): Bytestring of genes
            """
            # Reset the environment for this client
            commander.reset(client_id)
            
            # Send genes to client
            commander.set(client_id, gene_bytestring)  # Using updated set() method
            
            # Start the simulation for this client
            commander.start(client_id)

        # Start all agents in this batch simultaneously
        for i, agent in enumerate(batch_agents):
            client_id = i  # Client ID will be 0-31 (within the batch)
            gene_bytestring = genes_to_bytestring(agent.get_genes())
            setup_thread = Thread(target=set_thread_functions, args=(client_id, gene_bytestring))
            thread_list.append(setup_thread)
            setup_thread.start()

        for thread in thread_list:
            thread.join()  # Wait for all threads to finish
        
        # Wait for all agents in the batch to complete
        completed_agents: set = set()
        # Continue until all agents in this batch have completed or timed out
        while len(completed_agents) < batch_size:
            dead_agents = commander.getDead()
            # Check each active client in the batch
            for dead_agent, time_taken in dead_agents:
                # Skip already completed agents
                if dead_agent in completed_agents:
                    continue              

                print(f"Agent {batch_start + dead_agent} is dead. Time taken: {time_taken}")
                fitness = commander.get(dead_agent)
                # Add time bonus if it is fast (this should be useful only when reaching end states)
                # if fitness > 3:
                #     fitness += 0.2*(10-time_taken) # speed bonus that applies after first 2 checkpoints
                commander.reset(dead_agent)

                # Check if we got a valid fitness value
                if fitness is not None and fitness != -1:
                    # Store fitness in the correct agent
                    agent_idx = batch_start + dead_agent
                    agents[agent_idx].set_fitness(fitness)
                    
                    # Mark as completed
                    completed_agents.add(dead_agent)
                    print(f"Agent {agent_idx} evaluated. Fitness: {fitness}")
            
            # Short delay before checking again
            time.sleep(1)
        print(f"Batch {batch_start // BATCH_SIZE + 1} completed. All agents evaluated.\n")

def main() -> None:
    # Initialize paths
    genes_path = Path(GENES_FILE)
    metrics_path = Path(METRICS_FILE)

    # Load existing genes if available
    genes_dict = load_genes_from_file(genes_path)
    # genes_dict = {}
    
    # Create population
    population = create_population(genes_dict, NUM_AGENTS)
    
    # Initialize network commander for communicating with Minecraft instances
    commander = networkCommander(BATCH_SIZE)
    
    # Wait for Minecraft clients to connect
    print(f"Waiting for {BATCH_SIZE} Minecraft clients to connect...")
    client_wait_thread = Thread(target=commander.wait_for_clients, args=())
    client_wait_thread.start()
    start_agents([_+1 for _ in range(BATCH_SIZE)], "/mnt/c/Users/Max Linville/AppData/Local/Programs/PrismLauncher/prismlauncher.exe")
    client_wait_thread.join()
    print("All clients connected!")
    end_generation = STARTING_GENERATIONS + NUM_GENERATIONS
    # Generation loop
    subprocess.run([PYTHON_PATH, "C:/Users/Max Linville/Desktop/tile_minecraft.py"]) # tiles minecraft windows
    #autofocus windows
    time.sleep(5)
    # open observation window
    subprocess.run(["/mnt/c/Users/Max Linville/AppData/Local/Programs/PrismLauncher/prismlauncher.exe", "--launch", f"1.21.4(2)", "--profile", "MoopleMax"])
    time.sleep(5)
    autofocus_minecraft()


    try:
        for generation in range(STARTING_GENERATIONS,end_generation):
            print(f"\nGeneration {generation+1}/{end_generation}")
            
            # Evaluate agents in Minecraft
            evaluate_agents_in_minecraft(population, commander)
            
            # Display results
            population.sort(key=lambda agent: agent.get_fitness(), reverse=True)
            best_fitness = population[0].get_fitness()
            avg_fitness = sum(agent.get_fitness() for agent in population) / len(population)
            
            print(f"Best fitness: {best_fitness}")
            print(f"Average fitness: {avg_fitness}")
            print(f"Top 5 agents: {[agent.get_fitness() for agent in population[:5]]}")
            
            # Save metrics to CSV
            save_metrics_to_csv(generation, best_fitness, avg_fitness, metrics_path)
            
            # Save genes periodically
            if (generation + 1) % SAVE_EVERY == 0 or generation == NUM_GENERATIONS - 1:
                save_genes_to_file(population, genes_path, generation + 1)
            
            # Adaptive mutation parameters
            mutation_rate, mutation_strength = get_adaptive_mutation_params(generation, NUM_GENERATIONS)
            print(f"Adaptive mutation rate: {mutation_rate}, strength: {mutation_strength}")

            # Create next generation (except for the last iteration)
            if generation < end_generation - 1:
                print(f"Evolving population for generation {generation}...")
                population = evolve_population(
                    population, 
                    num_agents=NUM_AGENTS, 
                    mutation_rate=mutation_rate, 
                    mutation_strength=mutation_strength
                )
            save_figure()
        
        # Final save
        save_genes_to_file(population, genes_path, end_generation)
        print("Genetic algorithm completed!")
        print(f"Fitness metrics saved to {metrics_path}")
    except KeyboardInterrupt:
        print("Manual exit, closing minecraft clients...")
    finally:
        subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", "C:\\Users\\Max Linville\\Desktop\\killminecraft.ps1"])    

if __name__ == "__main__":
    main()