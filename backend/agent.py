"""
Agent class to track genes and fitness of each agent as an object
"""
import numpy as np
from mc_interface.minekour.neural_net import ControlNeuralNetwork

# global variables
# Global variables - use neural network to determine gene size
radial_distance = 6
hidden_layer_sizes = [256, 128]
gene_size = ControlNeuralNetwork.get_gene_size(hidden_layer_sizes, radial_distance)

class Agent:
    def __init__(self, genes: list[float]):
        """
        Initializes an agent with genes and fitness

        Args:
            genes (list[float]): List of genes for the agent
        """
        self.genes: list = genes if genes else print("No genes provided, using random genes")
        self.fitness: float = 0.0

    def __str__(self) -> str:
        """
        Returns:
            str: String representation of the agent in a readable format
        """        
        return f'Genes: {self.genes}, Fitness: {self.fitness}'

    def __repr__(self):
        """

        Returns:
            _type_: String representation of specific agent (for within containers mostly)
        """        
        return f"<Agent object {id(self)!r}, Genes: {self.genes!r}, Fitness: {self.fitness!r}>"

    def set_fitness(self, new_fitness: float = 0.0) -> None:
        """
        Sets the fitness of the agent

        Args:
            new_fitness (float, optional): Fitness value to assign to agent. Defaults to 0.0.
        """        
        self.fitness = new_fitness

    def get_fitness(self) -> float:
        """
        Gets the fitness of the agent

        Returns:
            float: Fitness value of the agent
        """        
        return self.fitness

    def set_genes(self, new_genes: list[float]) -> None:
        """
        Sets the genes of the agent

        Args:
            new_genes (list[float]): List of genes to assign to agent
        """        
        self.genes = new_genes

    def get_genes(self) -> list[float]:
        """
        Gets the genes of the agent

        Returns:
            list[float]: List of genes for the agent
        """        
        return self.genes    

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 1) -> None:
        """
        Mutates the genes of the agent based on a mutation rate and strength

        Args:
            mutation_rate (float, optional): Ratio of genes to mutate. Defaults to 0.1.
            mutation_strength (float, optional): Scaling factor for mutation amount for modified genes. Defaults to 1.
        """        
        for gene_index, gene in enumerate(self.genes):
            if np.random.uniform() < mutation_rate:
                mutation_amount: float = np.random.normal(0, mutation_strength)
                self.genes[gene_index] += mutation_amount
        


