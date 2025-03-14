'''
This represents all of the parameters assigned to a specific player
'''
import numpy as np

# global variables
gene_size = 10

class Agent:
    def __init__(self, genes: list):
        '''
        Genes: a list of floats representing the agent's parameters
        The genes are passed into a function that is used to determine movements based on environmental input
        '''
        self.genes: list = genes if genes else [0 for _ in range(gene_size)]
        self.fitness: float = 0.0

    def __str__(self) -> str:
        '''
        Returns a string formatted output of the agent's genes and fitness,
        called when the print() function is used on the object itself
        '''
        return f'Genes: {self.genes}, Fitness: {self.fitness}'

    def __repr__(self):
        '''
        Returns a string formatted output of the agent's genes and fitness,
        called when the object is printed in a container.
        '''
        return f"<Agent object {id(self)}>"

    def set_fitness(self, new_fitness: float = 0.0) -> None:
        '''
        Setter function to directly set the agent's fitness
        '''
        self.fitness = new_fitness

    def get_fitness(self) -> float:
        '''
        Getter function for agent's fitness
        '''
        return self.fitness

    def set_genes(self, new_genes: list) -> None:
        '''
        Setter function to directly set the agent's genes
        '''
        self.genes = new_genes

    def get_genes(self) -> dict:
        '''
        Getter function for agent's genes
        '''
        return self.genes    

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 1) -> None:
        '''
        Randomly mutates the agent's genes at a certain rate and strength
        '''
        for gene_index, gene in enumerate(self.genes):
            if np.random.uniform() < mutation_rate:
                mutation_amount = np.random.normal(0, mutation_strength)
                self.genes[gene_index] += mutation_amount
        


