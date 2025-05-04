"""
Neural network module for the Minecraft Parkour AI agent
This network processes environment observations and determines agent actions
"""
import numpy as np
from typing import List, Tuple, Union

class ControlNeuralNetwork:
    """Neural network for Minecraft agent that processes blocks, yaw, and pitch inputs"""
    
    def __init__(self, 
                 weights: List[np.ndarray],
                 hidden_layer_sizes: List[int],
                 radial_distance: int):
        """
        Initialize the neural network with weights and architecture
        
        Args:
            weights: List of weight matrices for each layer
            hidden_layer_sizes: List of integers specifying size of each hidden layer
            radial_distance: Radial distance for cubic block volume (input size = (2*radial_distance+1)^3 + 1)
        """
        self.weights = weights
        self.hidden_layer_sizes = hidden_layer_sizes
        self.radial_distance = radial_distance
        self.input_size = (2 * radial_distance + 1) ** 3 + 3  # Block volume + fractional coordinates
    
    def run_nn(self, 
              block_inputs: List[float],
              fractional_coordinates: tuple[float, float, float]) -> List[Union[bool, int, float]]:
        """
        Process inputs through the neural network
        
        Args:
            block_inputs: List of values representing Minecraft blocks
            yaw: Yaw angle input
            pitch: Pitch angle input
            
        Returns:
            List of 8 outputs:
            - 5 boolean values (True/False)
            - 1 state value (0, 1, or 2)
            - 2 continuous values (float in range [-1, 1])
        """
        # convert Simplified block inputs to int
        block_inputs = [int(block) for block in block_inputs]
        
        # Combine inputs
        inputs = np.array(block_inputs + list(fractional_coordinates))
        
        # Process through hidden layers
        current_layer = inputs
        
        # Process through each hidden layer
        for i in range(len(self.hidden_layer_sizes)):
            # Apply weights and ReLU activation
            current_layer = self._relu(np.dot(current_layer, self.weights[i]))
        
        # Final layer processing
        output_layer = np.dot(current_layer, self.weights[-1])
        
        # Process different output types
        results = []
        
        # 5 binary outputs (using sigmoid and threshold)
        for i in range(5):
            results.append(bool(self._sigmoid(output_layer[i]) > 0.5))
        
        # 1 three-state output (1,2,3)
        three_state_value = self._tanh(output_layer[5]/10000)
        if three_state_value < -0.4:
            results.append(1)
        elif three_state_value < 0.4:
            results.append(2)
        else:
            results.append(3)
        
        # 2 continuous outputs (using tanh to get values in [-1, 1])
        for i in range(6, 7):
            # Apply the periodic transformation: mod(-6arctan(x/10000),pi)-pi/2
            x = output_layer[i]
            arctanx = np.arctan(x / 10000)    # First apply arctan with scaling
            scaled = -6 * arctanx             # Scale the arctan output
            transformed = scaled % np.pi - np.pi/2  # Apply modulo and shift
            # Scale to [-1, 1] range
            compressed = transformed * (2/np.pi)
            results.append(compressed)
        # print(f"Yaw: {output_layer[6]}, Pitch: {output_layer[7]}, movement: {output_layer[5]}")
        return results
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        x_safe = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_safe))
    
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function for continuous outputs"""
        return np.tanh(x)
    
    @classmethod
    def from_genes(cls, 
                  genes: List[float], 
                  hidden_layer_sizes: List[int],
                  radial_distance: int,
                  output_size: int = 7):
        """
        Create a neural network from a flat list of genes
        
        Args:
            genes: Flat list of weights from genetic algorithm
            hidden_layer_sizes: List of integers specifying size of each hidden layer
            radial_distance: Radial distance for cubic block volume (input size = (2*radial_distance+1)^3 +1)
            output_size: Number of output nodes (default 8)
            
        Returns:
            MinecraftNeuralNetwork: Initialized neural network
        """
        input_size = (2 * radial_distance + 1) ** 3 + 3  # Block volume + yaw and pitch
        weights = cls._create_weights_from_genes(genes, input_size, hidden_layer_sizes, output_size)
        return cls(weights, hidden_layer_sizes, radial_distance)
    
    @staticmethod
    def _create_weights_from_genes(genes: List[float], 
                                  input_size: int,
                                  hidden_layer_sizes: List[int],
                                  output_size: int = 7) -> List[np.ndarray]:
        """
        Convert a flat list of genes (weights) into properly shaped weight matrices
        
        Args:
            genes: Flat list of weights from genetic algorithm
            input_size: Number of input nodes (block list size + 1 for yaw and pitch)
            hidden_layer_sizes: List of integers specifying size of each hidden layer
            output_size: Number of output nodes (default 7)
            
        Returns:
            List of weight matrices for each layer of the neural network
        """
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        weights = []
        
        gene_index = 0
        for i in range(len(layer_sizes) - 1):
            rows = layer_sizes[i]
            cols = layer_sizes[i + 1]
            
            # Extract genes for this layer and reshape into matrix
            gene_count = rows * cols
            layer_genes = genes[gene_index:gene_index + gene_count]
            
            if len(layer_genes) < gene_count:
                raise ValueError(f"Not enough genes provided. Need {gene_count} for layer {i}, but only {len(layer_genes)} available.")
                
            weight_matrix = np.array(layer_genes).reshape(rows, cols)
            weights.append(weight_matrix)
            
            gene_index += gene_count
            
        return weights
    
    @staticmethod
    def get_gene_size(hidden_layer_sizes: List[int], 
                     radial_distance: int,
                     output_size: int = 7) -> int:
        """
        Calculate the total number of genes (weights) needed for the network
        
        Args:
            hidden_layer_sizes: List of integers specifying size of each hidden layer
            radial_distance: Radial distance for cubic block volume
            output_size: Number of output nodes (default 8)
            
        Returns:
            Total number of genes needed
        """
        input_size = (2 * radial_distance + 1) ** 3 + 3  # Block volume + yaw and pitch
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        gene_count = 0
        
        for i in range(len(layer_sizes) - 1):
            gene_count += layer_sizes[i] * layer_sizes[i + 1]
            
        return gene_count