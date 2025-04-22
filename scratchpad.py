import numpy as np
from backend.neural_net import ControlNeuralNetwork

# Set the same parameters as in the main program
hidden_layer_sizes = [64, 32]
radial_distance = 6
output_size = 8

# Calculate gene size needed for the network
gene_size = ControlNeuralNetwork.get_gene_size(hidden_layer_sizes, radial_distance)
print(f"Gene size: {gene_size}")

# Generate random weights (genes)
input_dim = (2*radial_distance+1)**3 + 2
scaling_factor = 1/np.sqrt(input_dim)
random_genes = [np.random.normal(0, scaling_factor) for _ in range(gene_size)]

# Create neural network from genes
nn = ControlNeuralNetwork.from_genes(
    genes=random_genes,
    hidden_layer_sizes=hidden_layer_sizes,
    radial_distance=radial_distance
)

# Create test inputs
# For block inputs, we need (2*radial_distance+1)^3 values (representing the cube of blocks)
# Let's generate random block values (in a real scenario these would be actual block types)
cube_size = (2*radial_distance+1)**3

# Generate random block type values (1-13)

test_block_inputs = [_%13+1 for _ in range(cube_size)]  # Random block type values

# Test with a few different yaw/pitch combinations
test_cases = [
    {"yaw": 0.0, "pitch": 0.0, "description": "Looking straight forward"},
    {"yaw": 90.0, "pitch": 0.0, "description": "Looking right"},
    {"yaw": 180.0, "pitch": 0.0, "description": "Looking backward"},
    {"yaw": 27.5, "pitch": -45.0, "description": "Looking up"},
    {"yaw": 0.0, "pitch": 45.0, "description": "Looking down"}
]

# Run each test case through the network
print("\nNeural Network Test Results:")
print("-" * 50)

for case in test_cases:
    yaw = case["yaw"]
    pitch = case["pitch"]
    desc = case["description"]
    normalized_block_inputs = [val/13 for val in test_block_inputs]  # Assuming max block type is 13
    normalized_yaw = yaw / 360  # Scale to [0,1]
    normalized_pitch = (pitch + 90) / 180  # Scale from [-90,90] to [0,1]
    inputs = np.array(normalized_block_inputs + [normalized_yaw, normalized_pitch])
    # Run the network
    outputs = nn.run_nn(normalized_block_inputs, normalized_yaw, normalized_pitch)
    
    # Parse outputs
    binary_outputs = outputs[:5]
    state_output = outputs[5]
    continuous_outputs = outputs[6:8]
    
    # Display results
    print(f"\nTest: {desc} (yaw={yaw}, pitch={pitch})")
    print(f"  Binary outputs (actions): {binary_outputs}")
    print(f"  State output (movement type): {state_output}")
    print(f"  Continuous outputs (direction): {continuous_outputs}")

print("\nTest complete!")