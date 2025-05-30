timeout = 20
radial_distance = 5
hidden_layer_sizes = [256, 128, 128]
output_size = 7

STARTING_GENERATIONS = 0  # will get most recent generation from load file
NUM_AGENTS = 48
NUM_GENERATIONS = 40
SAVE_EVERY = 1  # Save genes every N generations
HIDDEN_LAYER_SIZES = hidden_layer_sizes
RADIAL_DISTANCE = radial_distance
MUTATION_RATE = 0.015
MUTATION_STRENGTH = 0.15
BATCH_SIZE = 8  # Number of agents to evaluate in parallel
TEST_NAME = "straight_map_rebalanced_params"