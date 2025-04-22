"""
Script that each minescript agent will autorun upon start
"""
import numpy as np

from minekour.PlayerManager import PlayerManager, init
from minekour.networkReceiver import networkReceiver
from minekour.neural_net import ControlNeuralNetwork
from minekour.PlayerMotion import Motion, MoveType

from minescript import execute

# callback definitions
def reset():
    """
    resets the models position and score
    """
    x = 0
    y = 1
    z = -0
    execute(f"/tp @p {x} {y} {z}")
    execute("/kill @p")

def kill():
    """
    halts minescript function
    """
    execute("\killjob")
    raise SystemExit(0)

def modifyNet(recieved_genes):
    """
    Modifies neural net and returns new neural net from genes recieved
    """
    global net
    net =  ControlNeuralNetwork.from_genes(
        genes=recieved_genes,
        hidden_layer_sizes=hidden_layer_sizes,
        radial_distance=radial_distance)

def runNet():
    """
    runs neural network and controls the player with the output
    """
    global net
    
    inputs = PlayerManager.getBlocksAroundPlayer()
    yaw, pitch = PlayerManager.getRotation()

    results = net.run_nn(block_inputs=inputs, 
                         yaw=yaw, 
                         pitch=pitch)
    
    # convert to playerMotion object
    move = Motion()
    
    move.jumping = results[0]
    move.forward = results[1]
    move.backward = results[2]
    move.left = results[3]
    move.right = results[4]
    
    move.movement_speed = results[5]
    
    move.yaw = 365 * results[6]
    move.pitch = 90 * results[7]
    
    # move player
    PlayerManager.movePlayer(move)

def stopNet():
    """
    executes actions needed when the neural net is stopped
    """
    pass

# parameters
hidden_layer_sizes = [64, 32]
radial_distance = 6
output_size = 8
gene_size = ControlNeuralNetwork.get_gene_size(hidden_layer_sizes, radial_distance)
# Generate random weights (genes)
input_dim = (2*radial_distance+1)**3 + 2
scaling_factor = 1/np.sqrt(input_dim)

# run starts here
init()

random_genes = [np.random.normal(0, scaling_factor) for _ in range(gene_size)]
net = None
modifyNet(random_genes)

networkReceiver.initCallbacks(set_val=modifyNet, 
                              run_model=runNet, 
                              stop_model=stopNet, 
                              get_score=PlayerManager.getScore, 
                              reset=reset,
                              kill=kill)

networkReceiver.initSocket()
networkReceiver.run()