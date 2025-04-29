"""
Script that each minescript agent will autorun upon start
"""
import numpy as np
import time
import socket

from minekour.PlayerManager import PlayerManager, init
from minekour.networkReceiver import networkReceiver, BUFFER_SIZE, ENCODING
from minekour.neural_net import ControlNeuralNetwork
from minekour.PlayerMotion import Motion, MoveType

from minescript import execute, player_position, echo

# callback definitions
def reset(sock: socket):
    """
    resets the models position and score
    """
    x = 0
    y = 1
    z = -0
    execute(f"/tp @p {x} {y} {z}")
    execute("/kill @p")

def kill(sock: socket):
    """
    halts minescript function
    """
    execute("\\killjob")
    raise SystemExit(0)

def modifyNet(recieved_genes):
    """
    Modifies neural net and returns new neural net from genes recieved
    """
    global net
    
    if hidden_layer_sizes is None:
        echo("hidden_layer_sizes was None, returning")
        return
    if radial_distance is None:
        echo("radial_distance was None, returning")
        return
    
    net =  ControlNeuralNetwork.from_genes(
        genes=recieved_genes,
        hidden_layer_sizes=hidden_layer_sizes,
        radial_distance=radial_distance)

def checkDead() -> tuple[bool, float]:
    """
    Checks if the player is in the dead zone and returns true if so
    """
    global last_run_time
    if (last_run_time == -1):
        echo("last_run_time was -1, returning false")
        return (False, -1,)
    
    death_loc = np.array((23,1,-1))
    death_tolerance = 2
    
    pos = np.array(player_position())
    
    at_death =  abs(np.linalg.norm(death_loc - pos)) <= death_tolerance
    current_time = time.time()
    runtime = current_time - last_run_time
    death_time = runtime > timeout
    
    is_dead = at_death or death_time

    if is_dead:
        echo(f"at_death: {at_death}, death_time: {death_time}")
        echo(f"last_run_time: {last_run_time}, current time: {current_time}")
        last_run_time = -1
    return (is_dead, runtime,)

def runNet():
    """
    runs neural network and controls the player with the output
    """
    global net
    global last_run_time
    
    if last_run_time == -1:
        echo("last_run_time was -1, setting to current time")
        last_run_time = time.time()
        echo(f"current time: {last_run_time}")
    
    # run network and move
    inputs = PlayerManager.getBlocksAroundPlayer()
    yaw, pitch = PlayerManager.getRotation()

    results = net.run_nn(block_inputs=inputs, 
                         yaw=yaw)
    
    # convert to playerMotion object
    move = Motion()
    
    move.jumping = results[0]
    move.forward = results[1]
    move.backward = results[2]
    move.left = results[3]
    move.right = results[4]
    
    move.movement_speed = results[5]
    
    move.yaw = 180 * results[6]
    move.pitch = 0
    
    # move player
    PlayerManager.movePlayer(move)

def stopNet():
    """
    executes actions needed when the neural net is stopped
    """
    pass

def setConsts(sock: socket):
    """
    sets the constants for the neural network in the following order
    
    hidden_layer_sizes, radial_distance, timeout
    """
    global hidden_layer_sizes
    global radial_distance
    global input_dim
    global scaling_factor
    global timeout
    global gene_size
    
    # set constants from socket data
    value = sock.recv(BUFFER_SIZE).decode(ENCODING)
    hidden = list()
    
    while value != "STOP":
        sock.send("OK".encode(ENCODING))
        hidden.append(int(value))
        value = sock.recv(BUFFER_SIZE).decode(ENCODING)
    
    sock.send("OK".encode(ENCODING))
    
    radial = int(sock.recv(BUFFER_SIZE).decode(ENCODING))
    sock.send("OK".encode(ENCODING))
    time = int(sock.recv(BUFFER_SIZE).decode(ENCODING))
    sock.send("OK".encode(ENCODING))
    
    hidden_layer_sizes = hidden
    radial_distance = radial
    timeout = time
    
    input_dim = (2*radial_distance+1)**3 + 1
    scaling_factor = 1/np.sqrt(input_dim)
    gene_size = ControlNeuralNetwork.get_gene_size(hidden_layer_sizes, radial_distance)
    
# parameters
hidden_layer_sizes = None
radial_distance = None
output_size = 7
gene_size = None
#gene_size = ControlNeuralNetwork.get_gene_size(hidden_layer_sizes, radial_distance)
# Generate random weights (genes)

input_dim = None
scaling_factor = None
timeout = None
#input_dim = (2*radial_distance+1)**3 + 1
#scaling_factor = 1/np.sqrt(input_dim)
#timeout = 30 #seconds before giving up

# run starts here
init()

#random_genes = [np.random.normal(0, scaling_factor) for _ in range(gene_size)]
net = None
#modifyNet(random_genes)

last_run_time = -1

funcs = dict()

funcs["SET_CONST"] = setConsts

networkReceiver.initCallbacks(otherFunctions = funcs,
                              set_val=modifyNet, 
                              run_model=runNet, 
                              stop_model=stopNet, 
                              get_score=PlayerManager.getScore, 
                              reset=reset,
                              kill=kill,
                              dead=checkDead)

networkReceiver.initSocket("127.0.0.1", 25567, 25568)
networkReceiver.run()