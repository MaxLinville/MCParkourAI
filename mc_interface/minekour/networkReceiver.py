import socket
import selectors

from minescript import echo

"""
This file defines functions responsible for communicating with the evolution model and executing instructions
"""

DEFAULT_CONTROL_PORT = 25567
DEFAULT_DEAD_PORT = 25568
CONTROL_SOCKET_FAMILY = socket.AF_INET #IPV4
CONTROL_SOCKET_TYPE = socket.SOCK_STREAM #TCP

BUFFER_SIZE = 2**12
ENCODING = "UTF-8"

SELECTOR_DELAY = 2 #2 seconds
"""
Commands and their descriptions

RESET
    Stops current model from running and resets it to the base position and score
GET
    Gets current score of model, model will return OK then return the values
SET
    Sets models new parameters, follwed by parameters and ending with STOP
    
    STOP
        End of list of parameters
START
    Starts running the neural network
OK
    Confirms message was recieved
KILL
    kills the main loop    

TCP Implementation Details
    Between each command and response an OK will be sent, this includes for Set and in between each parameter of SET
    For example:
        ->GET
        <-OK
        <-SCORE
        ->OK
        
    Example 2:
        ->SET
        <-OK
        ->Parameter1
        <-OK
        ->Parameter2
        ...
        <-OK
        ->STOP
        <-OK

Note that any command while the model is running will implicitly stop the model from running

DEAD SOCKET
    Dead socket will not expect any response from the server, this is a write-only socket

    example:
        *player dies*
        <- "DEAD"

CONNECTION PROTOCOL
    When connecting the server will send an ID to the client, the client will then connect to the DEAD_SOCKET
    and send this ID

    example:
        <- Connect to Control Socket
        -> [ID]
        
        <- Connect to Dead Socket
        <- [ID]

"""

class networkReceiver:
    """
    Class for managing network connections and parsing information from the evolution model
    This class is modeled as a singleton as only program will run per client
    """
    
    control_socket: socket = None
    dead_socket: socket = None
    selector = selectors.DefaultSelector()
    
    #callbacks will be called whenever the command is recieved, each callback is mapped to one command
    #note that the SET and RUN callback will be given its own special callback as this will be how the model interacts with Player
    command_map: dict = None
    callback_set_values: callable = None # for setting parameters of the model, this is expected to be some setter function for the neural network
    callback_run_model: callable = None # function to call to run the model again
    callback_stop_model: callable = None # function to call when the model is stopped
    callback_get_score: callable = None
    callback_is_dead: callable = None
    
    isRunning: bool = False # used to indicate whether or not the model is currently running
    
    id = -1
    
    @staticmethod
    def initCallbacks(set_val: callable, run_model: callable, stop_model: callable, get_score: callable, reset: callable, kill: callable, dead: callable):
        """
        Initializes callback dictionary with commands
        """
        networkReceiver.command_map = dict()
        networkReceiver.callback_set_values = set_val
        networkReceiver.callback_run_model = run_model
        networkReceiver.callback_stop_model = stop_model
        networkReceiver.callback_get_score = get_score
        networkReceiver.callback_is_dead = dead
        
        networkReceiver.command_map["SET"] = networkReceiver.setValues
        networkReceiver.command_map["GET"] = networkReceiver.getScore
        networkReceiver.command_map["START"] = networkReceiver.startModel
        networkReceiver.command_map["RESET"] = reset
        networkReceiver.command_map["KILL"] = kill
    
    @staticmethod
    def initSocket(address: str = "", port: int = DEFAULT_CONTROL_PORT, dead_port = DEFAULT_DEAD_PORT):
        """
        Opens socket to the given address and starts listening
        """
        try:
            address_control = (address, port)
            address_dead = (address, dead_port)
            
            # build socket and attempt to connect
            networkReceiver.control_socket = socket.socket(CONTROL_SOCKET_FAMILY, CONTROL_SOCKET_TYPE)
            networkReceiver.control_socket.connect(address_control)
            
            # get id
            networkReceiver.id = int(networkReceiver.control_socket.recv(BUFFER_SIZE).decode(ENCODING))
            
            # add socket to selector
            networkReceiver.selector = selectors.DefaultSelector()
            networkReceiver.selector.register(networkReceiver.control_socket, selectors.EVENT_READ, networkReceiver.recieve)
            
            # build dead socket and attempt to connect
            networkReceiver.dead_socket = socket.socket(CONTROL_SOCKET_FAMILY, CONTROL_SOCKET_TYPE)
            networkReceiver.dead_socket.connect(address_dead)
            
            # send ID to dead socket
            networkReceiver.dead_socket.send(str(networkReceiver.id).encode(ENCODING))
            
            echo(f"Successfully connected to {address_control}")
            
        except OSError as err:
            echo(f"FAILED TO CONNECT TO {address_control} OR {address_dead}")
            raise SystemExit(1)
        
    @staticmethod
    def run():
        """
        Main loop that runs both recieving network information and running
        """
        networkReceiver.isRunning = False
        
        while True:
            #execute
            if networkReceiver.isRunning:
                networkReceiver.checkDead()
            
            if networkReceiver.isRunning:
                networkReceiver.callback_run_model()
                
            #since select is 0 shouldnt ever block
            for selectable, _ in networkReceiver.selector.select(SELECTOR_DELAY * int(not networkReceiver.isRunning)):
                networkReceiver.isRunning = False
                networkReceiver.callback_stop_model()
                
                call = selectable.data
                sock = selectable.fileobj
                
                call(sock)
        
    @staticmethod
    def recieve(sock: socket):
        """
        This function takes the socket reads the next input and determines the next command to run, also sends OK
        """
        
        try:
            data = sock.recv(BUFFER_SIZE).decode(ENCODING)
            sock.send("OK".encode(ENCODING))
            
            echo(f"Recived {data} command")
            
            func = networkReceiver.command_map[data]
            func()
            
        except KeyError as err:
            echo(f"Warning: recieved malformed command from server: {data}")
        except OSError as err:
            echo(f"FAILED TO COMMUNICATE WITH SERVER: {err}")
            raise SystemExit(1)
        
    @staticmethod
    def setValues():
        """
        Receives gene parameters from the network until STOP command is received
        Passes the received parameters to the callback_set_values function
        """
        gene_parameters = []
        sock = networkReceiver.control_socket
        
        while True:
            # Receive parameter
            data = sock.recv(BUFFER_SIZE).decode(ENCODING)
            
            # Check if we've received the STOP command
            if data == "STOP":
                sock.send("OK".encode(ENCODING))
                break
            
            # Try to convert the received data to a float (gene value)
            try:
                # For gene values transmitted as strings with delimiter ";"
                if ";" in data:
                    values = [float(val) for val in data.split(";") if val]
                    gene_parameters.extend(values)
                else:
                    # Single value
                    gene_parameters.append(float(data))
                    
            except ValueError:
                echo(f"Warning: Received non-numeric parameter: {data}")
            
            # Send OK to acknowledge receipt
            sock.send("OK".encode(ENCODING))
        
        # Process the received parameters using the callback
        if networkReceiver.callback_set_values and gene_parameters:
            networkReceiver.callback_set_values(gene_parameters)
            echo(f"Set {len(gene_parameters)} gene parameters")
        else:
            echo("Warning: No parameters received or callback not set")
        
    @staticmethod
    def getScore():
        """
        Gets the score from minecraft scoreboard of the specific player
        """
        score = networkReceiver.callback_get_score()
        networkReceiver.control_socket.send(str(score).encode(ENCODING))
        
        response = networkReceiver.control_socket.recv(BUFFER_SIZE).decode(ENCODING)
        if response != "OK":
            echo("Warning: Non OK recieved on getScore")
            
    @staticmethod
    def startModel():
        """
        Starts the running of the model
        """
        networkReceiver.isRunning = True
        
    @staticmethod
    def checkDead():
        """
        Checks if player is dead using a callback and then sends "DEAD" to the server and stops the model
        """
        dead, time = networkReceiver.callback_is_dead()
        if dead:
            networkReceiver.dead_socket.send(str(time).encode(ENCODING))
            networkReceiver.isRunning = False
            echo("PLAYER IS DEAD")