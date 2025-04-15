import socket
import selectors

from minescript import echo

"""
This file defines functions responsible for communicating with the evolution model and executing instructions
"""

DEFAULT_CONTROL_PORT = 25567
CONTROL_SOCKET_FAMILY = socket.AF_INET #IPV4
CONTROL_SOCKET_TYPE = socket.SOCK_STREAM #TCP

BUFFER_SIZE = 2**12
ENCODING = "UTF-8"

"""
Commands and their descriptions

RESET
    Stops current model from running and resets it to the base position
GET
    Gets current score of model
SET
    Sets models new parameters, follwed by parameters and ending with STOP
    
    Model Paramters are as follows:
        #TODO: 
    
STOP
    End of list of parameters
START
    Starts running the neural network
OK
    Confirms message was recieved
KILL
    kills the main loop    

TCP Implementation Details
    Each command will be sent individually then an OK will be recieved, then the next command may be sent.
    This includes for SET after each parameter set will wait for an OK

Note that any command while the model is running will implicitly stop the model from running
"""


class networkReciever:
    """
    Class for managing network connections and parsing information from the evolution model
    This class is modeled as a singleton as only program will run per client
    """
    
    control_socket: socket = None
    selector = selectors.DefaultSelector()
    
    #callbacks will be called whenever the command is recieved, each callback is mapped to one command
    #note that the SET and RUN callback will be given its own special callback as this will be how the model interacts with Player
    command_map: dict = None
    callback_set_values: function = None # for setting parameters of the model, this is expected to be some setter function for the neural network
    callback_run_model: function = None # function to call to run the model again
    callback_stop_model: function = None # function to call when the model is stopped
    
    isRunning: bool = False # used to indicate whether or not the model is currently running
    
    def initCallbacks(commands: dict, set_val: function, run_model: function, stop_model: function):
        """
        Initializes callback dictionary with commands
        """
        networkReciever.command_map = commands
        networkReciever.callback_set_values = set_val
        networkReciever.callback_run_model = run_model
        networkReciever.callback_stop_model = stop_model
        
        networkReciever.command_map["SET"] = networkReciever.setValues
        
    
    def initSocket(address: str, port: int = DEFAULT_CONTROL_PORT):
        """
        Opens socket to the given address and starts listening
        """
        try:
            address_control = (address, port)
            
            # build socket and attempt to connect
            networkReciever.control_socket = socket.socket(CONTROL_SOCKET_FAMILY, CONTROL_SOCKET_TYPE)
            networkReciever.control_socket.connect(address_control)
            
            # add socket to selector
            networkReciever.selector = selectors.DefaultSelector()
            networkReciever.selector.register(networkReciever.control_socket, selectors.EVENT_READ, networkReciever.recieve)
            
            echo(f"Sucessfully connected to {address_control}")
            
        except OSError as err:
            echo(f"FAILED TO CONNECT TO {address_control}")
            raise SystemExit(1)
        
    def run():
        """
        Main loop that runs both recieving network information and running
        """
        networkReciever.isRunning = False
        
        while True:
            #execute
            if isRunning:
                networkReciever.callback_run_model()
                
            #since select is 0 shouldnt ever block
            for selectable, _ in networkReciever.selector.select(0):
                isRunning = False
                networkReciever.callback_stop_model()
                
                call = selectable.data
                sock = selectable.fileobj
                
                call(sock)
        
    def recieve(sock: socket):
        """
        This function takes the socket reads the next input and determines the next command to run, also sends OK
        """
        
        try:
            data = sock.recv(BUFFER_SIZE).decode(ENCODING)
            sock.send("OK".encode(ENCODING))
            
            echo(f"Recived {data} command")
            
            func = networkReciever.command_map[data]
            func()
            
        except KeyError as err:
            echo(f"Warning: recieved malformed command from server: {data}")
        except OSError as err:
            echo(f"FAILED TO COMMUNICATE WITH SERVER")
            raise SystemExit(1)
        
    def setValues(sock: socket):
        None
        #TODO: MAX cause I dont understand how to interact with his model