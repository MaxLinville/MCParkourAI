import subprocess
import socket
import selectors
from threading import Lock

"""
implementation of running multiple agents asynchronously
"""

def start_agents(agents: list[int], location: str):
    """
    starts the agent #'s listed in agents
    location is the path to the prismlauncher exe
    
    This function expects the prism instance folder to have the following format:
        MinescriptClient[n]
    
    This function also expects for the offline acounts to be set up in the following format:
        Minescript[n]
    
    """
    for n in agents:
        subprocess.run([location, "--launch", f"MinescriptClient{n}", "--profile", f"Minescript{n}"])

# start_agents([_+1 for _ in range(32)], "/mnt/c/Users/Max Linville/AppData/Local/Programs/PrismLauncher/prismlauncher.exe")

class networkCommander:
    """
    This class manages incoming connections and provides methods to send commands to clients
    """
    DEFAULT_CONTROL_PORT = 25567
    DEFAULT_DEAD_PORT = 25568
    CONTROL_SOCKET_FAMILY = socket.AF_INET #IPV4
    CONTROL_SOCKET_TYPE = socket.SOCK_STREAM #TCP
    BUFFER_SIZE = 2**12
    ENCODING = "UTF-8"
    
    
    def __init__(self, clients: int, ip: str="", port: int = DEFAULT_CONTROL_PORT, dead_port: int = DEFAULT_DEAD_PORT):
        self.clients = list() #this is just a list of sockets
        self.num_clients = clients
        self._lock = Lock()
        
        # init socket
        try:
            
            #listener socket
            self.server_socket = socket.socket(self.CONTROL_SOCKET_FAMILY, self.CONTROL_SOCKET_TYPE)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            address = (ip, port)
            
            self.server_socket.bind(address)
            self.server_socket.listen()
            
            #do dead port now
            self.dead_socket = socket.socket(self.CONTROL_SOCKET_FAMILY, self.CONTROL_SOCKET_TYPE)
            self.dead_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            dead_address = (ip, dead_port)
            
            self.dead_socket.bind(dead_address)
            self.dead_socket.listen()
            
            self.dead_selector = selectors.DefaultSelector()
            self.num_dead = 0
            
            # add to selector for multiplex
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.server_socket, selectors.EVENT_READ, self.accept_client)
            self.selector.register(self.dead_socket, selectors.EVENT_READ, self.accept_client_dead)
            
        except OSError as err:
            print(f"Failed to open listener on {ip}:{port}")
            
    def wait_for_clients(self):
        """
        Waits for numClient number of clients to connect and adds them to the socket, clients are assigned numerical IDs based on join order
        This function will block until all clients connect
        """        
        while len(self.clients) + self.num_dead < self.num_clients * 2:
            for selectable, _ in self.selector.select():
                call = selectable.data
                call(selectable.fileobj)
        
    def accept_client(self, server_socket: socket):
        peer_socket, _ = server_socket.accept()
        self.clients.append(peer_socket)
        print(f"connected to client #{len(self.clients)}")
        
        # send ID to client
        peer_socket.send(str(len(self.clients)-1).encode(networkCommander.ENCODING))
        
    def accept_client_dead(self, server_socket: socket):
        peer_socket, _ = server_socket.accept()
        
        # get ID
        id = int(peer_socket.recv(networkCommander.BUFFER_SIZE).decode(networkCommander.ENCODING))
        self.dead_selector.register(peer_socket, selectors.EVENT_READ, id)
        print(f"Accepted client on dead socket for ID {id}")
        self.num_dead = self.num_dead + 1
        
    def get_client(self, n:int) -> socket:
        """
        Gets n client socket from the list, supports threading
        """
        sock = None
        
        with self._lock:
            sock = self.clients[n]
            
        return sock
        
    def reset(self, n: int):
        """
        Sends the reset command to the specified client #
        """
        client = self.get_client(n)
        client.send("RESET".encode(self.ENCODING))
        
        #wait for OK
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        if response != "OK":
            print(f"WARNING:client {n} responded with non-ok on RESET with response: {response}")
            
    def get(self, n: int):
        """
        sends the get commands and returns the score
        """
        client = self.get_client(n)
        client.send("GET".encode(self.ENCODING))
        
        #wait for OK
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        if response != "OK":
            print(f"WARNING:client {n} responded with non-ok on GET with response: {response}")
        
        #get score
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        print(f"Recieved {response} from client {n}")
        score = float(response)
        
        #respond OK
        client.send("OK".encode(self.ENCODING))
        
        return score
    
    def set(self, n: int, gene_bytestring: bytes):
        """
        Sets new parameters for the client
        """
        client = self.get_client(n)
        client.send("SET".encode(self.ENCODING))

        # Wait for OK
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        if response != "OK":
            print(f"WARNING: client {n} responded with non-OK on SET with response: {response}")
            return
        
        # Split gene data into smaller chunks (each under 1KB)
        MAX_CHUNK_SIZE = 1000
        genes = gene_bytestring.decode(self.ENCODING).split(';')
        
        # Send genes in batches
        batch = []
        batch_size = 0
        for gene in genes:
            if batch_size + len(gene) + 1 > MAX_CHUNK_SIZE:  # +1 for semicolon
                # Send current batch
                batch_str = ';'.join(batch)
                client.send(batch_str.encode(self.ENCODING))
                
                # Wait for OK
                response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
                if response != "OK":
                    print(f"WARNING: client {n} responded with non-OK on gene batch")
                    return
                    
                # Reset batch
                batch = [gene]
                batch_size = len(gene)
            else:
                batch.append(gene)
                batch_size += len(gene) + 1  # +1 for semicolon
        
        # Send any remaining genes
        if batch:
            batch_str = ';'.join(batch)
            client.send(batch_str.encode(self.ENCODING))
            
            # Wait for OK
            response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
            if response != "OK":
                print(f"WARNING: client {n} responded with non-OK on final gene batch")
                return
        
        # Send STOP command
        client.send("STOP".encode(self.ENCODING))
        
        # Wait for final OK
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        if response != "OK":
            print(f"WARNING: client {n} responded with non-OK on STOP with response: {response}")
        
    def start(self, n: int):
        """
        Sends the start command to the specified client #
        """
        client = self.get_client(n)
        client.send("START".encode(self.ENCODING))
        
        #wait for OK
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        if response != "OK":
            print(f"WARNING:client {n} responded with non-ok on START with response: {response}")
            
    def kill(self, n: int):
        """
        Sends the kill command to the specified client #
        """
        client = self.get_client(n)
        client.send("KILL".encode(self.ENCODING))
        
        #wait for OK
        response = client.recv(self.BUFFER_SIZE).decode(self.ENCODING)
        if response != "OK":
            print(f"WARNING:client {n} responded with non-ok on KILL with response: {response}")
            
    def getDead(self) -> list[int]:
        """
        gets a list of dead IDs
        """
        dead_ids = list()
        
        for selectable, _ in self.dead_selector.select(0):
            id = selectable.data
            socket = selectable.fileobj
            socket.recv(networkCommander.BUFFER_SIZE) #drop dont care
            dead_ids.append(id)
            
        return dead_ids