import subprocess

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
        
#start_agents([1,2,3], "C:/Users/ericy/AppData/Local/Programs/PrismLauncher/prismlauncher.exe")