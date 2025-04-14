from dataclasses import dataclass
from enum import Enum

class MoveType(Enum):
    """
    Used to define player movement speed state
    """
    SNEAKING = 1
    NORMAL = 2
    SPRINTING = 3
    
@dataclass
class Motion:
    """
    DataClass that stores all relevant information involving the motion of 
    """
    yaw: float = 0
    
    pitch: float = 0 
    #note that this has a valid range of -90, 90 but minescript will handle out of range values by truncating down to the nearest value
    
    movement_speed:MoveType = MoveType.NORMAL
    jumping: bool = False
    
    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False