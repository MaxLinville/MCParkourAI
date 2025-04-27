import csv
import re
import math

from minescript import echo, getblocklist, player_position
from minescript import player, player_set_orientation, player_press_sprint, player_press_sneak, player_press_jump, player_press_forward, player_press_backward, player_press_left, player_press_right, execute, player_orientation

from .SimplifiedBlock import SimplifiedBlock
from .PlayerMotion import (Motion, MoveType)

class PlayerManager:
    """
    class for managing everything about a player
    Since only one player will be managed per script 
    this will be modeled as a singleton where
    no object needs to be created and everything will be called from a class context
    """

    #constant for defining how far agent can see
    DETECTION_RANGE = 5
    block_map = None #none initially, need to set this later
    
    def __generate_cube_coords(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # Find min and max for each axis to cover the cube
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        z_min, z_max = sorted([z1, z2])

        # Generate all integer coordinates within the cube
        cube_coords = [
            (x, y, z)
            for x in range(x_min, x_max + 1)
            for y in range(y_min, y_max + 1)
            for z in range(z_min, z_max + 1)
        ]

        return cube_coords

    #also GPT
    def __get_cube_bounds(center: tuple[int, int, int], distance: int=4) -> tuple[int,int]:
        x, y, z = center
        x = math.floor(x)
        y = math.floor(y)
        z = math.floor(z)
        point1 = ((x - distance), (y - distance), (z - distance))
        point2 = ((x + distance), (y + distance), (z + distance))
        return point1, point2

    def getBlocksAroundPlayer() -> list[SimplifiedBlock]:
        """
        Returns a list of blocks around the player in Simplified_Block format
        """
        center_point = player_position()
        point1, point2 = PlayerManager.__get_cube_bounds(center_point, PlayerManager.DETECTION_RANGE)
        list_of_points = PlayerManager.__generate_cube_coords(point1, point2)
        blocks = getblocklist(list_of_points)
        
        #convert this block list to the correct simplified block
        returned_blocks_striped = [re.match(r".*:([^\[]+)", x.upper()).group(1) for x in blocks]
        returned_blocks_enum = [PlayerManager.block_map[x] for x in returned_blocks_striped]

        # Ensure we return exactly (2*DETECTION_RANGE+1)^3 blocks
        expected_size = (2*PlayerManager.DETECTION_RANGE+1)**3
        current_size = len(returned_blocks_enum)
        
        if current_size < expected_size:
            echo(f"Warning: Only got {current_size} blocks, padding with AIR to {expected_size}")
            returned_blocks_enum.extend([SimplifiedBlock.AIR] * (expected_size - current_size))
        elif current_size > expected_size:
            echo(f"Warning: Got {current_size} blocks, truncating to {expected_size}")
            returned_blocks_enum = returned_blocks_enum[:expected_size]

        return returned_blocks_enum
    
    def getRotation() -> tuple[float, float]:
        """
        Returns yaw, pitch of the player
        """
        return player_orientation()
    
    def readBlockData(file_path: str):
        """
        Populates the block translation map from a csv file
        """
        #WARNING this function will not compensate for invalid case etc... will just fail
        PlayerManager.block_map = dict()
        
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                PlayerManager.block_map[row[0].upper()] = SimplifiedBlock[row[1].upper()]
    
    def movePlayer(move: Motion):
        """
        Uses a move dataclass to set the motion of the player
        """
        player_set_orientation(move.yaw, move.pitch)
        
        match move.movement_speed:
            case MoveType.SNEAKING:
                player_press_sprint(False)
                player_press_sneak(True)
            case MoveType.NORMAL:
                player_press_sprint(False)
                player_press_sneak(False)
            case MoveType.SPRINTING:
                player_press_sprint(True)
                player_press_sneak(False)
                
        player_press_jump(move.jumping)
        player_press_forward(move.forward)
        player_press_backward(move.backward)
        player_press_left(move.left)
        player_press_right(move.right)
        
    def getScore() -> float:
        """
        Gets the score of this player
        """
        string = player(nbt=True).nbt
        match = re.search(r"XpLevel:(\d*),", string)
        score = int(match.group(1))

        try:
            if score is None:
                return 0.0  # Default value if score can't be read
            return score  # Make sure to convert to string
        except Exception as e:
            echo(f"Error getting score: {e}")
            return 0.0  # Return default on error
        
    def resetPlayer():
        """
        Resets the player position and score
        """
        execute("/kill @p")
        
def init():
    """
    Initializes any values and classes needed for PlayerManager
    """
    PlayerManager.readBlockData("./minescript/blockmap_excel.csv")
