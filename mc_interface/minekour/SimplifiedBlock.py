from enum import Enum

class SimplifiedBlock(Enum):
    AIR = 1 #this will include stuff like grass
    SOLID = 2 #note that stairs are modeled as solids
    WATER = 3
    LAVA = 4
    CLIMBABLE = 5
    DOOR = 6 #both doors and trapdoors are these as the top and bottom halves will be a thing
    FENCE = 7 #for stuff thats extra big like fences
    SMALL_SOLID = 8 #for stuff thats really small like pots
    POLE = 9 #for things like end rods and chains
    THIN = 10 #for stuff like glass panes
    SLAB = 11
    SLIME = 12
    MISC = 13 #all other unmapped blocks