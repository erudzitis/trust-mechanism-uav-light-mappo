from enum import Enum, auto

class FormationType(Enum):
    LINE = 1
    V = 2


class CollisionSeverity(Enum):
    """Enum representing the severity of a collision."""
    NONE = auto()          # No collision
    NEAR_MISS = auto()     # Close enough to be dangerous
    COLLISION = auto()     # Actual collision occurred
    SEVERE = auto()        # Severe collision (high velocity or prolonged)