from enum import Enum

class LaminographyMethodConstants(str, Enum):
    """Laminography geometry identifiers shared by QualityFlow and PerformanceFlow."""

    INCLINED = "INCLINED"
    COPLANAR = "COPLANAR"

    def __str__(self) -> str:
        return self.value