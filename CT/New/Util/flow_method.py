from enum import Enum


class VxFlowMethod(str, Enum):
    """Reconstruction flow selector: QualityFlow (TIGRE) vs PerformanceFlow (ASTRA)."""

    QUALITY = "QUALITY"
    PERFORMANCE = "PERFORMANCE"

    def __str__(self) -> str:
        return self.value
