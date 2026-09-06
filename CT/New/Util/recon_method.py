from enum import Enum

class ReconMethodConstants(str, Enum):
    """Reconstruction algorithm identifiers."""

    FDK = "FDK"

    # Gradient / ART family
    SART = "SART"
    OS_SART = "OS_SART"
    OSSART = "OSSART"
    SIRT = "SIRT"
    ASD_POCS = "ASD_POCS"
    OS_ASD_POCS = "OS_ASD_POCS"
    B_ASD_POCS_BETA = "B_ASD_POCS_BETA"
    AW_ASD_POCS = "AW_ASD_POCS"
    AWASD_POCS = "AWASD_POCS"
    OS_AW_ASD_POCS = "OS_AW_ASD_POCS"
    OS_AWASD_POCS = "OS_AWASD_POCS"
    PCSD = "PCSD"
    OS_PCSD = "OS_PCSD"
    AW_PCSD = "AW_PCSD"
    AWPCSD = "AWPCSD"
    OS_AW_PCSD = "OS_AW_PCSD"
    OS_AWPCSD = "OS_AWPCSD"

    # Krylov subspace family
    CGLS = "CGLS"
    LSQR = "LSQR"
    HYBRID_LSQR = "HYBRID_LSQR"
    H_LSQR = "H_LSQR"
    LSMR = "LSMR"
    IRN_TV_CGLS = "IRN_TV_CGLS"
    HYBRID_FLSQR_TV = "HYBRID_FLSQR_TV"
    H_FLSQR_TV = "H_FLSQR_TV"
    AB_GMRES = "AB_GMRES"
    BA_GMRES = "BA_GMRES"
    AB_BA_GMRES = "AB_BA_GMRES"

    # Statistical / variational
    MLEM = "MLEM"
    FISTA = "FISTA"
    ISTA = "ISTA"
    SART_TV = "SART_TV"
    OSSART_TV = "OSSART_TV"
    OS_SART_TV = "OS_SART_TV"

    def __str__(self) -> str:
        return self.value
