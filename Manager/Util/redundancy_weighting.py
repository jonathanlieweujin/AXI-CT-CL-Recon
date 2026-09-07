from Manager.Constants.laminography_method import LaminographyMethodConstants

TILT_X_NOMINAL = 90.0
TILT_X_TOLERANCE = 0.5

def to_apply_redundancy_weighting(
    offset_u: float,
    tilt_x: float,
    laminography_method: LaminographyMethodConstants,
) -> bool:
    """
    Whether Wang redundancy weights will be applied.

    Only valid for standard upright offset CT detector geometry
    (tilt_x ~ 90 deg) built by the default inclined vector computation, so any
    other laminography method is rejected outright.
    """
    if laminography_method != LaminographyMethodConstants.INCLINED:
        return False

    return offset_u != 0 and abs(tilt_x - TILT_X_NOMINAL) <= TILT_X_TOLERANCE
