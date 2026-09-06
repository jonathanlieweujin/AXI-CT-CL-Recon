"""to_apply_redundancy_weighting mirrors the C# ToApplyRedundancyWeighting."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Util.laminography_method import LaminographyMethodConstants
from Util.redundancy_weighting import to_apply_redundancy_weighting

INCLINED = LaminographyMethodConstants.INCLINED
COPLANAR = LaminographyMethodConstants.COPLANAR


@pytest.mark.parametrize("offset_u,tilt_x,method,expected", [
    # upright offset CT: the only case Wang weights are valid for
    (-38.52, 90.0, INCLINED, True),
    (2.5, 90.0, INCLINED, True),
    (2.5, 89.5, INCLINED, True),      # on the tolerance edge
    (2.5, 90.5, INCLINED, True),
    # outside the 0.5 deg tolerance
    (2.5, 89.4, INCLINED, False),
    (2.5, 90.6, INCLINED, False),
    (2.5, 55.0, INCLINED, False),
    # no offset means no redundancy to weight
    (0.0, 90.0, INCLINED, False),
    # coplanar is rejected outright, whatever the tilt/offset
    (2.5, 90.0, COPLANAR, False),
    (0.0, 55.0, COPLANAR, False),
])
def test_predicate(offset_u, tilt_x, method, expected):
    assert to_apply_redundancy_weighting(offset_u, tilt_x, method) is expected
