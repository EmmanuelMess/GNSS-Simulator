import numpy as np

from numpy_types import array3d
from conversions import ecef2aer


def is_satellite_overhead(receiver_position_ecef: array3d, satellite_positions_ecef: array3d,
                          elevation_cutoff_rad: np.float64):
    satellite_position_aer = ecef2aer(receiver_position_ecef, satellite_positions_ecef)

    is_overhead = satellite_position_aer[1] > elevation_cutoff_rad

    return is_overhead