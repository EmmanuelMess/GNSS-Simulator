from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from conversions import ecef2aer


@dataclass
class Skyplot:
    prns: List[int]
    satellite_positions: List[Tuple[int, int]]
    sixty_deg_line: int
    thirty_deg_line: int
    zero_deg_line: int


def get_skyplot(receiver_position_ecef, satellite_positions_ecef, satellite_prns, width, height,
                elevation_cutoff_rad: np.float64) -> Skyplot:
    max_altitude_rad = np.deg2rad(90)

    def convert_altitude(elevation):
        return -(elevation - max_altitude_rad) / max_altitude_rad * (width//2) * 0.90

    skyplot_prns = []
    skyplot_positions = []

    for prn, satellite_position_ecef in zip(satellite_prns, satellite_positions_ecef):
        satellite_position_aer = ecef2aer(receiver_position_ecef, satellite_position_ecef)
        azimuth, elevation = satellite_position_aer[0], satellite_position_aer[1]
        if elevation < elevation_cutoff_rad:
            continue

        skyplot_length = convert_altitude(elevation)
        skyplot_x = np.cos(azimuth - max_altitude_rad) * skyplot_length + width//2
        skyplot_y = np.sin(azimuth - max_altitude_rad) * skyplot_length + height//2

        skyplot_prns.append(prn)
        skyplot_positions.append((np.int64(skyplot_x).item(), np.int64(skyplot_y).item()))

    return Skyplot(
        prns=skyplot_prns, satellite_positions=skyplot_positions, sixty_deg_line=convert_altitude(np.deg2rad(60)),
        thirty_deg_line=convert_altitude(np.deg2rad(30)), zero_deg_line=convert_altitude(np.deg2rad(0))
    )