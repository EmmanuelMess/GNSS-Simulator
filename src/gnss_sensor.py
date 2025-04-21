from typing import Tuple

import numpy as np

from antenna_simulator import AntennaSimulator
from solver import Solver
from conversions import ecef2aer, array3d


class GnssSensor:
    def __init__(self, simulator: AntennaSimulator, solver: Solver, cutoff_elevation_rad: np.float64):
        self.simulator = simulator
        self.solver = solver
        self.cutoff_elevation_rad = cutoff_elevation_rad


    def _filter_by_elevation(self, player_position_ecef, satellite_positions_ecef):
        """
        Filter satellites that are below the horizon
        """
        satellite_positions_aer = np.array([ecef2aer(player_position_ecef, satellite_position) for satellite_position in satellite_positions_ecef])

        return satellite_positions_ecef[satellite_positions_aer[:,1] > self.cutoff_elevation_rad, :]


    def update(self, satellite_positions_ecef, satellite_velocities_ecef, player_positions, player_velocities,
               reciever_clock_bias, reciever_clock_drift, time_utc) -> Tuple[array3d, array3d, np.float64, np.float64]:
        print("===")
        print("UTC time")
        print(time_utc.utc_datetime()) # TODO

        print("Satelite positions")
        print(satellite_positions_ecef)

        print("Satelite velocities")
        print(satellite_velocities_ecef)

        # TODO check if satellites are over the horizon
        player_position = player_positions[-1]

        visible_satellite_positions_ecef = self._filter_by_elevation(player_position, satellite_positions_ecef)

        print(f"{visible_satellite_positions_ecef.shape[0]}/{satellite_positions_ecef.shape[0]} satellites over the horizon")

        pseudoranges = self.simulator.get_pseudoranges(visible_satellite_positions_ecef, player_position,
                                                       reciever_clock_bias, 0)

        # Computation of the satellite orbit, from ephimeris
        # Assume satellite position is known because ephimeris is transmitted during the first fix
        # From https://gssc.esa.int/navipedia/index.php/Coordinates_Computation_from_Almanac_Data
        position_aproximation, clock_bias_approximation, gnss_position_error = (
            self.solver.solve_position_scipy(visible_satellite_positions_ecef, pseudoranges, 1e-9))

        Q, gdop, hdop, vdop, pdop = self.simulator.get_dilution_of_presition(visible_satellite_positions_ecef, player_position)

        print(f"- Position")
        print(f"Cost {gnss_position_error}, estimated position {position_aproximation}m, estimated reciever clock bias {clock_bias_approximation}s")
        print(f"GDOP {gdop} HDOP {hdop} VDOP {vdop} PDOP {pdop}")
        print(f"Error {np.linalg.norm(position_aproximation-player_position)}m, real position {player_position}m, real reciever clock bias {reciever_clock_bias}s")

        player_velocity = player_velocities[-1]
        direct_doppler = self.simulator.get_doppler(visible_satellite_positions_ecef, satellite_velocities_ecef,
                                                    player_position, player_velocity, reciever_clock_drift)

        # Reciever estimation
        # From https://satellite-navigation.springeropen.com/counter/pdf/10.1186/s43020-023-00098-2.pdf
        # Also see Navigation from Low Earth Orbit – Part 2: Models, Implementation, and Performance section 2.2
        velocity_approximation, clock_drift_approximation, gnss_velocity_error = (
            self.solver.solve_velocity(visible_satellite_positions_ecef, satellite_velocities_ecef, direct_doppler,
                                       position_aproximation, 1e-9))

        print(f"- Velocity")
        print(f"Cost {gnss_velocity_error}, estimated velocity {velocity_approximation}m/s, linear velocity {np.linalg.norm(velocity_approximation)}m/s")
        print(f"Estimated receiver clock drift {clock_drift_approximation}s")
        print(f"Error {np.linalg.norm(velocity_approximation-player_velocity)}m/s, real velocity {player_velocity}m/s, linear_velocity {np.linalg.norm(player_velocity)}m/s")

        return position_aproximation, velocity_approximation, clock_bias_approximation, clock_drift_approximation
