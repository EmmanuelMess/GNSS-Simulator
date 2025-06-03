from typing import Tuple, List

import numpy as np
from hifitime.hifitime import Epoch

from src.antenna_simulator import AntennaSimulator
from src.solver import Solver
from src.conversions import array3d, time_gps2seconds_of_week
from src.is_overhead import is_satellite_overhead
from src.rinex_generator import RinexGenerator


class GnssSensor:
    def __init__(self, simulator: AntennaSimulator, solver: Solver, rinex_generator: RinexGenerator,
                 prn_satellites: np.ndarray[(-1,), np.int64], cutoff_elevation_rad: np.float64):
        self.simulator = simulator
        self.solver = solver
        self.rinex_generator = rinex_generator
        self.prn_satellites = prn_satellites
        self.cutoff_elevation_rad = cutoff_elevation_rad

    def update(self, satellite_positions_ecef, satellite_velocities_ecef, player_positions, player_velocities,
               reciever_clock_bias, reciever_clock_drift, time_gps: Epoch, time_utc: Epoch)\
            -> Tuple[array3d, array3d, np.float64, np.float64]:
        print("===")
        print(f"GPS time: {time_gps.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"UTC time: {time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"Satelite positions: {satellite_positions_ecef}")
        print(f"Satelite velocities: {satellite_velocities_ecef}")

        player_position = player_positions[-1]
        player_velocity = player_velocities[-1]

        time_of_week_gps_seconds = time_gps2seconds_of_week(np.float64(time_gps.to_gpst_seconds()))

        visible_index = np.array([
            is_satellite_overhead(player_position, satellite_position, self.cutoff_elevation_rad)
            for satellite_position in satellite_positions_ecef
        ])

        visible_satellite_positions_ecef = satellite_positions_ecef[visible_index, :]
        visible_satellites_prn = self.prn_satellites[visible_index]

        print(f"{visible_satellite_positions_ecef.shape[0]}/{satellite_positions_ecef.shape[0]} satellites over the horizon")

        pseudoranges = self.simulator.get_pseudoranges(visible_satellite_positions_ecef, player_position,
                                                       reciever_clock_bias, time_of_week_gps_seconds)

        # Computation of the satellite orbit, from ephimeris
        # Assume satellite position is known because ephimeris is transmitted during the first fix
        # From https://gssc.esa.int/navipedia/index.php/Coordinates_Computation_from_Almanac_Data
        position_aproximation, clock_bias_approximation, gnss_position_error = (
            self.solver.solve_position(visible_satellite_positions_ecef, pseudoranges, 1e-4))

        Q, gdop, hdop, vdop, pdop = self.simulator.get_dilution_of_presition(visible_satellite_positions_ecef, player_position)

        print(f"- Position")
        print(f"Cost {gnss_position_error}, estimated position {position_aproximation}m, estimated reciever clock bias {clock_bias_approximation}s")
        print(f"GDOP {gdop} HDOP {hdop} VDOP {vdop} PDOP {pdop}")
        print(f"Error {np.linalg.norm(position_aproximation-player_position)}m, real position {player_position}m, real reciever clock bias {reciever_clock_bias}s")

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

        # TODO the time_gps should be biased by the receiver clock
        self.rinex_generator.add_position(time_gps, list(visible_satellites_prn), list(pseudoranges), list(direct_doppler))

        return position_aproximation, velocity_approximation, clock_bias_approximation, clock_drift_approximation
