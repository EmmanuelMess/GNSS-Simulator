import datetime
import os
from pathlib import Path
from typing import List
import datetime as dt

from astropy.time import Time
from pyray import *
from raylib import *

from src.conversions import *
from src.antenna_simulator import AntennaSimulator
from src.gnss_sensor import GnssSensor
from src.solver import Solver
from src.constants import GPS_L1_FREQUENCY
from src.rinex_generator import RinexGenerator
from src.is_overhead import is_satellite_overhead
from src.skyplot import get_skyplot
from src import gps_orbital_parameters
from src.gps_satellite import GpsSatellite

SIMULATION_DUMG = {
    "start_time": "2025-01-01T09:00:00.000",
    "receiver_position_start": llh2ecef(np.array([np.deg2rad(-66.665169), np.deg2rad(140.002200), -3.38], dtype=np.float64)),
    "satellite_filenames": ["dumg/03.orbit", "dumg/04.orbit", "dumg/06.orbit", "dumg/09.orbit", "dumg/11.orbit",
                            "dumg/26.orbit", "dumg/28.orbit", "dumg/31.orbit"],
    "ionospheric_model": (
        np.array([0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6], dtype=np.float64),
        np.array([0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6], dtype=np.float64)
    ),
    "jammer_noise": np.float64(0.0) # dB
}
SIMULATION_0LAT0LON = {
    "start_time": "2025-05-01T09:00:00.000",
    "receiver_position_start": llh2ecef(np.array([np.deg2rad(0.0), np.deg2rad(0.0), 0.0], dtype=np.float64)),
    "satellite_filenames": ["invented-0lat-0lon/01.orbit", "invented-0lat-0lon/02.orbit", "invented-0lat-0lon/03.orbit",
                       "invented-0lat-0lon/04.orbit", "invented-0lat-0lon/05.orbit", "invented-0lat-0lon/06.orbit"],
    "ionospheric_model": (
        np.array([0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6], dtype=np.float64),
        np.array([0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6], dtype=np.float64)
    ),
    "jammer_noise": np.float64(0.0) # dB
}

PIXELS_TO_METERS = 1/10
METERS_TO_PIXELS = 1/PIXELS_TO_METERS
MOVEMENT_SPEED_METERS_PER_SECOND = 5.0
SKYPLOT_SIZE = 200

GNSS_MESSAGE_FREQUENCY = 5 # Hz
CUTOFF_ELEVATION = np.deg2rad(10)

SATELLITE_NUMBER = 6

RECEIVER_CLOCK_BIAS = np.float64(1 * 1e-3)
# Walk of 0.1us per second
# From https://www.e-education.psu.edu/geog862/node/1716
RECEIVER_CLOCK_WALK = np.float64(1 * 1e-6)

SATELLITE_NOISE_STD = np.float64(0.0)

# This is the angle at which satellites start not being affected by troposferic effects
TROPOSPHERIC_CUTOFF_ANGLE = np.deg2rad(5)

# This noise level does not affect the reciever, because it can correct for the noise
NOISE_CORRECTION_LEVEL = np.float64(7) # dB
# This noise level causes the receiver to lose the fix
NOISE_FIX_LOSS_LEVEL = np.float64(40) # dB

# Effect of noise in the range (NOISE_CORRECTION_LEVEL, NOISE_FIX_LOSS_LEVEL), in terms of the mean meter error of the
# pseudorange
NOISE_EFFECT_RATE = np.float64(5) / (NOISE_FIX_LOSS_LEVEL - NOISE_CORRECTION_LEVEL) # m / dB

GNSS_SIGNAL_FREQUENCY = GPS_L1_FREQUENCY


def create_rinex_generator(start_position_ecef, satellite_orbits: List[GpsSatellite], utc_start: Time,
                           gps_start: Time, satellite_alphas: np.ndarray[4, np.float64],
                           satellite_betas: np.ndarray[4, np.float64]) -> RinexGenerator:
    folder_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    folder_path = os.path.join("output", folder_name)
    Path(folder_path).mkdir(parents=False, exist_ok=True)

    rinex_generator = RinexGenerator(folder_path, start_position_ecef, utc_start, gps_start, satellite_alphas,
                                     satellite_betas)

    rinex_generator.add_satellites(satellite_orbits)

    return rinex_generator


def main():
    # Implementation specific constants
    width, height = 800, 450
    rng = np.random.default_rng()
    simulation_data = SIMULATION_0LAT0LON
    start_time = Time(simulation_data["start_time"], format="isot", scale="utc")
    start_receiver_position = simulation_data["receiver_position_start"]
    e_axis, n_axis, u_axis = pos2enu_base(ecef2llh(start_receiver_position))
    s_axis = -n_axis # HACK because y axis is inverted
    gps_start_time = time2gps(start_time)
    ionospheric_alphas, ionospheric_betas = simulation_data["ionospheric_model"]
    satellite_orbits: List[GpsSatellite] = []

    # Get satellites, filter by availability
    for filename in simulation_data["satellite_filenames"]:
        with open(os.path.join("resources", filename), 'r') as file:
            lines = file.read()
            parameters = gps_orbital_parameters.from_rinex(lines)
            satellite_orbits.append(GpsSatellite(parameters))

    visible_satellite_orbits = [
        satellite for satellite in satellite_orbits if is_satellite_overhead(start_receiver_position, satellite.position_velocity(gps_start_time)[0], CUTOFF_ELEVATION)
    ]
    cut_satellite_orbits = visible_satellite_orbits[:SATELLITE_NUMBER]
    satellite_prns: List[int] = [satellite.parameters().prn_number for satellite in cut_satellite_orbits]
    # WARNING: the model used keeps the satellite clock bias constant
    satellite_clock_bias = np.array([satellite.orbit_parameters.sv_clock_bias for satellite in cut_satellite_orbits], dtype=np.float64)

    print(f"Loaded {len(satellite_orbits)} satellites, {len(visible_satellite_orbits)} visible, cut to {SATELLITE_NUMBER}")
    print(satellite_prns)
    print(f"Sim start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Reveiver start position {start_receiver_position}, enu {e_axis} {n_axis} {u_axis}")
    print(f"Satellite epochs: {[satellite.parameters().time_of_ephemeris.strftime('%Y-%m-%d %H:%M:%S') for satellite in cut_satellite_orbits]}")
    print(f"Satellite positions: {[np.round(np.rad2deg(ecef2llh(satellite.position_velocity(gps_start_time)[0]))) for satellite in cut_satellite_orbits]}")
    print(f"Satellite velocities: {[satellite.position_velocity(gps_start_time)[1] for satellite in cut_satellite_orbits]}")
    print(f"Satelite clock biases: {satellite_clock_bias}")
    print(f"Seconds of week to first epoch {time_gps2seconds_of_week(gps_start_time.gps)}")

    rinex_generator = create_rinex_generator(start_receiver_position, cut_satellite_orbits, start_time, gps_start_time,
                                             ionospheric_alphas, ionospheric_betas)
    # Position simulation components
    simulator = AntennaSimulator(rng, SATELLITE_NUMBER, satellite_clock_bias, GNSS_SIGNAL_FREQUENCY, ionospheric_alphas,
                                 ionospheric_betas, simulation_data["jammer_noise"], NOISE_CORRECTION_LEVEL,
                                 NOISE_FIX_LOSS_LEVEL, NOISE_EFFECT_RATE, SATELLITE_NOISE_STD,
                                 TROPOSPHERIC_CUTOFF_ANGLE)
    solver = Solver(SATELLITE_NUMBER, satellite_clock_bias, GNSS_SIGNAL_FREQUENCY)
    sensor = GnssSensor(simulator, solver, rinex_generator, np.array(satellite_prns, dtype=np.int64), CUTOFF_ELEVATION)

    # State
    time_utc = start_time
    player_positions: List[array3d] = [start_receiver_position]
    player_velocities: List[array3d] = [np.array([0, 0, 0], dtype=np.float64)]
    gnss_positions: List[array3d] = []
    gnss_velocities: List[array3d] = []
    receiver_clock_bias = RECEIVER_CLOCK_BIAS
    time_since_gnss = np.inf

    init_window(width, height, "GNSS Simulator Prototype")
    while not window_should_close():
        delta = get_frame_time()

        time_utc += dt.timedelta(seconds=delta)
        time_gps = time2gps(time_utc)

        time_since_gnss += delta

        satellite_positions_velocities = np.array([satellite.position_velocity(time_gps) for satellite in cut_satellite_orbits], dtype=np.float64)
        satellite_positions = satellite_positions_velocities[:, 0, :]
        satellite_velocities = satellite_positions_velocities[:, 1, :]

        if is_key_down(KEY_LEFT) or is_key_down(KEY_A):
            player_delta_x = -1
        elif is_key_down(KEY_RIGHT) or is_key_down(KEY_D):
            player_delta_x = 1
        else:
            player_delta_x = 0
        if is_key_down(KEY_UP) or is_key_down(KEY_W):
            player_delta_y = -1
        elif is_key_down(KEY_DOWN) or is_key_down(KEY_S):
            player_delta_y = 1
        else:
            player_delta_y = 0

        player_delta = player_delta_x * e_axis + player_delta_y * s_axis
        player_delta = player_delta if np.linalg.norm(player_delta) == 0 else player_delta / np.linalg.norm(player_delta)
        player_position = player_positions[-1] + MOVEMENT_SPEED_METERS_PER_SECOND * delta * player_delta
        player_positions.append(player_position)

        player_velocity = MOVEMENT_SPEED_METERS_PER_SECOND * player_delta
        player_velocities.append(player_velocity)

        if time_since_gnss > 1/GNSS_MESSAGE_FREQUENCY:
            # See GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
            if time_since_gnss < np.inf:
                receiver_clock_bias += RECEIVER_CLOCK_WALK * rng.normal() * time_since_gnss

            gnss_position, gnss_velocity, _, _ = sensor.update(satellite_positions, satellite_velocities,
                                                               player_positions, player_velocities, receiver_clock_bias,
                                                               RECEIVER_CLOCK_WALK, time_gps, time_utc)
            gnss_positions.append(gnss_position)
            gnss_velocities.append(gnss_velocity)
            time_since_gnss = 0

        player_position_px = (player_position - start_receiver_position) * METERS_TO_PIXELS
        gnss_velocity_px = gnss_velocities[-1] / MOVEMENT_SPEED_METERS_PER_SECOND * METERS_TO_PIXELS

        skyplot = get_skyplot(player_position, satellite_positions, satellite_prns, SKYPLOT_SIZE, SKYPLOT_SIZE, np.deg2rad(10))

        class _draw:
            begin_drawing()
            clear_background(WHITE)
            draw_text(f"{time_utc.strftime('%Y-%m-%d %H:%M:%S')}", 10, 10, 14, BLACK)

            draw_circle_v(toVector2(player_position_px, e_axis, s_axis), 5, RED)

            for gnss_position in gnss_positions:
                draw_circle_v(toVector2((gnss_position - start_receiver_position) * METERS_TO_PIXELS, e_axis, s_axis), 2, GREEN)

            draw_line_v(toVector2(player_position_px, e_axis, s_axis), toVector2(player_position_px + gnss_velocity_px, e_axis, s_axis), BLUE)

            draw_rectangle(width - SKYPLOT_SIZE, 0, width, SKYPLOT_SIZE, WHITE)
            draw_line(width - SKYPLOT_SIZE, SKYPLOT_SIZE//2, width,  SKYPLOT_SIZE//2, GRAY)
            draw_line(width - SKYPLOT_SIZE//2, 0, width - SKYPLOT_SIZE//2, SKYPLOT_SIZE, GRAY)
            draw_circle_lines(width - SKYPLOT_SIZE//2, SKYPLOT_SIZE//2, skyplot.sixty_deg_line, GRAY)
            draw_circle_lines(width - SKYPLOT_SIZE//2, SKYPLOT_SIZE//2, skyplot.thirty_deg_line, GRAY)
            draw_circle_lines(width - SKYPLOT_SIZE//2, SKYPLOT_SIZE//2, skyplot.zero_deg_line, GRAY)

            for prn, satellite_position_px in zip(skyplot.prns, skyplot.satellite_positions):
                x, y = satellite_position_px
                draw_circle(x + (width - SKYPLOT_SIZE), y, 2, GREEN)
                draw_text(f"{prn}", x + (width - SKYPLOT_SIZE), y, 1, BLACK)

            draw_rectangle_lines(width - SKYPLOT_SIZE, 0, width, SKYPLOT_SIZE, BLACK)

            end_drawing()
    close_window()

if __name__ == '__main__':
    main()