from typing import List, Tuple
import datetime as dt

import numpy as np
from skyfield.api import load
from skyfield.iokit import parse_tle_file
from astropy.coordinates import SphericalRepresentation
import astropy.units as u
import scipy
from pyray import *
from raylib import *
from skyfield.toposlib import wgs84

import prn_from_name
from conversions import *
from simulator import Simulator
from gnss_sensor import GnssSensor
from solver import Solver
from constants import GPS_L1_FREQUENCY

PIXELS_TO_METERS = 1/10
METERS_TO_PIXELS = 1/PIXELS_TO_METERS
MOVEMENT_SPEED_METERS_PER_SECOND = 5.0
GNSS_MESSAGE_FREQUENCY = 5 # Hz
CUTOFF_ELEVATION = np.deg2rad(10)

SATELLITE_CLOCK_BIAS = np.array([
    5 * 1e-6,
    -10 * 1e-6,
    7 * 1e-6,
    10 * 1e-6,
    60 * 1e-6,
    -100 * 1e-6,
], dtype=np.float64)

SATELLITE_NUMBER = SATELLITE_CLOCK_BIAS.shape[0]

SATELLITE_ALPHAS = np.array([
    [0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6],
    [0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6],
    [0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6],
    [0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6],
    [0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6],
    [0.6519 * 1e-8, 0.1490 * 1e-7, -0.5960 * 1e-7, -0.1192 * 1e-6],
])

SATELLITE_BETAS = np.array([
    [0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6],
    [0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6],
    [0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6],
    [0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6],
    [0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6],
    [0.7782 * 1e5, 0.3277 * 1e5, -0.6554 * 1e5, -0.1966 * 1e6],
])

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

def main():
    width, height = 800, 450

    prn_visible = [27, 31, 29, 28, 25, 18, 32, 23, 10]

    timescale = load.timescale()

    with load.open('resources/gps.tle') as file:
        satellite_orbits = list(parse_tle_file(file, timescale))

    visible_satellite_orbits = [satellite for satellite in satellite_orbits if prn_from_name.get_prn(satellite.name) in prn_visible]
    cut_satellite_orbits = visible_satellite_orbits[:SATELLITE_NUMBER]
    start_time = timescale.utc(2025, 4, 19, 9, 0, 0)
    start_receiver_position = wgs84.latlon(0.0, 0.0).at(start_time).xyz.m

    print(f"Loaded {len(satellite_orbits)} satellites, {len(visible_satellite_orbits)} visible, cut to {SATELLITE_NUMBER}")
    print([satellite.name for satellite in cut_satellite_orbits])
    print(f"Sim start time: {start_time.utc_datetime()}")
    print(f"Satellite positions: {[list(np.round(np.rad2deg(ecef2llh(satellite.at(start_time).xyz.m)))) for satellite in cut_satellite_orbits]}")
    print(f"Satellite velocities: {[satellite.at(start_time).velocity.m_per_s for satellite in cut_satellite_orbits]}")
    print(f"Satelite clock biases: {SATELLITE_CLOCK_BIAS}")

    rng = np.random.default_rng()
    simulator = Simulator(rng, SATELLITE_NUMBER, SATELLITE_CLOCK_BIAS, GNSS_SIGNAL_FREQUENCY, SATELLITE_ALPHAS,
                          SATELLITE_BETAS, NOISE_CORRECTION_LEVEL, NOISE_FIX_LOSS_LEVEL, NOISE_EFFECT_RATE,
                          SATELLITE_NOISE_STD, TROPOSPHERIC_CUTOFF_ANGLE)
    solver = Solver(SATELLITE_NUMBER, SATELLITE_CLOCK_BIAS, GNSS_SIGNAL_FREQUENCY)
    sensor = GnssSensor(simulator, solver, CUTOFF_ELEVATION)

    time_utc = start_time
    player_positions: List[array3d] = [start_receiver_position]
    player_velocities: List[array3d] = [np.array([0, 0, 0], dtype=np.float64)]
    gnss_positions: List[array3d] = []
    gnss_velocities: List[array3d] = []
    receiver_clock_bias = RECEIVER_CLOCK_BIAS
    time_since_gnss = np.inf

    init_window(width, height, "Hello")
    while not window_should_close():
        delta = get_frame_time()

        time_utc += dt.timedelta(seconds=delta)
        time_since_gnss += delta

        satellite_positions = np.array([satellite.at(time_utc).xyz.m for satellite in cut_satellite_orbits], dtype=np.float64)
        satellite_velocities = np.array([satellite.at(time_utc).velocity.m_per_s for satellite in cut_satellite_orbits], dtype=np.float64)

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

        player_delta = np.array([player_delta_x, player_delta_y, 0], dtype=np.float64)
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
                                                               RECEIVER_CLOCK_WALK, time_utc)
            gnss_positions.append(gnss_position)
            gnss_velocities.append(gnss_velocity)
            time_since_gnss = 0

        player_position_px = (player_position - start_receiver_position) * METERS_TO_PIXELS
        gnss_velocity_px = gnss_velocities[-1] / MOVEMENT_SPEED_METERS_PER_SECOND * METERS_TO_PIXELS

        class _draw:
            begin_drawing()
            clear_background(WHITE)
            draw_text(f"{time_utc.utc_datetime()}", 10, 10, 14, BLACK)

            draw_circle_v(toVector2(player_position_px), 5, RED)

            for gnss_position in gnss_positions:
                draw_circle_v(toVector2((gnss_position - start_receiver_position) * METERS_TO_PIXELS), 2, GREEN)

            draw_line_v(toVector2(player_position_px), toVector2(player_position_px + gnss_velocity_px), BLUE)


            draw_rectangle(width - 200, 0, width, 200, WHITE)

            x = player_position
            x = x / np.array([10_000, 10_000, 1])
            x = x + np.array([width - 100, 100, 0])
            draw_circle_v(toVector2(x), 2, RED)

            for i, satellite_position in enumerate(satellite_positions):
                x = satellite_position
                x = x / np.array([1e6, 1e6, 1e6])
                x = x + np.array([width - 100, 100, 0])
                draw_circle_v(toVector2(x), 2, GREEN)
                draw_text(f"{i}", np.int64(x[0]).item(), np.int64(x[1]).item(), 1, BLACK)
            draw_rectangle_lines(width - 200, 0, width, 200, BLACK)

            end_drawing()
    close_window()

if __name__ == '__main__':
    main()