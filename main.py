from typing import List, Tuple

from astropy.coordinates import SphericalRepresentation, CartesianRepresentation
from astropy.constants import R_earth
import astropy.units as u
import numpy as np
import scipy
from pyray import *
from raylib import *

PIXELS_TO_METERS = 1/10
METERS_TO_PIXELS = 1/PIXELS_TO_METERS
MOVEMENT_SPEED_METERS_PER_SECOND = 5.0
GNSS_FREQUENCY = 5 # Hz

array3d = np.ndarray[(2,), np.float64]
tau = np.pi * 2

def getSatelliteAtAngle(lat, lon) -> array3d:
    MEO = np.float64(20_200_000) * u.m + R_earth
    spherical = SphericalRepresentation(lat=lat, lon=lon, distance=MEO)
    cartesian = spherical.to_cartesian().get_xyz()
    # HACK convert coordinates to a tangent plane on lat 0° lon 0°, with north pointing down for easier drawing
    up = (cartesian[0] - R_earth).value
    easting = cartesian[1].value
    northing = cartesian[2].value
    return np.array([easting, -northing, up])

# Assume static satelites because orbital mechanics is hard
# TODO implement http://grapenthin.org/teaching/geop555/LAB03_position_estimation.html
SATELLITE_HEIGHT = 20_200_000
SATELLITE_POSITIONS = np.array([
    getSatelliteAtAngle(45 * u.deg, 0 * u.deg),
    getSatelliteAtAngle(0 * u.deg, 45 * u.deg),
    getSatelliteAtAngle(0 * u.deg, -45 * u.deg),
    getSatelliteAtAngle(-45 * u.deg, 0 * u.deg),
    getSatelliteAtAngle(-45 * u.deg, 60 * u.deg),
    getSatelliteAtAngle(60 * u.deg, 60 * u.deg),
], dtype=np.float64)

print(SATELLITE_POSITIONS)

SATELLITE_CLOCK_BIAS = np.array([
    5 * 1e-6,
    -10 * 1e-6,
    7 * 1e-6,
    10 * 1e-6,
    60 * 1e-6,
    -100 * 1e-6,
], dtype=np.float64)

SATELLITE_NUMBER = SATELLITE_CLOCK_BIAS.shape[0]

RECEIVER_CLOCK_BIAS = np.float64(1 * 1e-3)
# Walk of 0.1ns per second
# From https://www.e-education.psu.edu/geog862/node/1716
RECEIVER_CLOCK_WALK = np.float64(1 * 1e-6)

SATELLITE_NOISE_STD = 0.0

def toVector2(array: array3d) -> Vector2:
    return Vector2(array[0].item(), array[1].item())

def getGnssPositionTaylor(pseudorange, gnss_satelite_position_aproximation, gnss_satelite_time_bias_aproximation, xtol):
    """
    A linearization of the pseudorange error, via a taylor aproximation, is minimized.
    But has problems on numerical precision (substracts large floating values)
    From http://www.grapenthin.org/notes/2019_03_11_pseudorange_position_estimation/
    :return:
    """
    gnss_position_aproximation = np.array([1, 1, 1], dtype=np.float64)
    gnss_receiver_clock_bias_approximation = np.float64(1e-6)
    gnss_pseudorange_approximation = (np.linalg.norm(gnss_satelite_position_aproximation - gnss_position_aproximation, axis=1)
                                      + (gnss_satelite_time_bias_aproximation + gnss_receiver_clock_bias_approximation) * scipy.constants.c)
    gnss_position_error = np.inf

    while gnss_position_error > xtol:
        delta_gnss_pseudorange = pseudorange.copy() - gnss_pseudorange_approximation

        delta_satelites = gnss_position_aproximation - gnss_satelite_position_aproximation
        cs = np.ones((1, SATELLITE_NUMBER), dtype=np.float64) * scipy.constants.c

        G = np.concatenate((delta_satelites, cs.T), axis=1) / gnss_pseudorange_approximation.reshape((-1, 1))

        m = (np.linalg.pinv(G) @ delta_gnss_pseudorange.T).reshape(-1)
        gnss_position_delta = m[:3]
        gnss_clock_bias_delta = m[-1]

        gnss_position_aproximation += gnss_position_delta
        gnss_receiver_clock_bias_approximation += gnss_clock_bias_delta
        gnss_pseudorange_approximation += delta_gnss_pseudorange

        gnss_position_error = np.linalg.norm(gnss_position_delta)

    return gnss_position_aproximation, gnss_receiver_clock_bias_approximation, gnss_position_error


def getGnssPositionScipy(pseudorange, gnss_satelite_position_aproximation, gnss_satelite_time_bias_aproximation, xtol):
    """
    A linearization of the pseudorange error, via scipy.optimize.least_squares
    :return:
    """
    def fun(x):
        gnss_position_aproximation = x[:3]
        gnss_receiver_clock_bias_approximation = x[-1]
        pseudorange_aproximation = (np.linalg.norm(gnss_satelite_position_aproximation - gnss_position_aproximation, axis=1)
                                    + (gnss_satelite_time_bias_aproximation + gnss_receiver_clock_bias_approximation) * scipy.constants.c)
        return (pseudorange - pseudorange_aproximation).reshape(-1)

    result = scipy.optimize.least_squares(fun, np.array([1, 1, 1, 1], dtype=np.float64), xtol=xtol)
    gnss_position_aproximation = result.x[:3]
    gnss_receiver_clock_bias_approximation = result.x[-1]
    gnss_position_error = result.cost

    return gnss_position_aproximation, gnss_receiver_clock_bias_approximation, gnss_position_error


def getGnssPVT(rng, player_positions, delta, reciever_clock_bias) -> Tuple[array3d, array3d, np.float64]:
    player_position = player_positions[-1]

    # TODO Ionospheric delay is function of the angle to the satelite (Klobuchar delay model)
    # See https://insidegnss.com/auto/marapr15-WP.pdf
    # And https://gssc.esa.int/navipedia/index.php/Klobuchar_Ionospheric_Model
    # And GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
    ionospheric_delay = rng.normal(0.0, 0.05, (1,))

    # TODO Troposferic delay is divided intro dry and wet and varies acording to satellite elevation (Saastamoinen model)
    # Dry constant is set to 10cm
    # See https://gssc.esa.int/navipedia/index.php/Galileo_Tropospheric_Correction_Model
    # And GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
    tropospheric_delay = 0.10

    # See GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
    bias_difference = scipy.constants.c * (SATELLITE_CLOCK_BIAS.reshape((-1)) + reciever_clock_bias)
    range = np.linalg.norm(SATELLITE_POSITIONS - player_position, axis=1).reshape((-1))

    # Assume open field
    # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
    multipath_bias = 0

    # Random noise
    # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
    epsilon = rng.normal(0.0, SATELLITE_NOISE_STD, (SATELLITE_NUMBER,))
    pseudorange = range + bias_difference + tropospheric_delay + ionospheric_delay + multipath_bias + epsilon

    # Computation of the satellite orbit, from ephimeris
    # Assume satelite position is known because ephimeris is transmitted during the first fix
    # From https://gssc.esa.int/navipedia/index.php/Coordinates_Computation_from_Almanac_Data
    gnss_satelite_position_aproximation = SATELLITE_POSITIONS.copy()
    gnss_satelite_time_bias_aproximation = SATELLITE_CLOCK_BIAS.copy()

    gnss_position_aproximation, gnss_receiver_clock_bias_approximation, gnss_position_error = (
        getGnssPositionScipy(pseudorange, gnss_satelite_position_aproximation, gnss_satelite_time_bias_aproximation,1e-8))

    A = np.concatenate(
        (
        (SATELLITE_POSITIONS - player_position) / np.linalg.norm(SATELLITE_POSITIONS - player_position, axis=1).reshape(
            (-1, 1)),
        np.ones((1, SATELLITE_NUMBER)).T),
        axis=1
    )
    Q = np.linalg.inv(A.T @ A)
    gdop = np.sqrt(np.sum([Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3]]))
    hdop = np.sqrt(np.sum([Q[0, 0], Q[1, 1]]))
    vdop = np.sqrt(Q[2, 2])
    pdop = np.sqrt(np.sum([Q[0, 0], Q[1, 1], Q[2, 2]]))

    print(f"===")
    print(f"Cost {gnss_position_error}, estimated position {gnss_position_aproximation}m, estimated reciever clock bias {gnss_receiver_clock_bias_approximation}s")
    print(f"GDOP {gdop} HDOP {hdop} VDOP {vdop} PDOP {pdop}")
    print(f"Error {np.linalg.norm(gnss_position_aproximation-player_position)}m, real position {player_position}m, real reciever clock bias {reciever_clock_bias}s")

    # TODO replace with GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.2
    gnss_velocity = rng.normal((player_positions[-1] - player_positions[-2]) / delta, (1, 1, 0), (3,))

    return gnss_position_aproximation, gnss_velocity, gnss_receiver_clock_bias_approximation

def main():
    width, height = 800, 450
    init_window(width, height, "Hello")

    rng = np.random.default_rng()

    player_positions: List[array3d] = [np.array([20, 20, 0], dtype=np.float64)]
    gnss_positions: List[array3d] = []
    gnss_velocities: List[array3d] = []
    receiver_clock_bias = RECEIVER_CLOCK_BIAS
    time_since_gnss = np.inf

    while not window_should_close():
        delta = get_frame_time()

        time_since_gnss += delta

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

        if time_since_gnss > 1/GNSS_FREQUENCY:
            # See GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
            if time_since_gnss < np.inf:
                receiver_clock_bias += RECEIVER_CLOCK_WALK * rng.normal() * time_since_gnss
            gnss_position, gnss_velocity, gnss_time_bias = getGnssPVT(rng, player_positions, delta, receiver_clock_bias)
            gnss_positions.append(gnss_position)
            gnss_velocities.append(gnss_velocity)
            time_since_gnss = 0

        player_position_px = player_position * METERS_TO_PIXELS
        gnss_velocity_px = gnss_velocities[-1] / MOVEMENT_SPEED_METERS_PER_SECOND * METERS_TO_PIXELS

        begin_drawing()
        clear_background(WHITE)
        draw_circle_v(toVector2(player_position_px), 5, RED)

        for gnss_position in gnss_positions:
            draw_circle_v(toVector2(gnss_position * METERS_TO_PIXELS), 2, GREEN)

        draw_line_v(toVector2(player_position_px), toVector2(player_position_px + gnss_velocity_px), BLUE)


        draw_rectangle(width - 200, 0, width, 200, WHITE)

        x = player_position
        x = x / np.array([10_000, 10_000, 1])
        x = x + np.array([width - 100, 100, 0])
        draw_circle_v(toVector2(x), 2, RED)

        for i, satellite_position in enumerate(SATELLITE_POSITIONS):
            x = satellite_position
            x = x / np.array([1e6, 1e6, 1e6])
            x = x + np.array([width - 100, 100, 0])
            draw_circle_v(toVector2(x), 2, GREEN)
            draw_text(f"{i}", np.int64(x[0]).item(), np.int64(x[1]).item(), 2, BLACK)
        draw_rectangle_lines(width - 200, 0, width, 200, BLACK)

        end_drawing()
    close_window()

if __name__ == '__main__':
    main()