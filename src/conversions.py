import numpy as np
from pyray import Vector2

from src.constants import OMEGA_EARTH
from src.numpy_types import array3d

DAYS_IN_WEEK = np.int64(7)
SECONDS_IN_WEEK = np.int64(DAYS_IN_WEEK * 24 * 60 * 60)
SECONDS_IN_NANOSECONDS = np.float64(1e9)

# WGS84 constants
A_EARTH = np.float64(6_378_137.0)
F_EARTH = 1 / np.float64(298.257_223_563)
B_EARTH = A_EARTH * (1.0 - F_EARTH)
E2_EARTH = 2 * F_EARTH - F_EARTH ** 2


def toVector2(array: array3d, x_axis: array3d, y_axis: array3d) -> Vector2:
    x = np.dot(array, x_axis) / np.linalg.norm(x_axis)
    y = np.dot(array, y_axis) / np.linalg.norm(y_axis)
    return Vector2(x.item(), y.item())


def llh2ecef(position: array3d) -> array3d:
    """
     From https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    """
    lat = position[0]
    lon = position[1]
    alt = position[2]

    n = A_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)

    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = ((1.0 - E2_EARTH) * n + alt) * np.sin(lat)

    return np.array([x, y, z], dtype=np.float64)


def ecef2llh(position: array3d):
    """
     From https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    """
    x, y, z = position[0], position[1], position[2]

    l = np.atan2(y, x)

    p = np.hypot(x, y)

    if p < 1e-20:
        if z >= 0:
            return np.array([np.pi/2, 0, z - B_EARTH])
        else:
            return np.array([-np.pi/2, 0, -z - B_EARTH])

    theta = np.atan(z/((1- E2_EARTH) * p))
    for i in range(100):
        N = A_EARTH / np.sqrt(1 - E2_EARTH * np.sin(theta) ** 2)
        h = p / np.cos(theta) - N
        new_theta = np.atan(z / ((1 - E2_EARTH * (N / (N + h))) * p))

        if np.abs(theta - new_theta) < 1e-9: # See https://wiki.openstreetmap.org/wiki/Precision_of_coordinates
            break
        theta = new_theta

    return np.array([theta, l, h])


def ecef2aer(receiver_ecef: array3d, satellite_ecef: array3d):
    """
    https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    """
    receiver_llh = ecef2llh(receiver_ecef)

    theta, l, h = receiver_llh[0], receiver_llh[1], receiver_llh[2]

    delta = satellite_ecef - receiver_ecef
    delta_unit_vector = delta / np.linalg.norm(delta)

    e_unit = np.array([- np.sin(l), np.cos(l), 0])
    n_unit = np.array([- np.cos(l) * np.sin(theta), -np.sin(l) * np.sin(theta), np.cos(theta)])
    u_unit = np.array([np.cos(l) * np.cos(theta), np.sin(l) * np.cos(theta), np.sin(theta)])

    elevation = np.asin(np.dot(delta_unit_vector, u_unit))
    atan = np.atan2(np.dot(delta_unit_vector, e_unit), np.dot(delta_unit_vector, n_unit))
    azimuth = atan if atan >= 0 else atan + 2*np.pi
    range = np.linalg.norm(delta)

    return np.array([azimuth, elevation, range])

def position_ecef2eci(position_ecef: np.ndarray, dt: np.float64) -> array3d:
    # ECEF to ECI provided by IS-GPS-200N section 20.3.3.4.3.3.2
    theta = OMEGA_EARTH * dt
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return (rotation_matrix @ position_ecef.reshape((-1, 3, 1))).reshape(position_ecef.shape)

def velocity_ecef2eci(position_ecef: np.ndarray, velocity_ecef: np.ndarray, dt: np.float64) -> array3d:
    # ECEF to ECI provided from Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems section 2.5.1
    angular_velocity_matrix = np.array([
        [0, - OMEGA_EARTH, 0],
        [OMEGA_EARTH, 0, 0],
        [0, 0, 0]
    ])

    velocity_of_position = (angular_velocity_matrix @ position_ecef.reshape((-1, 3, 1))).reshape(position_ecef.shape)
    velocity_eci = position_ecef2eci(velocity_ecef + velocity_of_position, dt)
    return velocity_eci

def position_eci2ecef(positions_eci: np.ndarray, dt: np.float64) -> array3d:
    # TODO test
    # ECI to ECEF provided from Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems section 2.5.1
    theta = OMEGA_EARTH * dt
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return (rotation_matrix @ positions_eci.reshape((-1, 3, 1))).reshape(positions_eci.shape)

def velocity_eci2ecef(positions_eci: np.ndarray, velocities_eci: np.ndarray, dt: np.float64) -> array3d:
    # TODO test
    # ECI to ECEF provided from Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems section 2.5.1
    angular_velocity_matrix = np.array([
        [0, - OMEGA_EARTH, 0],
        [OMEGA_EARTH, 0, 0],
        [0, 0, 0]
    ])
    velocity_of_position = (angular_velocity_matrix @ positions_eci.reshape((-1, 3, 1))).reshape(positions_eci.shape)
    velocity_ecef = position_eci2ecef(velocities_eci - velocity_of_position, dt)
    return velocity_ecef


def pos2enu_base(position_llh: array3d) -> (array3d, array3d, array3d):
    theta, l, h = position_llh[0], position_llh[1], position_llh[2]

    e_unit = np.array([- np.sin(l), np.cos(l), 0])
    n_unit = np.array([- np.cos(l) * np.sin(theta), -np.sin(l) * np.sin(theta), np.cos(theta)])
    u_unit = np.array([np.cos(l) * np.cos(theta), np.sin(l) * np.cos(theta), np.sin(theta)])

    return e_unit, n_unit, u_unit


def rad2semicircles(value):
    return value / np.pi

def semicircles2rad(value):
    return value * np.pi

def seconds2day_of_year(value):
    # The GPS epoch is not at the start of the year and the day should start at 1
    return (((1 + 5) * 24 * 60 * 60 + value) / (24 * 60 *60)) % 365.25

def rev_per_day2rad_per_second(value):
    return 2 * np.pi / (24*60*60)

def time_gps2seconds_of_week(time_gps: np.float64) -> np.float64:
    """
    Given a time in seconds since GPS epoch, returns the seconds in the week. This can be done this way because the GPS
    time is aligned with the start of the week
    """
    return np.fmod(time_gps, SECONDS_IN_WEEK)


def time_gps2week_number(time_gps: np.float64) -> int:
    """
    Given a time in seconds since GPS epoch, returns the amount of weeks since GPS epoch. This can be done this way
    because the GPS time is aligned with the start of the week
    """

    return np.int64(np.floor(time_gps / SECONDS_IN_WEEK)).item()


def gps_seconds_wrap(seconds_of_week: np.float64) -> np.float64:
    """
    Wrap the seconds of week into the last of next week
    """
    if seconds_of_week > SECONDS_IN_WEEK / 2:
        return seconds_of_week - SECONDS_IN_WEEK
    elif seconds_of_week < -SECONDS_IN_WEEK / 2:
        return seconds_of_week + SECONDS_IN_WEEK
    else:
        return seconds_of_week
