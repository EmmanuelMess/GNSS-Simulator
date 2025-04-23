import numpy as np
from pyray import Vector2

from numpy_types import array3d

def toVector2(array: array3d) -> Vector2:
    return Vector2(array[0].item(), array[1].item())

def ecef2llh(position: array3d):
    """
     From https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    """
    # WGS84 constants
    a = np.float64(6_378_137.0)
    f = 1 / np.float64(298.257_223_563)
    b = a * (1.0 - f)
    e_2 = 2 * f - f**2

    x, y, z = position[0], position[1], position[2]

    l = np.atan2(y, x)

    p = np.hypot(x, y)

    if p < 1e-20:
        if z >= 0:
            return np.array([np.pi/2, 0, z - b])
        else:
            return np.array([-np.pi/2, 0, -z - b])

    theta = np.atan(z/((1- e_2) * p))
    for i in range(100):
        N = a / np.sqrt(1 - e_2 * np.sin(theta) ** 2)
        h = p / np.cos(theta) - N
        new_theta = np.atan(z / ((1 - e_2 * (N / (N + h))) * p))

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
    seconds_in_week = 7 * 24 * 60 * 60

    return time_gps % seconds_in_week


def time_gps2week_number(time_gps: np.float64) -> int:
    """
    Given a time in seconds since GPS epoch, returns the amount of weeks since GPS epoch. This can be done this way
    because the GPS time is aligned with the start of the week
    """
    seconds_in_week = 7 * 24 * 60 * 60

    return np.int64(np.floor(time_gps / seconds_in_week)).item()