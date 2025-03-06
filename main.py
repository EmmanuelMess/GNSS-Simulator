from typing import List, Tuple

from astropy.coordinates import SphericalRepresentation
from astropy.constants import R_earth, M_earth
import astropy.units as u
import numpy as np
import scipy
from pyray import *
from raylib import *

PIXELS_TO_METERS = 1/10
METERS_TO_PIXELS = 1/PIXELS_TO_METERS
MOVEMENT_SPEED_METERS_PER_SECOND = 5.0
GNSS_MESSAGE_FREQUENCY = 5 # Hz

array3d = np.ndarray[(2,), np.float64]
tau = np.pi * 2

MEO = np.float64(20_200_000) + R_earth.value

def getSatelliteAtAngle(lat, lon) -> array3d:
    # TODO replace astropy with a generic spherical geometry library
    spherical = SphericalRepresentation(lat=lat, lon=lon, distance=MEO * u.m )
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

SATELLITE_CLOCK_BIAS = np.array([
    5 * 1e-6,
    -10 * 1e-6,
    7 * 1e-6,
    10 * 1e-6,
    60 * 1e-6,
    -100 * 1e-6,
], dtype=np.float64)

SATELLITE_NUMBER = SATELLITE_CLOCK_BIAS.shape[0]

EARTH_MASS = M_earth.value
SEMI_MAJOR_AXIS = 26_560_000

# From Vis-viva equation
SATELLITE_VELOCITY = np.sqrt(scipy.constants.G * EARTH_MASS * ( 2 / MEO - 1 / SEMI_MAJOR_AXIS))

SATELLITE_VELOCITY_VECTORS = np.array([
    [0, 0.445, np.sqrt(1-0.445**2)],
    [0, 0.583, np.sqrt(1-0.583**2)],
    [0, 0.088, np.sqrt(1-0.088**2)],
    [0, 0.981, np.sqrt(1-0.981**2)],
    [0, 0.695, np.sqrt(1-0.695**2)],
    [0, 0.095, np.sqrt(1-0.095**2)],
], dtype=np.float64) * SATELLITE_VELOCITY

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
CUTOFF_ANGLE = np.deg2rad(5)

# This noise level does not affect the reciever, because it can correct for the noise
NOISE_CORRECTION_LEVEL = np.float64(7) # dB
# This noise level causes the receiver to lose the fix
NOISE_FIX_LOSS_LEVEL = np.float64(40) # dB

# Effect of noise in the range (NOISE_CORRECTION_LEVEL, NOISE_FIX_LOSS_LEVEL), in terms of the mean meter error of the
# pseudorange
NOISE_EFFECT_RATE = np.float64(5) / (NOISE_FIX_LOSS_LEVEL - NOISE_CORRECTION_LEVEL) # m / dB

GPS_L1_FREQUENCY = np.float64(1_575.42) * 1e6 # Hz
GPS_L5_FREQUENCY = np.float64(1_176.45) * 1e6 # Hz
GNSS_SIGNAL_FREQUENCY = GPS_L1_FREQUENCY

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


class Solver:
    def __init__(self, satellite_positions, satellite_clock_bias, satellite_velocities,
                 satellite_frequencies):
        self.satellite_positions = satellite_positions
        self.satellite_clock_bias = satellite_clock_bias
        self.satellite_velocities = satellite_velocities
        self.satellite_frequencies = satellite_frequencies
        self.satellite_number = satellite_positions.shape[0]

    def solve_position(self, pseudorange, xtol):
        """
        A linearization of the pseudorange error, via a taylor aproximation, is minimized.
        But has problems on numerical precision (substracts large floating values)
        From http://www.grapenthin.org/notes/2019_03_11_pseudorange_position_estimation/
        Numeric fixes from https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
        :return:
        """
        gnss_position_aproximation = np.array([1, 1, 1], dtype=np.float64)
        gnss_receiver_clock_bias_approximation = np.float64(1e-6) * scipy.constants.c
        gnss_position_error = np.inf

        # TODO add satelite weighting

        for _ in range(100):
            if gnss_position_error < xtol:
                break

            gnss_pseudorange_approximation = (
                        np.linalg.norm(self.satellite_positions - gnss_position_aproximation, axis=1)
                        + (gnss_receiver_clock_bias_approximation - self.satellite_clock_bias * scipy.constants.c))

            delta_gnss_pseudorange = pseudorange.copy() - gnss_pseudorange_approximation

            delta_satelites = gnss_position_aproximation - self.satellite_positions
            cs = np.ones((1, self.satellite_number), dtype=np.float64)

            G = np.concatenate((delta_satelites / gnss_pseudorange_approximation.reshape((-1, 1)), cs.T), axis=1)

            m = (np.linalg.pinv(G) @ delta_gnss_pseudorange.T).reshape(-1)
            gnss_position_delta = m[:3]
            gnss_clock_bias_delta = m[-1]

            gnss_position_aproximation += gnss_position_delta
            gnss_receiver_clock_bias_approximation += gnss_clock_bias_delta

            gnss_position_error = np.linalg.norm(gnss_position_delta)

        gnss_receiver_clock_bias_approximation /= scipy.constants.c

        return gnss_position_aproximation, gnss_receiver_clock_bias_approximation, np.linalg.norm(delta_gnss_pseudorange)


    def solve_position_scipy(self, pseudorange, xtol):
        """
        A linearization of the pseudorange error, via scipy.optimize.least_squares
        This is an adaptation of getGnssPositionTaylor
        :return:
        """
        # TODO add satelite weighting

        def pseudoranges_approximation(gnss_position_aproximation, gnss_receiver_clock_bias_approximation):
            return (np.linalg.norm(self.satellite_positions - gnss_position_aproximation, axis=1)
                                        + (gnss_receiver_clock_bias_approximation - self.satellite_clock_bias * scipy.constants.c))

        def fun(x):
            gnss_position_aproximation = x[:3]
            gnss_receiver_clock_bias_approximation = x[-1]
            approx = pseudoranges_approximation(gnss_position_aproximation, gnss_receiver_clock_bias_approximation)
            return (pseudorange - approx).reshape(-1)

        def jac(x):
            gnss_position_aproximation = x[:3]
            gnss_receiver_clock_bias_approximation = x[-1]

            delta_satelites = self.satellite_positions - gnss_position_aproximation
            cs = np.ones((1, self.satellite_number), dtype=np.float64) * -1

            approx = pseudoranges_approximation(gnss_position_aproximation, gnss_receiver_clock_bias_approximation)

            G = np.concatenate((delta_satelites / approx.reshape((-1, 1)), cs.T), axis=1)

            return G

        result = scipy.optimize.least_squares(fun, x0=np.array([1, 1, 1, 1e-6 * scipy.constants.c], dtype=np.float64), method='lm', jac=jac, xtol=xtol)
        gnss_position_aproximation = result.x[:3]
        gnss_receiver_clock_bias_approximation = result.x[-1] / scipy.constants.c
        gnss_position_error = np.linalg.norm(result.fun)

        return gnss_position_aproximation, gnss_receiver_clock_bias_approximation, gnss_position_error

    def solve_velocity(self, direct_doppler, receiver_position, xtol):
        """
        A linearization of the pseudorange rate error, and least squares solutions
        From Global Positioning System section 6.2.1, adapted from getGnssPositionTaylor
        """
        # TODO add satelite weighting

        pseudorange_rates = scipy.constants.c / self.satellite_frequencies * direct_doppler

        gnss_velocity_aproximation = np.array([1, 1, 1], dtype=np.float64)
        gnss_receiver_clock_drift_approximation = np.float64(1e-6)
        gnss_velocity_error = np.inf

        # TODO add satelite weighting

        for _ in range(100):
            if gnss_velocity_error < xtol:
                break

            velocity_difference = gnss_velocity_aproximation - self.satellite_velocities
            satellite_user_delta = self.satellite_positions - receiver_position
            satellite_line_of_sight = satellite_user_delta / np.linalg.norm(satellite_user_delta, axis=1).reshape((-1, 1))
            velocity_scalar_projection = np.sum(velocity_difference * satellite_line_of_sight, axis=1)
            pseudorange_rates_approximation = velocity_scalar_projection + gnss_receiver_clock_drift_approximation

            delta_gnss_pseudorange_rates = pseudorange_rates - pseudorange_rates_approximation

            cs = np.ones((1, self.satellite_number), dtype=np.float64)

            G = np.concatenate((satellite_line_of_sight, cs.T), axis=1)

            m = (np.linalg.pinv(G) @ delta_gnss_pseudorange_rates.T).reshape(-1)
            gnss_velocity_delta = m[:3]
            gnss_clock_drift_delta = m[-1]

            gnss_velocity_aproximation += gnss_velocity_delta
            gnss_receiver_clock_drift_approximation += gnss_clock_drift_delta

            gnss_velocity_error = np.linalg.norm(gnss_velocity_delta)

        return gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation, gnss_velocity_error


    def solve_velocity_scipy(self, direct_doppler, receiver_position, xtol):
        """
        A linearization of the pseudorange rate error, via scipy.optimize.least_squares
        This is an adaptation of getGnssPositionScipy
        From Navigation from Low Earth Orbit – Part 2: Models, Implementation, and Performance section 2.2
        """
        # TODO add satelite weighting

        pseudorange_rate = (- scipy.constants.c / self.satellite_frequencies) * direct_doppler

        def pseudorange_rates_approximation(gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation):
            rate_difference = gnss_velocity_aproximation - self.satellite_velocities

            line_of_sight = receiver_position - self.satellite_positions
            line_of_sight_unit = line_of_sight / np.linalg.norm(line_of_sight, axis=1).reshape((-1,1))
            velocity_scalar_projection = np.sum(rate_difference * line_of_sight_unit, axis=1)

            clock_drift_effect = gnss_receiver_clock_drift_approximation - scipy.constants.c * self.satellite_clock_drift

            return velocity_scalar_projection + clock_drift_effect

        def fun(x):
            gnss_velocity_aproximation = x[:3]
            gnss_receiver_clock_drift_approximation = x[-1]
            approx = pseudorange_rates_approximation(gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation)
            return (pseudorange_rate - approx).reshape(-1)

        result = scipy.optimize.least_squares(fun, x0=np.array([1, 1, 1, 1e-6 * scipy.constants.c], dtype=np.float64), method='lm', xtol=xtol)
        gnss_velocity_aproximation = result.x[:3]
        gnss_receiver_clock_drift_approximation = result.x[-1] / scipy.constants.c
        gnss_velocity_error = np.linalg.norm(result.fun)

        return gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation, gnss_velocity_error


class Simulator:
    def __init__(self, rng, satellite_positions, satellite_clock_bias, satellite_velocities,
                 satellite_frequency, satellite_alphas, satellite_betas, noise_correction_level, noise_fix_loss_level,
                 noise_effect_rate, satellite_noise_std):
        self.rng = rng
        self.satellite_positions = satellite_positions
        self.satellite_clock_bias = satellite_clock_bias
        self.satellite_velocities = satellite_velocities
        self.satellite_frequency = satellite_frequency # TODO make a vector
        self.satellite_alphas = satellite_alphas
        self.satellite_betas = satellite_betas
        self.satellite_noise_std = satellite_noise_std
        self.noise_correction_level = noise_correction_level
        self.noise_fix_loss_level = noise_fix_loss_level
        self.noise_effect_rate = noise_effect_rate
        self.satellite_amount = satellite_positions.shape[0]

    def _ionospheric_delay_calculation(self, receiver_ecef, time_of_week_gps_seconds):
        # From https://gssc.esa.int/navipedia/index.php/Klobuchar_Ionospheric_Model
        # And GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        # Reference implementation https://geodesy.noaa.gov/gps-toolbox/ovstedal/klobuchar.for

        receiver_llh = ecef2llh(receiver_ecef)
        satellites_aer_rad = np.array([ecef2aer(receiver_ecef, satellite_position) for satellite_position in self.satellite_positions])

        if receiver_llh[1] < 0:
            receiver_llh[1] += 2 * np.pi # because for some reason the algorithm uses longitude in [0, 2*pi]

        receiver_semicircles = rad2semicircles(receiver_llh)

        elevation_semicircles = rad2semicircles(satellites_aer_rad[:, 1])

        ec_angle = 0.0137 / (elevation_semicircles + 0.11) - 0.022

        subionospheric_latitude_semicircles = receiver_semicircles[0] + ec_angle * np.cos(satellites_aer_rad[:, 0])
        subionospheric_latitude_semicircles = np.clip(subionospheric_latitude_semicircles, a_min=-0.416, a_max=0.416)
        subionospheric_latitude_rad = semicircles2rad(subionospheric_latitude_semicircles)

        subionospheric_longitude_semicircles = receiver_semicircles[1] + ec_angle * np.sin(satellites_aer_rad[:, 0]) / np.cos(subionospheric_latitude_rad)

        geomagnetic_latitude_semicircles = subionospheric_latitude_semicircles + 0.064 * np.cos(semicircles2rad(subionospheric_longitude_semicircles - 1.617))

        local_time_pierce_point = 43_200 * subionospheric_longitude_semicircles + time_of_week_gps_seconds
        local_time_pierce_point = np.divmod(local_time_pierce_point, 86_400)[1]
        local_time_pierce_point[local_time_pierce_point >= 86_400] -= 86_400
        local_time_pierce_point[local_time_pierce_point < 0] += 86_400

        slant_factor = 1 + 16 * (0.53 - elevation_semicircles) ** 3

        period_ionospheric_delay = self.satellite_betas[:, 0] \
                                   + self.satellite_betas[:, 1] * geomagnetic_latitude_semicircles \
                                   + self.satellite_betas[:, 2] * geomagnetic_latitude_semicircles**2 \
                                   + self.satellite_betas[:, 3] * geomagnetic_latitude_semicircles**3

        period_ionospheric_delay = np.clip(period_ionospheric_delay, a_min=72_000, a_max=None)

        phase_ionospheric_delay = 2 * np.pi * (local_time_pierce_point - 50_400) / period_ionospheric_delay

        amplitude_ionospheric_delay = self.satellite_alphas[:, 0] \
                                    + self.satellite_alphas[:, 1] * geomagnetic_latitude_semicircles \
                                    + self.satellite_alphas[:, 2] * geomagnetic_latitude_semicircles**2 \
                                    + self.satellite_alphas[:, 3] * geomagnetic_latitude_semicircles**3
        amplitude_ionospheric_delay = np.clip(amplitude_ionospheric_delay, a_min=0, a_max=None)


        day = (5 * 1e-9 + amplitude_ionospheric_delay * (
                    1 - phase_ionospheric_delay ** 2 / 2 + phase_ionospheric_delay ** 4 / 24)) * slant_factor
        night = 5 * 1e-9 * slant_factor

        ionospheric_delay_gps_l1 = np.where(np.abs(phase_ionospheric_delay) > 1.57, night, day)
        ionospheric_delay = (GPS_L1_FREQUENCY / self.satellite_frequency) ** 2 * ionospheric_delay_gps_l1
        return ionospheric_delay

    def _tropospheric_average_table(self, latitude):
        latitudes = np.deg2rad(np.array([     15,      30,      45,      60,      75], dtype=np.float64))
        average_pressures    = np.array([1013.25, 1017.25, 1015.75, 1011.75, 1013.00], dtype=np.float64)
        average_temperatures = np.array([ 299.65,  294.15,  283.15,  272.15,  263.65], dtype=np.float64)
        average_es           = np.array([  26.31,   21.79,   11.66,    6.78,    4.11], dtype=np.float64)
        average_betas        = np.array([6.30e-3,  6.5e-3, 5.58e-3, 5.39e-3, 4.53e-3], dtype=np.float64)
        average_lambdas      = np.array([   2.77,    3.15,    2.57,    1.81,    1.55], dtype=np.float64)

        average_pressure = np.interp(latitude, latitudes, average_pressures)
        average_temperature = np.interp(latitude, latitudes, average_temperatures)
        average_e = np.interp(latitude, latitudes, average_es)
        average_beta = np.interp(latitude, latitudes, average_betas)
        average_lambda = np.interp(latitude, latitudes, average_lambdas)

        return average_pressure, average_temperature, average_e, average_beta, average_lambda

    def _tropospheric_deltas_table(self, latitude):
        latitudes = np.deg2rad(np.array([ 15,      30,      45,      60,      75], dtype=np.float64))
        delta_pressures      = np.array([0.0,   -3.75,   -2.25,   -1.75,    -0.5], dtype=np.float64)
        delta_temperatures   = np.array([0.0,     7.0,    11.0,    15.0,    14.5], dtype=np.float64)
        delta_es             = np.array([0.0,    8.85,    7.24,    5.36,    3.39], dtype=np.float64)
        delta_betas          = np.array([0.0, 0.25e-3, 0.32e-3, 0.81e-3, 0.62e-3], dtype=np.float64)
        delta_lambdas        = np.array([0.0,    0.33,    0.46,    0.74,     0.3], dtype=np.float64)

        delta_pressure = np.interp(latitude, latitudes, delta_pressures)
        delta_temperature = np.interp(latitude, latitudes, delta_temperatures)
        delta_e = np.interp(latitude, latitudes, delta_es)
        delta_beta = np.interp(latitude, latitudes, delta_betas)
        delta_lambda = np.interp(latitude, latitudes, delta_lambdas)

        return delta_pressure, delta_temperature, delta_e, delta_beta, delta_lambda

    def _per_satelite_tropospheric_delay(self, position_llh, satellite_elevation, day_of_year, cutoff_angle):
        # UNB4 model
        # See https://gssc.esa.int/navipedia/index.php/Galileo_Tropospheric_Correction_Model
        # And Assessment and Development of a Tropospheric Delay Model for Aircraft Users of the Global Positioning System
        # And GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1

        if satellite_elevation <= cutoff_angle:
            return 0

        player_latitude = np.abs(position_llh[0])
        northern = position_llh[0] > 0
        day_of_year_min = 28 if northern else 211

        elevation_effect = 1.001 / np.sqrt(0.002001 + np.sin(satellite_elevation) ** 2)

        season_multiplier = np.cos(2 * np.pi * (day_of_year - day_of_year_min) / 365.25)
        average_pressure, average_temperature, average_e, average_beta, average_lambda = self._tropospheric_average_table(
            player_latitude)
        delta_pressure, delta_temperature, delta_e, delta_beta, delta_lambda = self._tropospheric_deltas_table(
            player_latitude)

        pressure = average_pressure - delta_pressure * season_multiplier  # mbar
        temperature = average_temperature - delta_temperature * season_multiplier  # K
        e = average_e - delta_e * season_multiplier  # mbar # vapour pressure
        beta = average_beta - delta_beta * season_multiplier  # K/m #  temperature "lapse" rate
        l = average_lambda - delta_lambda * season_multiplier  # 1 # water vapour "lapse" rate

        # TODO this is wrong, this has height over the ellipsoid, but requires height over the sea
        h = position_llh[2]  # m # height above mean-sea-level

        k1 = 77.604  # K/mbar
        k2 = 382_000  # K²/mbar
        Rd = 287.054  # J / Kg / K
        gm = 9.784  # m / s²
        g = 9.80665  # m / s²

        delay_0_dry = 1e-6 * k1 * Rd * pressure / gm
        delay_0_wet = (1e-6 * k2 * Rd / ((l + 1) * gm - beta * Rd)) * (e / temperature)

        base = 1 - beta * h / temperature
        delay_dry = base ** (g / (Rd * beta)) * delay_0_dry
        delay_wet = base ** ((l + 1) * g / (Rd * beta) - 1) * delay_0_wet

        return (delay_dry + delay_wet) * elevation_effect

    def _tropospheric_delay_calculation(self, position_ecef, day_of_year, cutoff_angle):
        # Troposferic delay is divided intro dry and wet and varies acording to satellite elevation (Saastamoinen model)
        # And Global Positioning System: Signals, Measurements, and Performance section 5.3.3

        position_llh = ecef2llh(position_ecef)
        satellites_aer = np.array([ecef2aer(position_ecef, satellite_position) for satellite_position in self.satellite_positions])

        tropospheric_delay = np.array([self._per_satelite_tropospheric_delay(position_llh, elevation, day_of_year, cutoff_angle) for elevation in satellites_aer[:, 1]])
        return tropospheric_delay


    def _saastamoinen_model(self, position_ecef, pressure, temperature, partial_pressure_water_vapor):
        # From Global Positioning System: Signals, Measurements, and Performance section 5.3.3
        position_llh = ecef2llh(position_ecef)

        dry_delay = 0.002277 * (1+0.0026 * np.cos(2 * position_llh[0]) + 0.00028 * position_llh[2] * 1e-3) * pressure
        wet_delay = 0.002277 * (1255 / temperature + 0.05) * partial_pressure_water_vapor

        return dry_delay, wet_delay


    def get_pseudoranges(self, player_position_ecef, reciever_clock_bias, time_gps):
        ionospheric_delay = self._ionospheric_delay_calculation(player_position_ecef, time_gps)

        day_of_year = seconds2day_of_year(time_gps)
        tropospheric_delay = self._tropospheric_delay_calculation(player_position_ecef, day_of_year, CUTOFF_ANGLE)

        # See GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        bias_difference = scipy.constants.c * (reciever_clock_bias - self.satellite_clock_bias.reshape((-1)))
        range = np.linalg.norm(self.satellite_positions - player_position_ecef, axis=1).reshape((-1))

        # Assume open field
        # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        multipath_bias = 0

        # Satelite dependent random noise
        # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        epsilon = self.rng.normal(0.0, self.satellite_noise_std, (self.satellite_amount,))

        # Noise from sources local to the antenna, helps to model interference
        # Extrapolated from GNSS interference mitigation: A measurement and position domain assessment
        jammer = 30 #self.rng.normal(30.0, 0.1)  # dB

        def correction(noiseLevel):
            if noiseLevel <= self.satellite_noise_std:
                return 0
            if self.satellite_noise_std < noiseLevel < self.noise_fix_loss_level:
                return self.rng.normal(0.0, noiseLevel * self.noise_effect_rate)
            if self.noise_fix_loss_level <= noiseLevel:
                print("Too much noise")
                return None

        localNoiseEffect = correction(jammer)

        pseudorange = range + bias_difference + tropospheric_delay + scipy.constants.c * ionospheric_delay + multipath_bias + epsilon + localNoiseEffect

        return pseudorange

    def get_doppler(self, player_position, player_velocity, receiver_clock_drift):
        # Doppler effect simulation
        # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.2
        velocity_difference = player_velocity - self.satellite_velocities
        satellite_user_delta = self.satellite_positions - player_position
        satellite_line_of_sight = satellite_user_delta / np.linalg.norm(satellite_user_delta, axis=1).reshape((-1, 1))
        velocity_scalar_projection = np.sum(velocity_difference * satellite_line_of_sight, axis=1)
        velocity_base = velocity_scalar_projection * (self.satellite_frequency / scipy.constants.c)
        doppler_contribution_clock = receiver_clock_drift * (self.satellite_frequency / scipy.constants.c)
        epsilon = self.rng.normal(0.0, self.satellite_noise_std, (self.satellite_amount,))

        # Satellite clock drift, ionospheric and troposferic effects are negligeble
        # See Global Positioning System_ Signals, Measurements, and Performance section 6.2.1

        direct_doppler = velocity_base + doppler_contribution_clock + epsilon
        return direct_doppler

    def get_dilution_of_presition(self, player_position):
        A = np.concatenate(
            (
                (self.satellite_positions - player_position) / np.linalg.norm(self.satellite_positions - player_position,
                                                                         axis=1).reshape(
                    (-1, 1)),
                np.ones((1, self.satellite_amount)).T),
            axis=1
        )
        eps = np.eye(A.shape[1]) * 1e-15  # Prevent heavily broken satellite configurations from crashing the progarm
        Q = np.linalg.inv(A.T @ A + eps)
        gdop = np.sqrt(np.sum([Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3]]))
        hdop = np.sqrt(np.sum([Q[0, 0], Q[1, 1]]))
        vdop = np.sqrt(Q[2, 2])
        pdop = np.sqrt(np.sum([Q[0, 0], Q[1, 1], Q[2, 2]]))

        return Q, gdop, hdop, vdop, pdop

class GnssSensor:
    def __init__(self, simulator: Simulator, solver: Solver):
        self.simulator = simulator
        self.solver = solver


    def update(self, player_positions, player_velocities, reciever_clock_bias, reciever_clock_drift)\
            -> Tuple[array3d, array3d, np.float64, np.float64]:
        print("===")

        player_position = player_positions[-1]
        pseudoranges = self.simulator.get_pseudoranges(player_position, reciever_clock_bias, 0)

        # Computation of the satellite orbit, from ephimeris
        # Assume satellite position is known because ephimeris is transmitted during the first fix
        # From https://gssc.esa.int/navipedia/index.php/Coordinates_Computation_from_Almanac_Data
        position_aproximation, clock_bias_approximation, gnss_position_error = (
            self.solver.solve_position_scipy(pseudoranges, 1e-9))

        Q, gdop, hdop, vdop, pdop = self.simulator.get_dilution_of_presition(player_position)

        print(f"- Position")
        print(f"Cost {gnss_position_error}, estimated position {position_aproximation}m, estimated reciever clock bias {clock_bias_approximation}s")
        print(f"GDOP {gdop} HDOP {hdop} VDOP {vdop} PDOP {pdop}")
        print(f"Error {np.linalg.norm(position_aproximation-player_position)}m, real position {player_position}m, real reciever clock bias {reciever_clock_bias}s")

        player_velocity = player_velocities[-1]
        direct_doppler = self.simulator.get_doppler(player_position, player_velocity, reciever_clock_drift)

        # Reciever estimation
        # From https://satellite-navigation.springeropen.com/counter/pdf/10.1186/s43020-023-00098-2.pdf
        # Also see Navigation from Low Earth Orbit – Part 2: Models, Implementation, and Performance section 2.2
        velocity_approximation, clock_drift_approximation, gnss_velocity_error = (
            self.solver.solve_velocity(direct_doppler, position_aproximation, 1e-9))

        print(f"- Velocity")
        print(f"Cost {gnss_velocity_error}, estimated velocity {velocity_approximation}m/s, linear velocity {np.linalg.norm(velocity_approximation)}m/s")
        print(f"Estimated receiver clock drift {clock_drift_approximation}s")
        print(f"Error {np.linalg.norm(velocity_approximation-player_velocity)}m/s, real velocity {player_velocity}m/s, linear_velocity {np.linalg.norm(player_velocity)}m/s")

        return position_aproximation, velocity_approximation, clock_bias_approximation, clock_drift_approximation


def main():
    width, height = 800, 450
    init_window(width, height, "Hello")

    print("Satelite positions")
    print(SATELLITE_POSITIONS)
    print("Satelite velocities")
    print(SATELLITE_VELOCITY_VECTORS)
    print("Satelite clock biases")
    print(SATELLITE_CLOCK_BIAS)

    rng = np.random.default_rng()
    simulator = Simulator(rng, SATELLITE_POSITIONS, SATELLITE_CLOCK_BIAS, SATELLITE_VELOCITY_VECTORS,
                          GNSS_SIGNAL_FREQUENCY, SATELLITE_ALPHAS, SATELLITE_BETAS, NOISE_CORRECTION_LEVEL,
                          NOISE_FIX_LOSS_LEVEL, NOISE_EFFECT_RATE, SATELLITE_NOISE_STD)
    solver = Solver(SATELLITE_POSITIONS, SATELLITE_CLOCK_BIAS, SATELLITE_VELOCITY_VECTORS, GNSS_SIGNAL_FREQUENCY)
    sensor = GnssSensor(simulator, solver)

    player_positions: List[array3d] = [np.array([20, 20,  R_earth.value], dtype=np.float64)]
    player_velocities: List[array3d] = [np.array([0, 0, 0], dtype=np.float64)]
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

        player_velocity = MOVEMENT_SPEED_METERS_PER_SECOND * player_delta
        player_velocities.append(player_velocity)

        if time_since_gnss > 1/GNSS_MESSAGE_FREQUENCY:
            # See GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
            if time_since_gnss < np.inf:
                receiver_clock_bias += RECEIVER_CLOCK_WALK * rng.normal() * time_since_gnss
            gnss_position, gnss_velocity, _, _ = sensor.update(player_positions, player_velocities, receiver_clock_bias,
                                                               RECEIVER_CLOCK_WALK)
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