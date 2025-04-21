import numpy as np
import scipy

from conversions import ecef2llh, rad2semicircles, ecef2aer, semicircles2rad, seconds2day_of_year
from constants import GPS_L1_FREQUENCY


class Simulator:
    def __init__(self, rng, satellite_amount, satellite_clock_bias, satellite_frequency, satellite_alphas, satellite_betas,
                 noise_correction_level, noise_fix_loss_level, noise_effect_rate, satellite_noise_std,
                 tropospheric_cutoff_angle):
        self.rng = rng
        self.satellite_amount = satellite_amount
        self.satellite_clock_bias = satellite_clock_bias
        self.satellite_frequency = satellite_frequency # TODO make a vector
        self.satellite_alphas = satellite_alphas
        self.satellite_betas = satellite_betas
        self.satellite_noise_std = satellite_noise_std
        self.noise_correction_level = noise_correction_level
        self.noise_fix_loss_level = noise_fix_loss_level
        self.noise_effect_rate = noise_effect_rate
        self.tropospheric_cutoff_angle = tropospheric_cutoff_angle

    def _ionospheric_delay_calculation(self, satellite_positions_ecef, receiver_ecef, time_of_week_gps_seconds):
        # From https://gssc.esa.int/navipedia/index.php/Klobuchar_Ionospheric_Model
        # And GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        # Reference implementation https://geodesy.noaa.gov/gps-toolbox/ovstedal/klobuchar.for

        receiver_llh = ecef2llh(receiver_ecef)
        satellites_aer_rad = np.array([ecef2aer(receiver_ecef, satellite_position) for satellite_position in satellite_positions_ecef])

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

    def _per_satelite_tropospheric_delay(self, position_llh, satellite_elevation, day_of_year):
        # UNB4 model
        # See https://gssc.esa.int/navipedia/index.php/Galileo_Tropospheric_Correction_Model
        # And Assessment and Development of a Tropospheric Delay Model for Aircraft Users of the Global Positioning System
        # And GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1

        if satellite_elevation <= self.tropospheric_cutoff_angle:
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

    def _tropospheric_delay_calculation(self, satellite_positions_ecef, position_ecef, day_of_year):
        # Troposferic delay is divided intro dry and wet and varies acording to satellite elevation (Saastamoinen model)
        # And Global Positioning System: Signals, Measurements, and Performance section 5.3.3

        position_llh = ecef2llh(position_ecef)
        satellites_aer = np.array([ecef2aer(position_ecef, satellite_position) for satellite_position in satellite_positions_ecef])

        tropospheric_delay = np.array([self._per_satelite_tropospheric_delay(position_llh, elevation, day_of_year) for elevation in satellites_aer[:, 1]])
        return tropospheric_delay


    def _saastamoinen_model(self, position_ecef, pressure, temperature, partial_pressure_water_vapor):
        # From Global Positioning System: Signals, Measurements, and Performance section 5.3.3
        position_llh = ecef2llh(position_ecef)

        dry_delay = 0.002277 * (1+0.0026 * np.cos(2 * position_llh[0]) + 0.00028 * position_llh[2] * 1e-3) * pressure
        wet_delay = 0.002277 * (1255 / temperature + 0.05) * partial_pressure_water_vapor

        return dry_delay, wet_delay


    def get_pseudoranges(self, satellite_positions_ecef, player_position_ecef, reciever_clock_bias,
                         time_gps):
        ionospheric_delay = self._ionospheric_delay_calculation(satellite_positions_ecef, player_position_ecef, time_gps)

        day_of_year = seconds2day_of_year(time_gps)
        tropospheric_delay = self._tropospheric_delay_calculation(satellite_positions_ecef, player_position_ecef, day_of_year)

        # See GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        bias_difference = scipy.constants.c * (reciever_clock_bias - self.satellite_clock_bias.reshape((-1)))
        range = np.linalg.norm(satellite_positions_ecef - player_position_ecef, axis=1).reshape((-1))

        # Assume open field
        # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        multipath_bias = 0

        # Satelite dependent random noise
        # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.1
        epsilon = self.rng.normal(0.0, self.satellite_noise_std, (self.satellite_amount,))

        # Noise from sources local to the antenna, helps to model interference
        # Extrapolated from GNSS interference mitigation: A measurement and position domain assessment
        jammer = self.rng.normal(30.0, 0.1)  # dB

        def correction(noiseLevel):
            if noiseLevel <= self.noise_correction_level:
                return 0
            if self.noise_correction_level < noiseLevel < self.noise_fix_loss_level:
                return noiseLevel * self.noise_effect_rate
            if self.noise_fix_loss_level <= noiseLevel:
                print("Too much noise")
                return None

        localNoiseEffect = correction(jammer)

        pseudorange = range + bias_difference + tropospheric_delay + scipy.constants.c * ionospheric_delay + multipath_bias + epsilon + localNoiseEffect

        return pseudorange

    def get_doppler(self, satellite_positions_ecef, satellite_velocities_ecef, player_position, player_velocity,
                    receiver_clock_drift):
        # Doppler effect simulation
        # From GNSS Applications and Methods (GNSS Technology and Applications) section 3.3.1.2
        velocity_difference = player_velocity - satellite_velocities_ecef
        satellite_user_delta = satellite_positions_ecef - player_position
        satellite_line_of_sight = satellite_user_delta / np.linalg.norm(satellite_user_delta, axis=1).reshape((-1, 1))
        velocity_scalar_projection = np.sum(velocity_difference * satellite_line_of_sight, axis=1)
        velocity_base = velocity_scalar_projection * (self.satellite_frequency / scipy.constants.c)
        doppler_contribution_clock = receiver_clock_drift * (self.satellite_frequency / scipy.constants.c)
        epsilon = self.rng.normal(0.0, self.satellite_noise_std, (self.satellite_amount,))

        # Satellite clock drift, ionospheric and troposferic effects are negligeble
        # See Global Positioning System_ Signals, Measurements, and Performance section 6.2.1

        direct_doppler = velocity_base + doppler_contribution_clock + epsilon
        return direct_doppler

    def get_dilution_of_presition(self, satellite_positions_ecef,  player_position):
        A = np.concatenate(
            (
                (satellite_positions_ecef - player_position) / np.linalg.norm(satellite_positions_ecef - player_position,
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