from dataclasses import dataclass

import numpy as np
from astropy.time import Time
from skyfield.sgp4lib import EarthSatellite

from prn_from_name import get_prn
from conversions import time_gps2seconds_of_week, time_gps2week_number, rev_per_day2rad_per_second
from constants import MU_EARTH


@dataclass
class GpsOrbitalParameters:
    satellite_system: str
    prn_number: int
    epoch_gps_time: Time
    sv_clock_bias: np.float64
    sv_clock_drift: np.float64
    sv_clock_drift_rate: np.float64
    issue_of_data_ephemeris: np.float64
    amplitude_sine_harmonic_correction_term_to_orbit_radius: np.float64
    mean_motion_difference_from_computed_value: np.float64
    mean_anomaly_at_reference_time: np.float64
    amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude: np.float64
    eccentricity: np.float64
    amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude: np.float64
    square_root_of_semi_major_axis: np.float64
    time_of_ephemeris: np.float64
    amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination: np.float64
    longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch: np.float64
    amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination: np.float64
    inclination_angle_at_reference_time: np.float64
    amplitude_of_cosine_harmonic_correction_term_to_orbit_radius: np.float64
    argument_of_perigee: np.float64
    rate_of_right_ascension: np.float64
    rate_of_inclination_angle: np.float64
    codes_on_l2_channel: np.float64
    gps_week_number: np.float64
    l2_p_data_flag: np.float64
    sv_accuracy: np.float64
    sv_health: np.float64
    tgd_total_group_delay: np.float64
    iodc_issue_of_data_clock: np.float64
    transmission_time_of_message: np.float64
    fit_interval_in_hours: np.float64


def convert_tle_to_gps_parameters(satellite: EarthSatellite, satellite_clock_bias: np.float64):
    tle = satellite.model

    gps_time = satellite.epoch.to_astropy()
    gps_time.format = "gps"

    gps_week_time = time_gps2seconds_of_week(gps_time.value)
    gps_week_number = time_gps2week_number(gps_time.value)

    # The equation implemented for sqrt_semi_major_axis is this but better numerically:
    # semi_major_axis = MU_EARTH ** (1/3) / (rev_per_day2rad_per_second(tle.no_kozai) ** (2/3))
    # sqrt_semi_major_axis = np.sqrt(semi_major_axis)
    mean_motion = rev_per_day2rad_per_second(np.float64(tle.no_kozai))
    sqrt_semi_major_axis = np.pow(np.sqrt(MU_EARTH) / mean_motion, 1/3)

    # In a few places a trick is used because the TLE assumes that the parameters stay constant, so some parameters
    # that are variable in time from GPS orbital parameters are assumed constant
    return GpsOrbitalParameters(
        satellite_system="G",
        prn_number=get_prn(satellite.name),
        epoch_gps_time=gps_time,
        sv_clock_bias=satellite_clock_bias,
        sv_clock_drift=np.float64(0),
        sv_clock_drift_rate=np.float64(0),
        issue_of_data_ephemeris=np.float64(0),
        amplitude_sine_harmonic_correction_term_to_orbit_radius=np.float64(0),
        mean_motion_difference_from_computed_value=np.float64(0),
        mean_anomaly_at_reference_time=tle.mo,
        amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude=np.float64(0),
        eccentricity=tle.ecco,
        amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude=np.float64(0),
        square_root_of_semi_major_axis=sqrt_semi_major_axis,
        time_of_ephemeris=np.float64(gps_week_time),
        amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination=np.float64(0),
        longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch=tle.nodeo,
        amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination=np.float64(0),
        inclination_angle_at_reference_time=tle.inclo,
        amplitude_of_cosine_harmonic_correction_term_to_orbit_radius=np.float64(0),
        argument_of_perigee=tle.argpo,
        rate_of_right_ascension=np.float64(0),
        rate_of_inclination_angle=np.float64(0),
        codes_on_l2_channel=np.float64(0),
        gps_week_number=np.float64(gps_week_number),
        l2_p_data_flag=np.float64(0),
        sv_accuracy=np.float64(0),
        sv_health=np.float64(0),
        tgd_total_group_delay=np.float64(0),
        iodc_issue_of_data_clock=np.float64(0),
        transmission_time_of_message=np.float64(gps_week_time),
        fit_interval_in_hours=np.float64(4),
    )