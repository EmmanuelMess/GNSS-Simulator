import re
from dataclasses import dataclass
import astropy.units as u
from astropy.time import TimeDelta

import numpy as np
from astropy.time import Time


@dataclass
class GpsOrbitalParameters:
    satellite_system: str
    prn_number: int
    epoch: Time
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


RINEX_REGEX = re.compile("^(?P<system>[A-Z])(?P<sv>\d{2}) (?P<year>\d{4}) (?P<month>\d{2}) (?P<day>\d{2}) (?P<hour>\d{2}) (?P<minute>\d{2}) (?P<second>\d{2})( ?(?P<clock_bias>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<clock_drift>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<clock_drift_rate>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<iode>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<crs>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<delta_n>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<m0>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<cuc>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<e>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<cus>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<sqrt_a>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<toe>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<cic>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<omega0>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<cis>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<i0>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<crc>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<omega>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<omega_dot>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<idot>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<codes_l2>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<gps_week>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<l2_data>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<accuracy>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<health>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<tgd>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<iodc>-?\+?\d.\d{12}E-?\+?\d{2}))([\n ]*)( ?(?P<transmission_time>-?\+?\d.\d{12}E-?\+?\d{2}))( ?(?P<fit_interval>-?\+?\d.\d{12}E-?\+?\d{2}))", re.MULTILINE)

def from_rinex(string: str) -> GpsOrbitalParameters:
    """
    Take a rinex orbit string and parse it.
    WARNING: this function is not a full parser, and has not been extensively tested. User is expected to ensure that
    the parameters were read correctly.
    """

    matches = re.match(RINEX_REGEX, string)

    week_of_year = np.float64(matches.group("gps_week"))
    toe = np.float64(matches.group("toe"))

    epoch = Time(week_of_year * u.week + toe * u.s, format='gps')

    parameters = GpsOrbitalParameters(
        satellite_system = matches.group("system"),
        prn_number = np.int64(matches.group("sv")).item(),
        epoch=epoch,
        sv_clock_bias = np.float64(matches.group("clock_bias")),
        sv_clock_drift = np.float64(matches.group("clock_drift")),
        sv_clock_drift_rate = np.float64(matches.group("clock_drift_rate")),
        issue_of_data_ephemeris = np.float64(matches.group("iode")),
        amplitude_sine_harmonic_correction_term_to_orbit_radius = np.float64(matches.group("crs")),
        mean_motion_difference_from_computed_value = np.float64(matches.group("delta_n")),
        mean_anomaly_at_reference_time = np.float64(matches.group("m0")),
        amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude = np.float64(matches.group("cuc")),
        eccentricity = np.float64(matches.group("e")),
        amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude = np.float64(matches.group("cus")),
        square_root_of_semi_major_axis = np.float64(matches.group("sqrt_a")),
        time_of_ephemeris = toe,
        amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination = np.float64(matches.group("cic")),
        longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch = np.float64(matches.group("omega0")),
        amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination = np.float64(matches.group("cis")),
        inclination_angle_at_reference_time = np.float64(matches.group("i0")),
        amplitude_of_cosine_harmonic_correction_term_to_orbit_radius = np.float64(matches.group("crc")),
        argument_of_perigee = np.float64(matches.group("omega")),
        rate_of_right_ascension = np.float64(matches.group("omega_dot")),
        rate_of_inclination_angle = np.float64(matches.group("idot")),
        codes_on_l2_channel = np.float64(matches.group("codes_l2")),
        gps_week_number = week_of_year,
        l2_p_data_flag = np.float64(matches.group("l2_data")),
        sv_accuracy = np.float64(matches.group("accuracy")),
        sv_health = np.float64(matches.group("health")),
        tgd_total_group_delay = np.float64(matches.group("tgd")),
        iodc_issue_of_data_clock = np.float64(matches.group("iodc")),
        transmission_time_of_message = np.float64(matches.group("transmission_time")),
        fit_interval_in_hours   = np.float64(matches.group("fit_interval")),
    )

    return parameters
