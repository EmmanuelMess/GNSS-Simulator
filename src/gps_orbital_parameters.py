import re
from dataclasses import dataclass

import numpy as np
from hifitime import Epoch, TimeScale

from src.conversions import SECONDS_IN_WEEK


@dataclass
class GpsOrbitalParameters:
    satellite_system: str
    prn_number: int
    epoch: Epoch
    time_of_ephemeris: Epoch
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
    time_of_ephemeris_seconds_of_week: np.float64
    amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination: np.float64
    longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch: np.float64
    amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination: np.float64
    inclination_angle_at_reference_time: np.float64
    amplitude_of_cosine_harmonic_correction_term_to_orbit_radius: np.float64
    argument_of_perigee: np.float64
    rate_of_right_ascension: np.float64
    rate_of_inclination_angle: np.float64
    codes_on_l2_channel: np.float64
    time_of_ephemeris_week_number: np.float64
    l2_p_data_flag: np.float64
    sv_accuracy: np.float64
    sv_health: np.float64
    tgd_total_group_delay: np.float64
    iodc_issue_of_data_clock: np.float64
    transmission_time_of_message: np.float64
    fit_interval_in_hours: np.float64

RINEX_START = "(?P<system>[A-Z])(?P<sv>[ \d]{2}) (?P<year>\d{4}) (?P<month>\d{2}) (?P<day>\d{2}) (?P<hour>\d{2}) (?P<minute>\d{2}) (?P<second>\d{2})"
RINEX_DOUBLE_REGEX = "[ \d.\-\+EeDd]{19}"
RINEX_LINE_JUMP = "\n *"
RINEX_REGEX = re.compile(f"^{RINEX_START}(?P<clock_bias>{RINEX_DOUBLE_REGEX})(?P<clock_drift>{RINEX_DOUBLE_REGEX})(?P<clock_drift_rate>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<iode>{RINEX_DOUBLE_REGEX})(?P<crs>{RINEX_DOUBLE_REGEX})(?P<delta_n>{RINEX_DOUBLE_REGEX})(?P<m0>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<cuc>{RINEX_DOUBLE_REGEX})(?P<e>{RINEX_DOUBLE_REGEX})(?P<cus>{RINEX_DOUBLE_REGEX})(?P<sqrt_a>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<toe>{RINEX_DOUBLE_REGEX})(?P<cic>{RINEX_DOUBLE_REGEX})(?P<omega0>{RINEX_DOUBLE_REGEX})(?P<cis>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<i0>{RINEX_DOUBLE_REGEX})(?P<crc>{RINEX_DOUBLE_REGEX})(?P<omega>{RINEX_DOUBLE_REGEX})(?P<omega_dot>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<idot>{RINEX_DOUBLE_REGEX})(?P<codes_l2>{RINEX_DOUBLE_REGEX})(?P<gps_week>{RINEX_DOUBLE_REGEX})(?P<l2_data>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<accuracy>{RINEX_DOUBLE_REGEX})(?P<health>{RINEX_DOUBLE_REGEX})(?P<tgd>{RINEX_DOUBLE_REGEX})(?P<iodc>{RINEX_DOUBLE_REGEX}){RINEX_LINE_JUMP}(?P<transmission_time>{RINEX_DOUBLE_REGEX})(?P<fit_interval>{RINEX_DOUBLE_REGEX})", re.MULTILINE)

def from_rinex(string: str) -> GpsOrbitalParameters:
    """
    Take a rinex orbit string and parse it.
    WARNING: this function is not a full parser, and has not been extensively tested. User is expected to ensure that
    the parameters were read correctly.
    """
    def convert_float(text: str) -> np.float64:
        text = text.replace("D", "E")
        text = text.replace("d", "E")

        return np.float64(text)

    matches = re.match(RINEX_REGEX, string)

    epoch_year = np.uint64(matches.group("year"))
    epoch_month = np.uint64(matches.group("month"))
    epoch_day = np.uint64(matches.group("day"))
    epoch_hour = np.uint64(matches.group("hour"))
    epoch_minute = np.uint64(matches.group("minute"))
    epoch_second = np.uint64(matches.group("second"))

    epoch = Epoch.init_from_gregorian(epoch_year, epoch_month, epoch_day, epoch_hour, epoch_minute, epoch_second, 0, TimeScale.GPST)

    week_of_year = convert_float(matches.group("gps_week"))
    toe = convert_float(matches.group("toe"))

    time_of_ephemeris = Epoch.init_from_gpst_seconds(week_of_year * SECONDS_IN_WEEK + toe)

    parameters = GpsOrbitalParameters(
        satellite_system = matches.group("system"),
        prn_number = np.int64(matches.group("sv")).item(),
        epoch=epoch,
        time_of_ephemeris=time_of_ephemeris,
        sv_clock_bias = convert_float(matches.group("clock_bias")),
        sv_clock_drift = convert_float(matches.group("clock_drift")),
        sv_clock_drift_rate = convert_float(matches.group("clock_drift_rate")),
        issue_of_data_ephemeris = convert_float(matches.group("iode")),
        amplitude_sine_harmonic_correction_term_to_orbit_radius = convert_float(matches.group("crs")),
        mean_motion_difference_from_computed_value = convert_float(matches.group("delta_n")),
        mean_anomaly_at_reference_time = convert_float(matches.group("m0")),
        amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude = convert_float(matches.group("cuc")),
        eccentricity = convert_float(matches.group("e")),
        amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude = convert_float(matches.group("cus")),
        square_root_of_semi_major_axis = convert_float(matches.group("sqrt_a")),
        time_of_ephemeris_seconds_of_week= toe,
        amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination = convert_float(matches.group("cic")),
        longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch = convert_float(matches.group("omega0")),
        amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination = convert_float(matches.group("cis")),
        inclination_angle_at_reference_time = convert_float(matches.group("i0")),
        amplitude_of_cosine_harmonic_correction_term_to_orbit_radius = convert_float(matches.group("crc")),
        argument_of_perigee = convert_float(matches.group("omega")),
        rate_of_right_ascension = convert_float(matches.group("omega_dot")),
        rate_of_inclination_angle = convert_float(matches.group("idot")),
        codes_on_l2_channel = convert_float(matches.group("codes_l2")),
        time_of_ephemeris_week_number= week_of_year,
        l2_p_data_flag = convert_float(matches.group("l2_data")),
        sv_accuracy = convert_float(matches.group("accuracy")),
        sv_health = convert_float(matches.group("health")),
        tgd_total_group_delay = convert_float(matches.group("tgd")),
        iodc_issue_of_data_clock = convert_float(matches.group("iodc")),
        transmission_time_of_message = convert_float(matches.group("transmission_time")),
        fit_interval_in_hours   = convert_float(matches.group("fit_interval")), # TODO this is incorrect, should be 1 for 4hs 2 for 6hs
    )

    return parameters
