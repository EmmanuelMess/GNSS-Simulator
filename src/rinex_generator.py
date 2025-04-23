import os
from typing import List

import numpy as np
from astropy.time import TimeGPS
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time

from numpy_types import array3d
from src import tle_to_gps


class RinexGenerator:
    """
    Generates the RINEX NAV and RINEX ORB from data provided
    RINEX can only be generated with GPS data
    """

    def __init__(self, folder: os.path, approximate_start_position: array3d, utc_start: Time, gps_start: TimeGPS,
                 satellite_amount: int):
        self.folder = folder

        # TODO make it auto close the files

        # TODO check if the week number starts at 0
        file_name = f"{utc_start.tt_strftime('%Y%m%d%H%M%S')}.{utc_start.tt_strftime('%W')}"

        self.navigation_file = open(os.path.join(folder, f"{file_name}P"), "w") #TODO fix name
        self.observations_file = open(os.path.join(folder, f"{file_name}O"), "w") #TODO fix name

        self._write_header_navigation_file()
        self._write_header_observations_file(approximate_start_position, gps_start, satellite_amount)


    def _write_header_navigation_file(self):
        #TODO add ionospheric and tropospheric corrections


        self.navigation_file.write(f"     3.03           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE\n")
        self.navigation_file.write(f"GnssSim CIFASIS                         20231226 160817 UTC PGM / RUN BY / DATE \n")
        # TODO do we need this? self.navigation_file.write(f"GPUT -2.7939677238E-09-5.329070518E-15 405504 2294          TIME SYSTEM CORR    ")
        # TODO do we need this? self.navigation_file.write(f"    {leap_seconds: <18}                                     LEAP SECONDS        \n")
        self.navigation_file.write(f"                                                            END OF HEADER       \n")

    def _write_header_observations_file(self, approximate_start_position: array3d, time_of_first_observation: TimeGPS,
                                        satellite_amount: int):
        position_x = np.round(approximate_start_position[0], 4)
        position_y = np.round(approximate_start_position[1], 4)
        position_z = np.round(approximate_start_position[2], 4)
        start_timestamp_str = time_of_first_observation.strftime('  %Y   %m   %d   %H   %M   %S.%f')

        self.observations_file.write(f"     3.03           OBSERVATION DATA    G: GPS              RINEX VERSION / TYPE\n")
        self.observations_file.write(f"GnssSim CIFASIS                         20231226 160817 UTC PGM / RUN BY / DATE \n")
        self.observations_file.write(f"                                                            MARKER NAME         \n")
        self.observations_file.write(f"GROUND_CRAFT                                                MARKER TYPE         \n")
        self.observations_file.write(f"CIFASIS                                                     OBSERVER / AGENCY   \n")
        self.observations_file.write(f"                    GnssSim Simulator 1.0.0                 REC # / TYPE / VERS \n")
        self.observations_file.write(f"                                                            ANT # / TYPE        \n") # TODO add info
        self.observations_file.write(f"  {position_x: <17} {position_y: <17} {position_z: <17}     APPROX POSITION XYZ \n")
        self.observations_file.write(f"        0.0000        0.0000        0.0000                  ANTENNA: DELTA H/E/N\n") # TODO know what these are
        self.observations_file.write(f"G    2 C1C  D1C                                             SYS / # / OBS TYPES \n") # TODO check that it is C code-based
        self.observations_file.write(f" {start_timestamp_str: <26}         GPS         TIME OF FIRST OBS   \n")
        self.observations_file.write(f"G                                                           SYS / PHASE SHIFT   \n")
        self.observations_file.write(f"  0                                                         GLONASS SLOT / FRQ #\n")
        self.observations_file.write(f"                                                            GLONASS COD/PHS/BIS \n") # TODO check this is valid
        self.observations_file.write(f" {satellite_amount: <23}                                    # OF SATELLITES     \n")
        self.observations_file.write(f"                                                            END OF HEADER       \n")

    def _add_satellite(self, satellite: EarthSatellite, satellite_clock_bias: np.float64):
        def format_rinex_float(value: np.float64) -> str:
            """
            This formats the value as 4X, 4D19.12 format. If the value is not finite, the result is malformed.
            """
            if not np.isfinite(value):
                return f"{value}"

            sign = '-' if value < 0 else ' '
            abs_value = np.abs(value)
            exponent = 1 + np.int32(np.floor(np.log10(abs_value))).item() if value != 0 else 0
            mantissa = abs_value / (10 ** exponent)
            mantissa_str = f'{mantissa:.12f}'.replace("0.", ".")
            return f'{sign}{mantissa_str}E{exponent:+03d}'

        gps_parameters = tle_to_gps.convert_tle_to_gps_parameters(satellite, satellite_clock_bias)

        satellite_system = gps_parameters.satellite_system
        prn = f"{gps_parameters.prn_number: <2}"
        epoch = gps_parameters.epoch_gps_time.strftime("%Y %m %d %H %M %S")
        sv_clock_bias = format_rinex_float(gps_parameters.sv_clock_bias)
        sv_clock_drift = format_rinex_float(gps_parameters.sv_clock_drift)
        sv_clock_drift_rate = format_rinex_float(gps_parameters.sv_clock_drift_rate)
        iode = format_rinex_float(gps_parameters.issue_of_data_ephemeris)
        crs = format_rinex_float(gps_parameters.amplitude_sine_harmonic_correction_term_to_orbit_radius)
        delta_n = format_rinex_float(gps_parameters.mean_motion_difference_from_computed_value)
        m0 = format_rinex_float(gps_parameters.mean_anomaly_at_reference_time)
        cuc = format_rinex_float(gps_parameters.amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude)
        e = format_rinex_float(gps_parameters.eccentricity)
        cus = format_rinex_float(gps_parameters.amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude)
        sqrt_a = format_rinex_float(gps_parameters.square_root_of_semi_major_axis)
        toe = format_rinex_float(gps_parameters.time_of_ephemeris)
        cic = format_rinex_float(gps_parameters.amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination)
        omega0 = format_rinex_float(gps_parameters.longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch)
        cis = format_rinex_float(gps_parameters.amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination)
        i0 = format_rinex_float(gps_parameters.inclination_angle_at_reference_time)
        crc = format_rinex_float(gps_parameters.amplitude_of_cosine_harmonic_correction_term_to_orbit_radius)
        omega = format_rinex_float(gps_parameters.argument_of_perigee)
        omega_dot = format_rinex_float(gps_parameters.rate_of_right_ascension)
        idot = format_rinex_float(gps_parameters.rate_of_inclination_angle)
        codes_on_l2 = format_rinex_float(gps_parameters.codes_on_l2_channel)
        gps_week_number = format_rinex_float(gps_parameters.gps_week_number)
        l2_p_flag = format_rinex_float(gps_parameters.l2_p_data_flag)
        sv_accuracy = format_rinex_float(gps_parameters.sv_accuracy)
        sv_health = format_rinex_float(gps_parameters.sv_health)
        tgd = format_rinex_float(gps_parameters.tgd_total_group_delay)
        iodc = format_rinex_float(gps_parameters.iodc_issue_of_data_clock)
        transmission_time = format_rinex_float(gps_parameters.transmission_time_of_message)
        fit_interval = format_rinex_float(gps_parameters.fit_interval_in_hours)

        self.navigation_file.write(f"{satellite_system}{prn} {epoch} {sv_clock_bias} {sv_clock_drift} {sv_clock_drift_rate}\n")
        self.navigation_file.write(f"     {iode} {crs} {delta_n} {m0}\n")
        self.navigation_file.write(f"     {cuc} {e} {cus} {sqrt_a}\n")
        self.navigation_file.write(f"     {toe} {cic} {omega0} {cis}\n")
        self.navigation_file.write(f"     {i0} {crc} {omega} {omega_dot}\n")
        self.navigation_file.write(f"     {idot} {codes_on_l2} {gps_week_number} {l2_p_flag}\n")
        self.navigation_file.write(f"     {sv_accuracy} {sv_health} {tgd} {iodc}\n")
        self.navigation_file.write(f"     {transmission_time} {fit_interval}\n")

    def add_satellites(self, satellites: List[EarthSatellite], satellite_clock_biases: List[np.float64]):
        if len(satellites) != len(satellite_clock_biases):
            return

        for satellite, satellite_clock_bias in zip(satellites, satellite_clock_biases):
            self._add_satellite(satellite, satellite_clock_bias)

    def add_position(self, time_gps: TimeGPS, prns: List[int], pseudoranges: List[np.float64],
                     direct_doppler: List[np.float64]):
        observed_satellites = len(pseudoranges)
        time = time_gps.strftime("%Y %m %d %H %M %S.%f")

        self.observations_file.write(f"> {time:<27} 0 {observed_satellites:<2}                     \n") # TODO check if this should be the UTC clock or the GPS clock

        for prn, pseudorange, doppler in zip(prns, pseudoranges, direct_doppler):
            self.observations_file.write(f"G{prn:<2}   {pseudorange:.3f}   {doppler:.3f}                    \n")
