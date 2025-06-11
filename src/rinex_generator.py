import os
from typing import List

import numpy as np
from hifitime import Epoch, Unit

from src.conversions import DAYS_IN_WEEK
from src.numpy_types import array3d
from src.gps_orbital_parameters import GpsOrbitalParameters
from src.gps_satellite import GpsSatellite


class RinexGenerator:
    """
    Generates the RINEX NAV and RINEX ORB from data provided
    RINEX can only be generated with GPS data
    """

    def __init__(self, folder: os.path, approximate_start_position: array3d, utc_start: Epoch, gps_start: Epoch,
                 satellite_alphas: np.ndarray[4, np.float64], satellite_betas: np.ndarray[4, np.float64]):
        self.folder = folder

        # TODO make it auto close the files

        # TODO check if the week number starts at 0
        file_name = f"{utc_start.strftime('%Y%m%d%H%M%S')}.{int(utc_start.day_of_year() // DAYS_IN_WEEK)}"

        self.navigation_file = open(os.path.join(folder, f"{file_name}P"), "w") #TODO fix name
        self.observations_file = open(os.path.join(folder, f"{file_name}O"), "w") #TODO fix name

        self._write_header_navigation_file(utc_start, satellite_alphas, satellite_betas)
        self._write_header_observations_file(approximate_start_position, utc_start, gps_start)

    def _write_header_navigation_file(self, utc_start: Epoch, satellite_alphas: np.ndarray[4, np.float64],
                                      satellite_betas: np.ndarray[4, np.float64]):
        def format(value: np.float64) -> str:
            return f"{value:+1.4E}"

        time = utc_start.strftime('%Y%m%d %H%M%S')

        alpha0, alpha1, alpha2, alpha3 = (
            format(satellite_alphas[0]), format(satellite_alphas[1]), format(satellite_alphas[2]), format(satellite_alphas[3])
        )
        beta0, beta1, beta2, beta3 = (
            format(satellite_betas[0]), format(satellite_betas[1]), format(satellite_betas[2]), format(satellite_betas[3])
        )

        self.navigation_file.write(f"     3.03           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE\n")
        self.navigation_file.write(f"GnssSim CIFASIS                         {time: >15} UTC PGM / RUN BY / DATE \n")
        self.navigation_file.write(f"GPSA {alpha0: >12}{alpha1: >12}{alpha2: >12}{alpha3: >12}       IONOSPHERIC CORR    \n")
        self.navigation_file.write(f"GPSB {beta0: >12}{beta1: >12}{beta2: >12}{beta3: >12}       IONOSPHERIC CORR    \n")
        # TODO do we need this? self.navigation_file.write(f"GPUT -2.7939677238E-09-5.329070518E-15 405504 2294          TIME SYSTEM CORR    ")
        # TODO do we need this? self.navigation_file.write(f"    {leap_seconds: <18}                                     LEAP SECONDS        \n")
        self.navigation_file.write(f"                                                            END OF HEADER       \n")

    def _write_header_observations_file(self, approximate_start_position: array3d, time_utc: Epoch,
                                        time_of_first_observation: Epoch):
        position_x = np.round(approximate_start_position[0], 4)
        position_y = np.round(approximate_start_position[1], 4)
        position_z = np.round(approximate_start_position[2], 4)

        time_utc_str = f"{time_utc.strftime('%Y%m%d')} {time_utc.strftime('%H%M%S')} UTC"

        reformatted_seconds = f"{np.float64(time_of_first_observation.strftime('%S.%f')): 2.7f}"
        start_timestamp_str = f" {time_of_first_observation.strftime('%Y    %m    %d    %H    %M')}   {reformatted_seconds:>10}"

        self.observations_file.write(f"     3.03           OBSERVATION DATA    G: GPS              RINEX VERSION / TYPE\n")
        self.observations_file.write(f"GnssSim             CIFASIS             {time_utc_str: >19} PGM / RUN BY / DATE \n")
        self.observations_file.write(f"                                                            MARKER NAME         \n")
        self.observations_file.write(f"GROUND_CRAFT                                                MARKER TYPE         \n")
        self.observations_file.write(f"CIFASIS                                                     OBSERVER / AGENCY   \n") # TODO move agency to correct column, add observer
        self.observations_file.write(f"                    GnssSim Simulator                       REC # / TYPE / VERS \n")
        self.observations_file.write(f"                                                            ANT # / TYPE        \n") # TODO add info
        self.observations_file.write(f" {position_x: >13} {position_y: >13} {position_z: >13}                  APPROX POSITION XYZ \n")
        self.observations_file.write(f"        0.0000        0.0000        0.0000                  ANTENNA: DELTA H/E/N\n") # TODO know what these are
        self.observations_file.write(f"G    3 C1C D1C S1C                                          SYS / # / OBS TYPES \n")
        self.observations_file.write(f" {start_timestamp_str: <26}            GPS         TIME OF FIRST OBS   \n")
        self.observations_file.write(f"G                                                           SYS / PHASE SHIFT   \n")
        self.observations_file.write(f"  0                                                         GLONASS SLOT / FRQ #\n")
        self.observations_file.write(f"                                                            GLONASS COD/PHS/BIS \n") # TODO check this is valid
        self.observations_file.write(f"                                                            END OF HEADER       \n")

    @staticmethod
    def _satellite_rinex(gps_parameters: GpsOrbitalParameters):
        """
        Separated into a function mostly for testing
        """
        def format(value: np.float64) -> str:
            return f"{value:+1.12E}"

        satellite_system = gps_parameters.satellite_system
        prn = gps_parameters.prn_number
        epoch = gps_parameters.epoch.strftime("%Y %m %d %H %M %S")
        sv_clock_bias = format(gps_parameters.sv_clock_bias)
        sv_clock_drift = format(gps_parameters.sv_clock_drift)
        sv_clock_drift_rate = format(gps_parameters.sv_clock_drift_rate)
        iode = format(gps_parameters.issue_of_data_ephemeris)
        crs = format(gps_parameters.amplitude_sine_harmonic_correction_term_to_orbit_radius)
        delta_n = format(gps_parameters.mean_motion_difference_from_computed_value)
        m0 = format(gps_parameters.mean_anomaly_at_reference_time)
        cuc = format(gps_parameters.amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude)
        e = format(gps_parameters.eccentricity)
        cus = format(gps_parameters.amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude)
        sqrt_a = format(gps_parameters.square_root_of_semi_major_axis)
        toe = format(gps_parameters.time_of_ephemeris_seconds_of_week)
        cic = format(gps_parameters.amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination)
        omega0 = format(gps_parameters.longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch)
        cis = format(gps_parameters.amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination)
        i0 = format(gps_parameters.inclination_angle_at_reference_time)
        crc = format(gps_parameters.amplitude_of_cosine_harmonic_correction_term_to_orbit_radius)
        omega = format(gps_parameters.argument_of_perigee)
        omega_dot = format(gps_parameters.rate_of_right_ascension)
        idot = format(gps_parameters.rate_of_inclination_angle)
        codes_on_l2 = format(gps_parameters.codes_on_l2_channel)
        gps_week_number = format(gps_parameters.time_of_ephemeris_week_number)
        l2_p_flag = format(gps_parameters.l2_p_data_flag)
        sv_accuracy = format(gps_parameters.sv_accuracy)
        sv_health = format(gps_parameters.sv_health)
        tgd = format(gps_parameters.tgd_total_group_delay)
        iodc = format(gps_parameters.iodc_issue_of_data_clock)
        transmission_time = format(gps_parameters.transmission_time_of_message)
        fit_interval = format(gps_parameters.fit_interval_in_hours)

        return (
            f"{satellite_system}{prn:>2} {epoch}{sv_clock_bias}{sv_clock_drift}{sv_clock_drift_rate}\n"
            f"    {iode}{crs}{delta_n}{m0}\n"
            f"    {cuc}{e}{cus}{sqrt_a}\n"
            f"    {toe}{cic}{omega0}{cis}\n"
            f"    {i0}{crc}{omega}{omega_dot}\n"
            f"    {idot}{codes_on_l2}{gps_week_number}{l2_p_flag}\n"
            f"    {sv_accuracy}{sv_health}{tgd}{iodc}\n"
            f"    {transmission_time}{fit_interval}\n"
        )

    def add_satellites(self, satellites: List[GpsSatellite]):
        for satellite in satellites:
            rinex = self._satellite_rinex(satellite.parameters())
            self.navigation_file.write(rinex)

    def add_position(self, time_gps: Epoch, prns: List[int], pseudoranges: List[np.float64],
                     direct_doppler: List[np.float64]):
        observed_satellites = len(pseudoranges)

        reformatted_seconds = np.float64(time_gps.strftime("%S.%f"))
        time = f"{time_gps.strftime('%Y %m %d %H %M')} {reformatted_seconds:2.7f}"

        self.observations_file.write(f"> {time:<27} 0 {observed_satellites:<2}                     \n")

        for prn, pseudorange, doppler in zip(prns, pseudoranges, direct_doppler):
            pseudorange_str = f"{pseudorange:.3f}"
            doppler_str = f"{doppler:.3f}"
            signal_strength_str = f"{54.0: 2.3f}" # in dbHz see 5.7 of RINEX v3.03 # TODO use actual noise data here
            # TODO add strength for each field in ratio with actual data
            self.observations_file.write(f"G{prn:>2} {pseudorange_str:>13}  {doppler_str:>13}  {signal_strength_str:>13}\n")
