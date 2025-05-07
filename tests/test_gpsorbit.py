import unittest

import numpy as np
from astropy.time import Time, TimeGPS

from src import gps_orbital_parameters
from src.gps_orbital_parameters import GpsOrbitalParameters
from src.gps_satellite import GpsSatellite


class GpsOrbitTest(unittest.TestCase):
    def test_crude(self):
        inclination: np.float64 = np.float64(0.958_138_635_455)
        longitude_of_ascending_node: np.float64 = np.float64(0.635_682_074_832)
        eccentricity: np.float64 = np.float64(0.003_710_3)
        argument_of_perigee: np.float64 = np.float64(4.057_441_961_27)
        mean_anomaly: np.float64 = np.float64(2.367_757_296_49)
        sqrt_semi_major_axis: np.float64 = np.float64(6_499.315_173_94)

        epoch = Time("2025-04-19 09:46:50.000", format="iso")
        epoch.format = "gps"

        satellite_parameters = GpsOrbitalParameters(
            satellite_system="G",
            prn_number=0,
            epoch=epoch,
            sv_clock_bias=np.float64(0.0),
            sv_clock_drift=np.float64(0.0),
            sv_clock_drift_rate=np.float64(0.0),
            issue_of_data_ephemeris=np.float64(0.0),
            amplitude_sine_harmonic_correction_term_to_orbit_radius=np.float64(0.0),
            mean_motion_difference_from_computed_value=np.float64(0.0),
            mean_anomaly_at_reference_time=mean_anomaly,
            amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude=np.float64(0.0),
            eccentricity=eccentricity,
            amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude=np.float64(0.0),
            square_root_of_semi_major_axis=sqrt_semi_major_axis,
            time_of_ephemeris=np.float64(0.0),
            amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination=np.float64(0.0),
            longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch=longitude_of_ascending_node,
            amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination=np.float64(0.0),
            inclination_angle_at_reference_time=inclination,
            amplitude_of_cosine_harmonic_correction_term_to_orbit_radius=np.float64(0.0),
            argument_of_perigee=argument_of_perigee,
            rate_of_right_ascension=np.float64(0.0),
            rate_of_inclination_angle=np.float64(0.0),
            codes_on_l2_channel=np.float64(0.0),
            gps_week_number=np.float64(0.0),
            l2_p_data_flag=np.float64(0.0),
            sv_accuracy=np.float64(0.0),
            sv_health=np.float64(0.0),
            tgd_total_group_delay=np.float64(0.0),
            iodc_issue_of_data_clock=np.float64(0.0),
            transmission_time_of_message=np.float64(0.0),
            fit_interval_in_hours=np.float64(0.0),
        )

        calculated_position_time = Time("2025-04-19 09:00:00.000", format="iso")
        calculated_position_time.format = "gps"
        satellite = GpsSatellite(satellite_parameters)
        (position, velocity) = satellite.position_velocity(calculated_position_time)

        # Calculated by hand
        self.assertAlmostEqual(position[0], np.float64(-12_214_024.99), delta=0.1)
        self.assertAlmostEqual(position[1], np.float64(-40_481_865.45), delta=0.1)
        self.assertAlmostEqual(position[2], np.float64(-1_945_051.56), delta=0.1)
        # Velocity makes no sense for this example

    def test_precise(self):
        # Data from COD0MGXFIN_20251000000_01D_05M_ORB.SP3 and CORD00ARG_R_20251000000_01D_GN.rnx
        # Courtesy of NASA's CDDIS database
        
        # 2025 04 09 23 59 44 GPS
        satellite_parameters = gps_orbital_parameters.from_rinex("""\
G06 2025 04 09 23 59 44-2.959421835840E-04-2.148681232939E-11 0.000000000000E+00
     5.000000000000E+00-9.875000000000E+00 3.730869691412E-09-1.899624399301E+00
    -5.550682544708E-07 3.463134868070E-03 1.232884824276E-05 5.153554180145E+03
     3.455840000000E+05-9.499490261078E-08 2.686281956689E+00-3.539025783539E-08
     9.890951254507E-01 1.600000000000E+02-6.486031939322E-01-7.595316375414E-09
    -1.135761594744E-10 1.000000000000E+00 2.361000000000E+03 0.000000000000E+00
     2.000000000000E+00 0.000000000000E+00 3.725290298462E-09 5.000000000000E+00
     3.398820000000E+05 4.000000000000E+00\
        """)
        satellite = GpsSatellite(satellite_parameters)
        # 2025  4 10  0  0  0.00000000 GPS
        calculated_position_time = Time(1428278400.0, format="gps")
        (position, velocity) = satellite.position_velocity(calculated_position_time)
        # Here error is larger because the estimated satellite position is an amalgamation of data
        # PG06  23201.392605  -4035.022068 -12344.501318   -295.941970
        self.assertAlmostEqual(position[0], np.float64(23_201_392.605), delta=5.0)
        self.assertAlmostEqual(position[1], np.float64(-4_035_022.068), delta=5.0)
        self.assertAlmostEqual(position[2], np.float64(-12_344_501.318), delta=5.0)
        

if __name__ == '__main__':
    unittest.main()
