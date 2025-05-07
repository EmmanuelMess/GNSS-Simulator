import unittest

import numpy as np

from src import gps_orbital_parameters

class ReadRinex(unittest.TestCase):
    def test_read(self):
        rinex_text ="""\
G06 2025 04 09 23 59 44-2.959421835840E-04-2.148681232939E-11 0.000000000000E+00
     5.000000000000E+00-9.875000000000E+00 3.730869691412E-09-1.899624399301E+00
    -5.550682544708E-07 3.463134868070E-03 1.232884824276E-05 5.153554180145E+03
     3.455840000000E+05-9.499490261078E-08 2.686281956689E+00-3.539025783539E-08
     9.890951254507E-01 1.600000000000E+02-6.486031939322E-01-7.595316375414E-09
    -1.135761594744E-10 1.000000000000E+00 2.361000000000E+03 0.000000000000E+00
     2.000000000000E+00 0.000000000000E+00 3.725290298462E-09 5.000000000000E+00
     3.398820000000E+05 4.000000000000E+00\
     """

        satellite = gps_orbital_parameters.from_rinex(rinex_text)

        self.assertEqual(satellite.satellite_system, "G")
        self.assertEqual(satellite.prn_number, 6)
        self.assertAlmostEqual(satellite.epoch.value, 2.361000000000E+03 * 7 * 24 * 60 * 60 + 3.455840000000E+05, delta=1.0)
        self.assertAlmostEqual(satellite.sv_clock_bias, np.float64(-2.959421835840E-04), places=13)
        self.assertAlmostEqual(satellite.sv_clock_drift, np.float64(-2.148681232939E-11), places=13)
        self.assertAlmostEqual(satellite.sv_clock_drift_rate, np.float64(0.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.issue_of_data_ephemeris, np.float64(5.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.amplitude_sine_harmonic_correction_term_to_orbit_radius, np.float64(-9.875000000000E+00), places=13)
        self.assertAlmostEqual(satellite.mean_motion_difference_from_computed_value, np.float64(3.730869691412E-09), places=13)
        self.assertAlmostEqual(satellite.mean_anomaly_at_reference_time, np.float64(-1.899624399301E+00), places=13)
        self.assertAlmostEqual(satellite.amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude, np.float64(-5.550682544708E-07), places=13)
        self.assertAlmostEqual(satellite.eccentricity, np.float64(3.463134868070E-03), places=13)
        self.assertAlmostEqual(satellite.amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude, np.float64(1.232884824276E-05), places=13)
        self.assertAlmostEqual(satellite.square_root_of_semi_major_axis, np.float64(5.153554180145E+03), places=13)
        self.assertAlmostEqual(satellite.time_of_ephemeris, np.float64(3.455840000000E+05), places=13)
        self.assertAlmostEqual(satellite.amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination, np.float64(-9.499490261078E-08), places=13)
        self.assertAlmostEqual(satellite.longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch, np.float64(2.686281956689E+00), places=13)
        self.assertAlmostEqual(satellite.amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination, np.float64(-3.539025783539E-08), places=13)
        self.assertAlmostEqual(satellite.inclination_angle_at_reference_time, np.float64(9.890951254507E-01), places=13)
        self.assertAlmostEqual(satellite.amplitude_of_cosine_harmonic_correction_term_to_orbit_radius, np.float64(1.600000000000E+02), places=13)
        self.assertAlmostEqual(satellite.argument_of_perigee, np.float64(-6.486031939322E-01), places=13)
        self.assertAlmostEqual(satellite.rate_of_right_ascension, np.float64(-7.595316375414E-09), places=13)
        self.assertAlmostEqual(satellite.rate_of_inclination_angle, np.float64(-1.135761594744E-10), places=13)
        self.assertAlmostEqual(satellite.codes_on_l2_channel, np.float64(1.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.gps_week_number, np.float64(2.361000000000E+03), places=13)
        self.assertAlmostEqual(satellite.l2_p_data_flag, np.float64(0.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.sv_accuracy, np.float64(2.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.sv_health, np.float64(0.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.tgd_total_group_delay, np.float64(3.725290298462E-09), places=13)
        self.assertAlmostEqual(satellite.iodc_issue_of_data_clock, np.float64(5.000000000000E+00), places=13)
        self.assertAlmostEqual(satellite.transmission_time_of_message, np.float64(3.398820000000E+05), places=13)
        self.assertAlmostEqual(satellite.fit_interval_in_hours, np.float64(4.000000000000E+00), places=13)


if __name__ == '__main__':
    unittest.main()
