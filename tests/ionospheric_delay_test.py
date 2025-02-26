import unittest

import numpy as np
import scipy

from tests.constants import *
import main


class IonosphericDelayTest(unittest.TestCase):
    def test_nighttime(self):
        receiver_ecef = np.array([ WSG84_SEMI_MAJOR_AXIS, 0, 0 ], dtype=np.float64)
        satellites_ecef = np.array([[WSG84_SEMI_MAJOR_AXIS * 2, 0, 0]], dtype=np.float64)
        satellite_clock_bias = np.array([0])

        alphas = np.array([[3.82e-8, 1.49e-8, -1.79e-7, 0]], dtype=np.float64)
        betas = np.array([[1.43e5, 0, -3.28e5, 1.13e5]], dtype=np.float64)
        simulator = main.Simulator(None, satellites_ecef, satellite_clock_bias,
                                   None, main.GPS_L1_FREQUENCY, alphas, betas, None,
                                   None, None, None)


        time_gps = 4*60*60 # 04hs at UTC
        delay = simulator._ionospheric_delay_calculation(receiver_ecef, time_gps)
        self.assertEqual(delay.shape, (1,))
        self.assertAlmostEqual(delay[0], 5 * 1e-9, delta=TIME_PRECISION)

    def test_paper(self):
        # Test from Klobuchar 1987

        receiver_ecef = np.array([-849_609.76, -4_818_376.38, 4_077_985.57 ], dtype=np.float64)
        receiver_llh = main.ecef2llh(receiver_ecef)
        self.assertAlmostEqual(np.rad2deg(receiver_llh[0]), 40, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(np.rad2deg(receiver_llh[1]), -100, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(receiver_llh[2], 0, delta=HEIGHT_PRECISION)

        theta, l = receiver_llh[0], receiver_llh[1]
        e_unit = np.array([-np.sin(l), np.cos(l), 0], dtype=np.float64)
        n_unit = np.array([-np.cos(l)*np.sin(theta), -np.sin(l)*np.sin(theta), np.cos(theta)], dtype=np.float64)
        u_unit = np.array([np.cos(l)*np.cos(theta), np.sin(l)*np.cos(theta), np.sin(theta)], dtype=np.float64)
        direction_unit = (np.sin(np.deg2rad(210))*np.cos(np.deg2rad(20))*e_unit + np.cos(np.deg2rad(210))*np.cos(np.deg2rad(20))*n_unit + np.sin(np.deg2rad(20))*u_unit)
        satellite_ecef = receiver_ecef + direction_unit * 30_200
        aer = main.ecef2aer(receiver_ecef, satellite_ecef)
        self.assertAlmostEqual(np.rad2deg(aer[0]), 210, delta=AZIMUTH_PRECISION)
        self.assertAlmostEqual(np.rad2deg(aer[1]), 20, delta=ELEVATION_PRECISION)

        satellite_clock_bias = np.array([0])
        alphas = np.array([[3.82e-8, 1.49e-8, -1.79e-7, 0]], dtype=np.float64)
        betas = np.array([[1.43e5, 0, -3.28e5, 1.13e5]], dtype=np.float64)
        simulator = main.Simulator(None, np.array([satellite_ecef], dtype=np.float64), satellite_clock_bias,
                                   None, main.GPS_L1_FREQUENCY, alphas, betas, None,
                                   None, None, None)


        time_gps = 2045
        delay = simulator._ionospheric_delay_calculation(receiver_ecef, time_gps)
        self.assertEqual(delay.shape, (1,))
        # TODO paper example is broken self.assertAlmostEqual(delay[0], 77.6 * 1e-9, delta=TIME_PRECISION)

    def test_noaa(self):
        # Test from https://geodesy.noaa.gov/gps-toolbox/ovstedal.htm

        receiver_ecef = np.array( [ -849_609.76, -4_818_376.38, 4_077_985.57 ]  , dtype=np.float64)
        receiver_llh = main.ecef2llh(receiver_ecef)
        self.assertAlmostEqual(np.rad2deg(receiver_llh[0]), 40, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(np.rad2deg(receiver_llh[1]), -100, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(receiver_llh[2], 0, delta=HEIGHT_PRECISION)

        theta, l = receiver_llh[0], receiver_llh[1]
        e_unit = np.array([-np.sin(l), np.cos(l), 0], dtype=np.float64)
        n_unit = np.array([-np.cos(l)*np.sin(theta), -np.sin(l)*np.sin(theta), np.cos(theta)], dtype=np.float64)
        u_unit = np.array([np.cos(l)*np.cos(theta), np.sin(l)*np.cos(theta), np.sin(theta)], dtype=np.float64)
        direction_unit = (np.sin(np.deg2rad(210))*np.cos(np.deg2rad(20))*e_unit + np.cos(np.deg2rad(210))*np.cos(np.deg2rad(20))*n_unit + np.sin(np.deg2rad(20))*u_unit)
        satellite_ecef = receiver_ecef + direction_unit * 30_200
        aer = main.ecef2aer(receiver_ecef, satellite_ecef)
        self.assertAlmostEqual(np.rad2deg(aer[0]), 210, delta=AZIMUTH_PRECISION)
        self.assertAlmostEqual(np.rad2deg(aer[1]), 20, delta=ELEVATION_PRECISION)

        satellite_clock_bias = np.array([0])
        alphas = np.array([[.3820e-7,  .1490e-7,  -.1790e-06,   .0000]], dtype=np.float64)
        betas = np.array([[.1430e6, .0000,  -.3280e+6,   .1130e+06]], dtype=np.float64)
        simulator = main.Simulator(None, np.array([satellite_ecef], dtype=np.float64), satellite_clock_bias,
                                   None, main.GPS_L1_FREQUENCY, alphas, betas, None,
                                   None, None, None)

        #time = 2000 01 01 20 45 00.

        time_gps = 593100
        delay = simulator._ionospheric_delay_calculation(receiver_ecef, time_gps)
        self.assertEqual(delay.shape, (1,))
        self.assertAlmostEqual(scipy.constants.c * delay[0], 23.7, delta=DISTANCE_PRECISION)


if __name__ == '__main__':
    unittest.main()
