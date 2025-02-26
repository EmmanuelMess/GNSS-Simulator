import unittest

import numpy as np

import main
from tests.constants import *

class SemicirclesTest(unittest.TestCase):
    def test_trivial_0(self):
        self.assertEqual(main.rad2semicircles(0), 0)

    def test_trivial_1(self):
        self.assertEqual(main.rad2semicircles(np.pi), 1)

    def test_simple(self):
        # Test from Klobuchar 1987, delta is high because the results from the paper seem wrong by a large amount
        self.assertAlmostEqual(main.rad2semicircles(np.deg2rad(np.float64(7.2))), 0.03996, delta=0.1)
        self.assertAlmostEqual(main.rad2semicircles(np.deg2rad(np.float64(38.7))), 0.215, delta=0.1)
        self.assertAlmostEqual(main.rad2semicircles(np.deg2rad(np.float64(-124.795))), -0.6399, delta=0.1)
        self.assertAlmostEqual(main.rad2semicircles(np.deg2rad(np.float64(45.16))), 0.2529, delta=0.1)


class LlhTest(unittest.TestCase):
    def test_trivial_0(self):
        llh = main.ecef2llh(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], 0, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_1(self):
        llh = main.ecef2llh(np.array([0, WSG84_SEMI_MAJOR_AXIS, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], np.pi/2, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_2(self):
        llh = main.ecef2llh(np.array([0, 0, WSG84_SEMI_MINOR_AXIS]))

        self.assertAlmostEqual(llh[0], np.pi/2, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], 0, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_3(self):
        llh = main.ecef2llh(np.array([-WSG84_SEMI_MAJOR_AXIS, 0, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], np.pi, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_4(self):
        llh = main.ecef2llh(np.array([0, -WSG84_SEMI_MAJOR_AXIS, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], -np.pi / 2, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_5(self):
        llh = main.ecef2llh(np.array([0, 0, -WSG84_SEMI_MINOR_AXIS]))

        self.assertAlmostEqual(llh[0], -np.pi / 2, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], 0, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

class AerTest(unittest.TestCase):
    def test_trivial(self):
        aer = main.ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 0, 0]))

        # Azimuth not defined
        self.assertAlmostEqual(aer[1],  np.pi/2)
        self.assertAlmostEqual(aer[2], 500)

    def test_north(self):
        aer = main.ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 0, 500]))

        self.assertAlmostEqual(aer[0], 0)
        self.assertAlmostEqual(aer[1], np.pi/4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, 0, 500]))

    def test_east(self):
        aer = main.ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 500, 0]))

        self.assertAlmostEqual(aer[0], np.pi/2)
        self.assertAlmostEqual(aer[1], np.pi/4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, 500, 0]))

    def test_south(self):
        aer = main.ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 0, -500]))

        self.assertAlmostEqual(aer[0], np.pi)
        self.assertAlmostEqual(aer[1], np.pi/4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, 0, -500]))

    def test_west(self):
        aer = main.ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, -500, 0]))

        self.assertAlmostEqual(aer[0], np.pi / 2 * 3)
        self.assertAlmostEqual(aer[1], np.pi / 4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, -500, 0]))

    def test_1(self):
        receiver_position_ecef = np.array([ 1_509_771.99, -4_475_238.73, 4_272_181.45 ], dtype=np.float64)
        satellite_position_ecef = np.array([10_766_080.3, 14_143_607.0, 33_992_388.0], dtype=np.float64)

        receiver_position_llh = main.ecef2llh(receiver_position_ecef)
        self.assertAlmostEqual(np.rad2deg(receiver_position_llh[0]), 42.3221, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(np.rad2deg(receiver_position_llh[1]), -71.3576, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(receiver_position_llh[2], 84.7, delta=HEIGHT_PRECISION)

        aer = main.ecef2aer(receiver_position_ecef, satellite_position_ecef)

        self.assertAlmostEqual(np.rad2deg(aer[0]), 24.8012, delta=AZIMUTH_PRECISION)
        self.assertAlmostEqual(np.rad2deg(aer[1]), 14.6185, delta=ELEVATION_PRECISION)
        self.assertAlmostEqual(aer[2], 36_271_632.7, delta=RANGE_PRECISION)

if __name__ == '__main__':
    unittest.main()
