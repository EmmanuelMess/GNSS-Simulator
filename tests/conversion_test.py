import unittest

import numpy as np

import main

WSG84_SEMI_MAJOR_AXIS = np.float64(6_378_137)

class LlhTest(unittest.TestCase):
    def test_trivial(self):
        llh = main.ecef2llh(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]))

        self.assertAlmostEqual(llh[0], 0)
        self.assertAlmostEqual(llh[1], 0)
        self.assertAlmostEqual(llh[2], 0)

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

if __name__ == '__main__':
    unittest.main()
