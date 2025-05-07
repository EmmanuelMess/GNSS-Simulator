import unittest

from src.antenna_simulator import AntennaSimulator
from src.conversions import ecef2llh
from tests.constants import *

class TrosposphericModelTest(unittest.TestCase):
    def test_laplata(self):
        # AGGO00ARG 20240101 000000 UTC data courtesy of NASA's CDDIS
        position = np.array([ 2_765_120.6553, -4_449_249.8563, -3_626_405.2770 ], dtype=np.float64)

        position_llh = ecef2llh(position)

        # From https://network.igs.org/AGGO00ARG
        self.assertAlmostEqual(np.rad2deg(position_llh[0]), -34.873708, delta=1e-5)
        self.assertAlmostEqual(np.rad2deg(position_llh[1]), -58.139861, delta=1e-5)
        self.assertAlmostEqual(position_llh[2], 42.085, delta=1)

        cutoff_angle = np.deg2rad(7)

        simulator = AntennaSimulator(None, np.array([]), None, None,
                                          None, None, np.float64(0.0), None,
                                          None, None, None, cutoff_angle)

        # Atmospheric data from SMN, Argentina
        # Vapor pressure of water from CRC Handbook of Chemistry and Physics, 85th Edition
        dry_delay, wet_delay = simulator._saastamoinen_model(position, 1_016.0, 18.6 + 273.15, 2.0644 * 10)
        delay_saastamoinen = dry_delay + wet_delay

        trotot = np.float64(2465.8) # mm

        self.assertAlmostEqual(delay_saastamoinen * 1e3, trotot, delta=0.1 * 1e3)

        delay_unb4 = simulator._per_satelite_tropospheric_delay(position_llh, np.pi/2, 0)

        self.assertAlmostEqual(delay_unb4 * 1e3, trotot, delta=0.1 * 1e3)


if __name__ == '__main__':
    unittest.main()
