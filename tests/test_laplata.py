import unittest

import numpy as np
import scipy

from src.solver import Solver

SATELLITE_POSITIONS = np.array([
    [ 23313.808953, -10932.861917,   6118.858765],  # G03
    [ 20721.531386,  -2122.508536, -16544.439971],  # G04
    [ -6551.363939, -24529.143017,  -7545.001989],  # G06
    [  6145.944054, -22903.241280, -10980.933975],  # G07
    [ 10459.030806, -11836.885217, -21446.749063],  # G09
    [-11918.525405, -17026.839853, -16540.090888],  # G11
    [ 17704.951923,   5584.874094, -19488.460054],  # G16
    [-12454.294776,  -9241.513189, -21493.998367],  # G20
    [ 10034.802439,  12333.836320, -21413.360002],  # G26
    [  -877.603991, -26355.777864,  -1405.678434],  # G30
], dtype=np.float64) * 1_000

SATELLITE_CLOCK_BIAS = np.array([
      94.788844 * 1e-6,  # G03
     254.337408 * 1e-6,  # G04
     460.051663 * 1e-6,  # G06
     -26.063657 * 1e-6,  # G07
      44.413765 * 1e-6,  # G09
    -522.809294 * 1e-6,  # G11
    -393.237187 * 1e-6,  # G16
     393.809848 * 1e-6,  # G20
     204.435378 * 1e-6,  # G26
    -450.877485 * 1e-6,  # G30
], dtype=np.float64) * 1e-6

PSEUDORANGES = np.array([
    23_619_948.516,  # G03
    22_165_885.860,  # G04
    22_342_010.927,  # G06
    20_158_816.025,  # G07
    20_755_193.441,  # G09
    23_406_732.283,  # G11
    24_106_990.242,  # G16
    23_836_960.614,  # G20
    25_451_174.549,  # G26
    22_453_153.857,  # G30
], dtype=np.float64)

REAL_POSITION  = np.array([2765120.6553, -4449249.8563, -3626405.2770], dtype=np.float64)
REAL_RECEIVER_BIAS = np.float64(-1000.0 * 1e-9) # Assumed, I don't know

class GroundTruthTest(unittest.TestCase):
    def test_laplata_position(self):
        for i in range(PSEUDORANGES.shape[0]):
            pseudorange_estimation = np.linalg.norm(SATELLITE_POSITIONS[i] - REAL_POSITION) + (REAL_RECEIVER_BIAS - SATELLITE_CLOCK_BIAS[i]) * scipy.constants.c
            self.assertAlmostEqual(PSEUDORANGES[i], pseudorange_estimation, places=-3, msg=f"at index {i}")


class TaylorAproximationTest(unittest.TestCase):
    def test_laplata_position(self):
        solver = Solver(SATELLITE_POSITIONS.shape[0], SATELLITE_CLOCK_BIAS,None)
        approx_position, clock_bias, gnss_position_error = solver.solve_position(SATELLITE_POSITIONS, PSEUDORANGES, 1e-5)

        distance = np.linalg.norm(approx_position - REAL_POSITION)

        self.assertLess(distance, 1)
        self.assertAlmostEqual(clock_bias, REAL_RECEIVER_BIAS, delta=0.001)


if __name__ == '__main__':
    unittest.main()
