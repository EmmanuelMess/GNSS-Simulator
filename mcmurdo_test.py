import unittest

import numpy as np
import scipy

import main

SATELLITE_POSITIONS = np.array([
    [10011.699639, 16084.389751, -19050.244266],   # G04
    [14697.108223, 12954.318873, -18522.373638],   # G11
    [-21395.252372, 13872.727013, -6882.375231],   # G14
    [-3750.820337, -17956.885654, -19189.766833],  # G15
    [-17051.363484, -4161.527113, -19503.448336],  # G18
    [-75.524186, 18164.222089, -19415.548695],     # G19
    [-12933.794843, 10205.759827, -20569.279215],  # G22
    [-16548.513468, -14236.584786, -15267.729561], # G24
    [-9678.067077, 22140.496825, -10913.108494],   # G27
    [13506.960104, -4894.537400, -21719.611795],   # G28
], dtype=np.float64) * 1_000

SATELLITE_CLOCK_BIAS = np.array([
    -2.469151,    # G04
    -592.243492,  # G11
    47.983976,    # G14
    -261.224294,  # G15
    418.339886,   # G18
    -516.232497,  # G19
    372.435926,   # G22
    -1.557331,    # G24
    5.315444,     # G27
    454.532750,   # G28
], dtype=np.float64) * 1e-6

PSEUDORANGES = np.array([
    23_299_637.109,  # G04
    24_033_202.547,  # G11
    24_191_714.477,  # G14
    22_621_249.406,  # G15
    20_928_744.125,  # G18
    22_397_361.672,  # G19
    20_822_467.477,  # G22
    22_907_727.289,  # G24
    23_827_164.883,  # G27
    21_969_433.359,  # G28
], dtype=np.float64)

REAL_POSITION  = np.array([-1353875.822, 314824.972, -6205811.52], dtype=np.float64)
REAL_RECEIVER_BIAS = np.float64(-795.52 * 1e-9)

FOUR_SATELLITES = np.array([0, 1, 2, 3]) # G04 G11 G14 G15
FOUR_GOOD_SATELLITES = np.array([0, 4, 5, 6]) # G04 G18 G19 G22

class GroundTruthTest(unittest.TestCase):
    def test_mcmurdo_position(self):
        for i in range(PSEUDORANGES.shape[0]):
            pseudorange_estimation = np.linalg.norm(SATELLITE_POSITIONS[i] - REAL_POSITION) + (REAL_RECEIVER_BIAS - SATELLITE_CLOCK_BIAS[i]) * scipy.constants.c
            self.assertAlmostEqual(PSEUDORANGES[i], pseudorange_estimation, places=-3, msg=f"at index {i}")


class TaylorAproximationTest(unittest.TestCase):
    def test_mcmurdo_position_few_satellites(self):
        solver = main.Solver(SATELLITE_POSITIONS[FOUR_SATELLITES], SATELLITE_CLOCK_BIAS[FOUR_SATELLITES],
                             None, None)
        approx_position, clock_bias, gnss_position_error =\
            solver.getGnssPositionTaylor(PSEUDORANGES[FOUR_SATELLITES], 1e-3)

        distance = np.linalg.norm(approx_position - REAL_POSITION)

        self.assertLess(distance, 300)
        self.assertAlmostEqual(clock_bias, REAL_RECEIVER_BIAS, delta=0.001)

    def test_mcmurdo_position(self):
        solver = main.Solver(SATELLITE_POSITIONS, SATELLITE_CLOCK_BIAS,None, None)
        approx_position, clock_bias, gnss_position_error = solver.getGnssPositionTaylor(PSEUDORANGES, 1e-3)

        distance = np.linalg.norm(approx_position - REAL_POSITION)

        self.assertLess(distance, 1)
        self.assertAlmostEqual(clock_bias, REAL_RECEIVER_BIAS, delta=0.001)


class LSEAproximationTest(unittest.TestCase):
    def test_mcmurdo_position_few_satellites(self):
        solver = main.Solver(SATELLITE_POSITIONS[FOUR_SATELLITES], SATELLITE_CLOCK_BIAS[FOUR_SATELLITES],
                             None, None)
        approx_position, clock_bias, gnss_position_error =\
            solver.getGnssPositionScipy(PSEUDORANGES[FOUR_SATELLITES], 1e-3)

        distance = np.linalg.norm(approx_position - REAL_POSITION)

        self.assertLess(distance, 300)
        self.assertAlmostEqual(clock_bias, REAL_RECEIVER_BIAS, delta=0.001)

    def test_mcmurdo_position(self):
        solver = main.Solver(SATELLITE_POSITIONS, SATELLITE_CLOCK_BIAS, None, None)
        approx_position, clock_bias, gnss_position_error = solver.getGnssPositionScipy(PSEUDORANGES, 1e-3)

        distance = np.linalg.norm(approx_position - REAL_POSITION)

        self.assertLess(distance, 1)
        self.assertAlmostEqual(clock_bias, REAL_RECEIVER_BIAS, delta=0.001)


if __name__ == '__main__':
    unittest.main()
