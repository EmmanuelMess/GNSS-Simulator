import numpy as np

OMEGA_EARTH = np.float64(72.92115e-6)
"""
WGS84 earth angular velocity [rad / s]
"""

# WGS 84 value of the earth's gravitational constant for GPS user [m^3 / s^2]
MU_EARTH = np.float64(3.986005e14)

GPS_L1_FREQUENCY = np.float64(1_575.42) * 1e6 # Hz
GPS_L5_FREQUENCY = np.float64(1_176.45) * 1e6 # Hz