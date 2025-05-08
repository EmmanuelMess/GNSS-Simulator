import unittest
from typing import Optional
import re

import numpy as np
from skyfield.api import load
from skyfield.iokit import parse_tle_file
from skyfield.timelib import Time, Timescale
from skyfield.toposlib import wgs84

from src.conversions import ecef2llh, seconds2day_of_year, rad2semicircles, ecef2aer, llh2ecef
from tests.constants import *


class DayOfYearTest(unittest.TestCase):
    def test_trivial_0(self):
        self.assertAlmostEqual(seconds2day_of_year(0), 6, delta=0.5)

    def test_2024_01_01(self):
        self.assertAlmostEqual(seconds2day_of_year(1388102418), 1, delta=0.25)

class SemicirclesTest(unittest.TestCase):
    def test_trivial_0(self):
        self.assertEqual(rad2semicircles(0), 0)

    def test_trivial_1(self):
        self.assertEqual(rad2semicircles(np.pi), 1)

    def test_simple(self):
        # Test from Klobuchar 1987, delta is high because the results from the paper seem wrong by a large amount
        self.assertAlmostEqual(rad2semicircles(np.deg2rad(np.float64(7.2))), 0.03996, delta=0.1)
        self.assertAlmostEqual(rad2semicircles(np.deg2rad(np.float64(38.7))), 0.215, delta=0.1)
        self.assertAlmostEqual(rad2semicircles(np.deg2rad(np.float64(-124.795))), -0.6399, delta=0.1)
        self.assertAlmostEqual(rad2semicircles(np.deg2rad(np.float64(45.16))), 0.2529, delta=0.1)

class Llh2EcefTest(unittest.TestCase):
    def test_trivial_0(self):
        ecef = llh2ecef(np.array([0, 0, 0], dtype=np.float64))

        self.assertAlmostEqual(ecef[0], WSG84_SEMI_MAJOR_AXIS, delta=DISTANCE_PRECISION)
        self.assertAlmostEqual(ecef[1], 0, delta=DISTANCE_PRECISION)
        self.assertAlmostEqual(ecef[2], 0, delta=DISTANCE_PRECISION)

    def test_mcmurdo(self):
        # From http://grapenthin.org/teaching/geop555/LAB03_position_estimation.html
        # WARNING: I think the presition of the measurement is lower than it should be

        llh = np.array([np.deg2rad(-77.453051), np.deg2rad(166.909306), 1805.735], dtype=np.float64)

        ecef = llh2ecef(llh)

        self.assertAlmostEqual(ecef[0], -1_353_875.822, delta=500) # see warning
        self.assertAlmostEqual(ecef[1], 314_824.972, delta=500) # see warning
        self.assertAlmostEqual(ecef[2], -6_205_811.52, delta=500) # see warning

    def test_dumont(self):
        # Position from https://network.igs.org/DUMG00ATA converted with https://www.convertecef.com/
        llh = np.array([np.deg2rad(-66.665169), np.deg2rad(140.002200), -3.38], dtype=np.float64)

        ecef = llh2ecef(llh)
        self.assertAlmostEqual(ecef[0], -1940884.12, delta=DISTANCE_PRECISION)
        self.assertAlmostEqual(ecef[1], 1628468.15, delta=DISTANCE_PRECISION)
        self.assertAlmostEqual(ecef[2], -5833719.93, delta=DISTANCE_PRECISION)

    def test_receiver(self):
        receiver_position_llh = np.array([ np.deg2rad(42.3221), np.deg2rad(-71.3576), 84.7], dtype=np.float64)

        receiver_position_ecef = llh2ecef(receiver_position_llh)
        self.assertAlmostEqual(receiver_position_ecef[0], 1_509_771.99, delta=DISTANCE_PRECISION)
        self.assertAlmostEqual(receiver_position_ecef[1], -4_475_238.73, delta=DISTANCE_PRECISION)
        self.assertAlmostEqual(receiver_position_ecef[2], 4_272_181.45, delta=DISTANCE_PRECISION)

    def test_matlab(self):
        # Position from https://la.mathworks.com/help/map/ref/geodetic2ecef.html
        # WARNING: low presicion estimate
        lat = 48.8562
        lon = 2.3508
        h = 0.0674
        llh = np.array([np.deg2rad(lat), np.deg2rad(lon), h], dtype=np.float64)
        ecef = llh2ecef(llh)

        self.assertAlmostEqual(ecef[0], 4.2010e+06, delta=1000) # see warning
        self.assertAlmostEqual(ecef[1], 172.4603e+03, delta=1000) # see warning
        self.assertAlmostEqual(ecef[2], 4.7801e+06, delta=1000) # see warning


class Ecef2LlhTest(unittest.TestCase):
    def test_trivial_0(self):
        llh = ecef2llh(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], 0, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_1(self):
        llh = ecef2llh(np.array([0, WSG84_SEMI_MAJOR_AXIS, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], np.pi/2, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_2(self):
        llh = ecef2llh(np.array([0, 0, WSG84_SEMI_MINOR_AXIS]))

        self.assertAlmostEqual(llh[0], np.pi/2, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], 0, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_3(self):
        llh = ecef2llh(np.array([-WSG84_SEMI_MAJOR_AXIS, 0, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], np.pi, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_4(self):
        llh = ecef2llh(np.array([0, -WSG84_SEMI_MAJOR_AXIS, 0]))

        self.assertAlmostEqual(llh[0], 0, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], -np.pi / 2, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

    def test_trivial_5(self):
        llh = ecef2llh(np.array([0, 0, -WSG84_SEMI_MINOR_AXIS]))

        self.assertAlmostEqual(llh[0], -np.pi / 2, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(llh[1], 0, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(llh[2], 0, delta=HEIGHT_PRECISION)

class AerTest(unittest.TestCase):
    def test_trivial(self):
        aer = ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 0, 0]))

        # Azimuth not defined
        self.assertAlmostEqual(aer[1],  np.pi/2)
        self.assertAlmostEqual(aer[2], 500)

    def test_north(self):
        aer = ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 0, 500]))

        self.assertAlmostEqual(aer[0], 0)
        self.assertAlmostEqual(aer[1], np.pi/4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, 0, 500]))

    def test_east(self):
        aer = ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 500, 0]))

        self.assertAlmostEqual(aer[0], np.pi/2)
        self.assertAlmostEqual(aer[1], np.pi/4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, 500, 0]))

    def test_south(self):
        aer = ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, 0, -500]))

        self.assertAlmostEqual(aer[0], np.pi)
        self.assertAlmostEqual(aer[1], np.pi/4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, 0, -500]))

    def test_west(self):
        aer = ecef2aer(np.array([WSG84_SEMI_MAJOR_AXIS, 0, 0]), np.array([WSG84_SEMI_MAJOR_AXIS+500, -500, 0]))

        self.assertAlmostEqual(aer[0], np.pi / 2 * 3)
        self.assertAlmostEqual(aer[1], np.pi / 4)
        self.assertAlmostEqual(aer[2], np.linalg.norm([500, -500, 0]))

    def test_1(self):
        receiver_position_ecef = np.array([ 1_509_771.99, -4_475_238.73, 4_272_181.45 ], dtype=np.float64)
        satellite_position_ecef = np.array([10_766_080.3, 14_143_607.0, 33_992_388.0], dtype=np.float64)

        receiver_position_llh = ecef2llh(receiver_position_ecef)
        self.assertAlmostEqual(np.rad2deg(receiver_position_llh[0]), 42.3221, delta=LATITUDE_PRECISION)
        self.assertAlmostEqual(np.rad2deg(receiver_position_llh[1]), -71.3576, delta=LONGITUDE_PRECISION)
        self.assertAlmostEqual(receiver_position_llh[2], 84.7, delta=HEIGHT_PRECISION)

        aer = ecef2aer(receiver_position_ecef, satellite_position_ecef)

        self.assertAlmostEqual(np.rad2deg(aer[0]), 24.8012, delta=AZIMUTH_PRECISION)
        self.assertAlmostEqual(np.rad2deg(aer[1]), 14.6185, delta=ELEVATION_PRECISION)
        self.assertAlmostEqual(aer[2], 36_271_632.7, delta=RANGE_PRECISION)

    def test_2(self):
        # We use a list of satellites visible from 0° 00' 00" N 0° 00' 00" E at 2025-04-19 9:00 UTC
        # Rough estimates of skyplot obtained from gnssplanning.com

        PRN_REGEX = re.compile(r"(GPS [A-Z]+-\d+)+\s+\(PRN(?P<prn> \d+)\)")

        def get_prn(name: str) -> Optional[int]:
            match = re.match(PRN_REGEX, name)
            if match is None:
                return None

            return int(match.group("prn"))

        prn_visible = [27, 31, 29, 28, 25, 18, 32, 23, 10]
        receiver_position = wgs84.latlon(0.0, 0.0)

        timescale = load.timescale()
        time_utc = timescale.utc(2025, 4, 19, 9, 0, 0)

        with load.open('gps.tle') as file:
            satellite_orbits = {
                get_prn(satellite.name): satellite for satellite in list(parse_tle_file(file, timescale))
            }

        satellite_elevations_gt = {
            prn: (satellite - receiver_position).at(time_utc).altaz()[0].radians for prn, satellite in satellite_orbits.items()
        }

        satellite_positions = {
            prn: np.array(satellite.at(time_utc).xyz.m, dtype=np.float64) for prn, satellite in satellite_orbits.items()
        }

        satellite_positions_aer = {
            prn: ecef2aer(receiver_position.at(time_utc).xyz.m, satellite_position) for prn, satellite_position in
            satellite_positions.items()
        }

        for prn in satellite_orbits.keys():
            self.assertAlmostEqual(satellite_positions_aer[prn][1], satellite_elevations_gt[prn], delta=np.deg2rad(1))

        for prn in prn_visible:
            self.assertGreaterEqual(satellite_positions_aer[prn][1], np.deg2rad(10))

        for prn in satellite_orbits.keys() - prn_visible:
            self.assertLess(satellite_positions_aer[prn][1], np.deg2rad(10))

        self.assertGreaterEqual(satellite_positions_aer[18][1], np.deg2rad(60))
        self.assertGreaterEqual(satellite_positions_aer[28][1], np.deg2rad(50))
        self.assertGreaterEqual(satellite_positions_aer[32][1], np.deg2rad(30))
        self.assertGreaterEqual(satellite_positions_aer[23][1], np.deg2rad(30))

        for prn in [29, 25, 27, 10]:
            self.assertGreaterEqual(satellite_positions_aer[prn][1], np.deg2rad(10))

if __name__ == '__main__':
    unittest.main()
