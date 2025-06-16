import numpy as np
import scipy

from src.conversions import velocity_eci2ecef


class Solver:
    def __init__(self, satellite_clock_bias, satellite_frequencies):
        self.satellite_clock_bias = satellite_clock_bias
        self.satellite_frequencies = satellite_frequencies

    def solve_position(self, satellite_positions_ecef, pseudorange, xtol):
        """
        A linearization of the pseudorange error, via a taylor aproximation, is minimized.
        But has problems on numerical precision (substracts large floating values)
        From http://www.grapenthin.org/notes/2019_03_11_pseudorange_position_estimation/
        Numeric fixes from https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
        :return:
        """
        satellite_amount = satellite_positions_ecef.shape[0]
        gnss_position_aproximation = np.array([1, 1, 1], dtype=np.float64)
        gnss_receiver_clock_bias_approximation = np.float64(1e-6) * scipy.constants.c
        gnss_position_error = np.inf

        # TODO add satelite weighting

        for _ in range(100):
            if gnss_position_error < xtol:
                break

            gnss_pseudorange_approximation = (
                        np.linalg.norm(satellite_positions_ecef - gnss_position_aproximation, axis=1)
                        + (gnss_receiver_clock_bias_approximation - self.satellite_clock_bias * scipy.constants.c))

            delta_gnss_pseudorange = pseudorange.copy() - gnss_pseudorange_approximation

            delta_satelites = gnss_position_aproximation - satellite_positions_ecef
            cs = np.ones((1, satellite_amount), dtype=np.float64)

            G = np.concatenate((delta_satelites / gnss_pseudorange_approximation.reshape((-1, 1)), cs.T), axis=1)

            m = (np.linalg.pinv(G) @ delta_gnss_pseudorange.T).reshape(-1)
            gnss_position_delta = m[:3]
            gnss_clock_bias_delta = m[-1]

            gnss_position_aproximation += gnss_position_delta
            gnss_receiver_clock_bias_approximation += gnss_clock_bias_delta

            gnss_position_error = np.linalg.norm(gnss_position_delta)

        gnss_receiver_clock_bias_approximation /= scipy.constants.c

        return gnss_position_aproximation, gnss_receiver_clock_bias_approximation, np.linalg.norm(delta_gnss_pseudorange)


    def solve_position_scipy(self, satellite_positions_ecef, pseudorange, xtol):
        """
        A linearization of the pseudorange error, via scipy.optimize.least_squares
        This is an adaptation of getGnssPositionTaylor
        :return:
        """
        # TODO add satelite weighting
        satellite_amount = satellite_positions_ecef.shape[0]

        def pseudoranges_approximation(gnss_position_aproximation, gnss_receiver_clock_bias_approximation):
            return (np.linalg.norm(satellite_positions_ecef - gnss_position_aproximation, axis=1)
                                        + (gnss_receiver_clock_bias_approximation - self.satellite_clock_bias * scipy.constants.c))

        def fun(x):
            gnss_position_aproximation = x[:3]
            gnss_receiver_clock_bias_approximation = x[-1]
            approx = pseudoranges_approximation(gnss_position_aproximation, gnss_receiver_clock_bias_approximation)
            return (pseudorange - approx).reshape(-1)

        def jac(x):
            gnss_position_aproximation = x[:3]
            gnss_receiver_clock_bias_approximation = x[-1]

            delta_satelites = satellite_positions_ecef - gnss_position_aproximation
            cs = np.ones((1, satellite_amount), dtype=np.float64) * -1

            approx = pseudoranges_approximation(gnss_position_aproximation, gnss_receiver_clock_bias_approximation)

            G = np.concatenate((delta_satelites / approx.reshape((-1, 1)), cs.T), axis=1)

            return G

        result = scipy.optimize.least_squares(fun, x0=np.array([1, 1, 1, 1e-6 * scipy.constants.c], dtype=np.float64), method='lm', jac=jac, xtol=xtol)
        gnss_position_aproximation = result.x[:3]
        gnss_receiver_clock_bias_approximation = result.x[-1] / scipy.constants.c
        gnss_position_error = np.linalg.norm(result.fun)

        return gnss_position_aproximation, gnss_receiver_clock_bias_approximation, gnss_position_error

    def solve_velocity(self, satellite_positions_eci, satellite_velocities_eci, direct_doppler, receiver_position_ecef,
                       xtol):
        """
        A linearization of the pseudorange rate error, and least squares solutions
        From Global Positioning System section 6.2.1, adapted from getGnssPositionTaylor
        """
        # TODO add satelite weighting
        satellite_amount = satellite_positions_eci.shape[0]

        pseudorange_rates = scipy.constants.c / self.satellite_frequencies * direct_doppler

        gnss_velocity_eci_aproximation = np.array([1, 1, 1], dtype=np.float64)
        gnss_receiver_clock_drift_approximation = np.float64(1e-6)
        gnss_velocity_error = np.inf

        # TODO add satelite weighting

        for _ in range(100):
            if gnss_velocity_error < xtol:
                break

            velocity_difference = gnss_velocity_eci_aproximation - satellite_velocities_eci
            satellite_user_delta = satellite_positions_eci - receiver_position_ecef
            satellite_line_of_sight = satellite_user_delta / np.linalg.norm(satellite_user_delta, axis=1).reshape((-1, 1))
            velocity_scalar_projection = np.sum(velocity_difference * satellite_line_of_sight, axis=1)
            pseudorange_rates_approximation = velocity_scalar_projection + gnss_receiver_clock_drift_approximation

            delta_gnss_pseudorange_rates = pseudorange_rates - pseudorange_rates_approximation

            cs = np.ones((1, satellite_amount), dtype=np.float64)

            G = np.concatenate((satellite_line_of_sight, cs.T), axis=1)

            m = (np.linalg.pinv(G) @ delta_gnss_pseudorange_rates.T).reshape(-1)
            gnss_velocity_delta = m[:3]
            gnss_clock_drift_delta = m[-1]

            gnss_velocity_eci_aproximation += gnss_velocity_delta
            gnss_receiver_clock_drift_approximation += gnss_clock_drift_delta

            gnss_velocity_error = np.linalg.norm(gnss_velocity_delta)

        receiver_position_eci = receiver_position_ecef
        gnss_velocity_ecef_aproximation = velocity_eci2ecef(receiver_position_eci, gnss_velocity_eci_aproximation, 0.0)

        return gnss_velocity_ecef_aproximation, gnss_receiver_clock_drift_approximation, gnss_velocity_error


    def solve_velocity_scipy(self, satellite_positions_ecef, satellite_velocities_ecef, direct_doppler, receiver_position, xtol):
        """
        A linearization of the pseudorange rate error, via scipy.optimize.least_squares
        This is an adaptation of getGnssPositionScipy
        From Navigation from Low Earth Orbit – Part 2: Models, Implementation, and Performance section 2.2
        """
        # TODO add satelite weighting

        pseudorange_rate = (- scipy.constants.c / self.satellite_frequencies) * direct_doppler

        def pseudorange_rates_approximation(gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation):
            rate_difference = gnss_velocity_aproximation - satellite_velocities_ecef

            line_of_sight = receiver_position - satellite_positions_ecef
            line_of_sight_unit = line_of_sight / np.linalg.norm(line_of_sight, axis=1).reshape((-1,1))
            velocity_scalar_projection = np.sum(rate_difference * line_of_sight_unit, axis=1)

            clock_drift_effect = gnss_receiver_clock_drift_approximation

            return velocity_scalar_projection + clock_drift_effect

        def fun(x):
            gnss_velocity_aproximation = x[:3]
            gnss_receiver_clock_drift_approximation = x[-1]
            approx = pseudorange_rates_approximation(gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation)
            return (pseudorange_rate - approx).reshape(-1)

        result = scipy.optimize.least_squares(fun, x0=np.array([1, 1, 1, 1e-6 * scipy.constants.c], dtype=np.float64), method='lm', xtol=xtol)
        gnss_velocity_aproximation = result.x[:3]
        gnss_receiver_clock_drift_approximation = result.x[-1] / scipy.constants.c
        gnss_velocity_error = np.linalg.norm(result.fun)

        return gnss_velocity_aproximation, gnss_receiver_clock_drift_approximation, gnss_velocity_error

