from typing import Tuple

import numpy as np
import scipy
from hifitime import Epoch, Unit

from src.gps_orbital_parameters import GpsOrbitalParameters
from src.numpy_types import array3d
from src.conversions import time_gps2seconds_of_week, gps_seconds_wrap, position_ecef2eci, velocity_ecef2eci


class GpsSatellite:
    def __init__(self, orbit_parameters: GpsOrbitalParameters):
        self.orbit_parameters = orbit_parameters

        if orbit_parameters.sv_clock_drift != 0 or orbit_parameters.sv_clock_drift_rate != 0:
            print(f"WARNING: drift and drift rate parameters are not 0 for satellite {orbit_parameters.prn_number} but clock drift will NOT be modeled!")

    def position_velocity_for_receiver(self, receiver_position_ecef: array3d, arrival_gps_time: Epoch,
                                       tolerance: np.float64 = 0.1) -> Tuple[np.float64, Tuple[array3d, array3d]]:
        """
        Compute the satellite position at the time the signal was sent,
        for it to be received at position receiver_position_ecef at time gps_time
        """
        # Algorithm from DESARROLLO E IMPLEMENTACIÓN DE UN SIMULADOR DE DATOS GNSS section 2.3

        dt = np.float64(0)

        # ECI at delta-time 0s is ECEF
        satellite_position_eci, satellite_velocity_eci = self.position_velocity(arrival_gps_time)
        receiver_position_eci = receiver_position_ecef

        for _ in range(10):
            pseudorange = np.linalg.norm(receiver_position_eci - satellite_position_eci)
            error = np.abs(pseudorange - dt * scipy.constants.c)

            if error < tolerance:
                break

            dt = pseudorange / scipy.constants.c

            satellite_position_ecef, satellite_velocity_ecef = self.position_velocity(arrival_gps_time - Unit.Second * dt)
            satellite_position_eci = position_ecef2eci(satellite_position_ecef, -dt)
            satellite_velocity_eci = velocity_ecef2eci(satellite_position_ecef, satellite_velocity_ecef, -dt)

        return -dt, (satellite_position_eci, satellite_velocity_eci)

    def position_velocity(self, gps_time: Epoch) -> Tuple[array3d, array3d]:
        # From IS-GPS-200M table 20-IV

        # Gravitational effect of the Earth
        mu = np.float64(3.986005e14)
        # Rate of rotation of the Earth
        omega_dot_e = np.float64(7.2921151467e-5)

        prn = self.orbit_parameters.prn_number
        e = self.orbit_parameters.eccentricity
        sqrt_a =  self.orbit_parameters.square_root_of_semi_major_axis
        omega_0 =  self.orbit_parameters.longitude_of_ascending_node_of_orbit_plane_at_weekly_epoch
        delta_n = self.orbit_parameters.mean_motion_difference_from_computed_value
        m_0 = self.orbit_parameters.mean_anomaly_at_reference_time
        i_0 = self.orbit_parameters.inclination_angle_at_reference_time
        omega = self.orbit_parameters.argument_of_perigee
        omega_dot = self.orbit_parameters.rate_of_right_ascension
        idot = self.orbit_parameters.rate_of_inclination_angle
        c_uc = self.orbit_parameters.amplitude_of_cosine_harmonic_correction_term_to_argument_of_latitude
        c_us = self.orbit_parameters.amplitude_of_sine_harmonic_correction_term_to_argument_of_latitude
        c_rc = self.orbit_parameters.amplitude_of_cosine_harmonic_correction_term_to_orbit_radius
        c_rs = self.orbit_parameters.amplitude_sine_harmonic_correction_term_to_orbit_radius
        c_ic = self.orbit_parameters.amplitude_of_cosine_harmonic_correction_term_to_angle_of_inclination
        c_is = self.orbit_parameters.amplitude_of_sine_harmonic_correction_term_to_angle_of_inclination

        # Semi-major axis
        a = sqrt_a ** 2
        # Mean motion
        n_0 = np.sqrt(mu / a ** 3)
        # TODO add the satellite bias to the time calculation
        t_oe = time_gps2seconds_of_week(np.float64(self.orbit_parameters.time_of_ephemeris.to_gpst_seconds()))
        # TODO add the receiver bias to the time calculation
        t = time_gps2seconds_of_week(np.float64(gps_time.to_gpst_seconds()))
        tk = gps_seconds_wrap(t - t_oe)
        # Corrected mean motion
        n = n_0 + delta_n
        # Mean anomaly
        m_k = m_0 + n * tk
        # Kepler to solve for Eccentric Anomaly
        ea = np.array([m_k, 0.0, 0.0, 0.0], dtype=np.float64)
        for i in range(1, 4):
            ea[i] = ea[i-1] + (m_k - ea[i-1] + e * np.sin(ea[i-1])) / (1.0 - e * np.cos(ea[i-1]))
        e_k = ea[3]
        # True anomaly
        vk = 2.0 * np.atan(np.sqrt((1.0+e) / (1.0-e)) * np.tan(e_k /2.0))
        # Argument of latitude
        phi_k = vk + omega
        phi_k_2 = phi_k * 2.0

        # Second harmonic perturbations
        # argument_of_latitude_correction
        delta_u_k = c_us * np.sin(phi_k_2) + c_uc * np.cos(phi_k_2)
        # radius_correction
        delta_r_k = c_rs * np.sin(phi_k_2) + c_rc * np.cos(phi_k_2)
        # inclination_correction
        delta_i_k = c_is * np.sin(phi_k_2) + c_ic * np.cos(phi_k_2)

        # corrected_argument_latitude
        u_k = phi_k + delta_u_k
        # corrected_radius
        r_k = a * (1.0 - e * np.cos(e_k)) + delta_r_k
        # corrected_inclination
        i_k = i_0 + delta_i_k + idot * tk

        plane_x = r_k * np.cos(u_k)
        plane_y = r_k * np.sin(u_k)

        # Corrected longitude of the ascending node
        omega_k = omega_0 + (omega_dot - omega_dot_e) * tk - omega_dot_e * t_oe

        x = plane_x * np.cos(omega_k) - plane_y * np.cos(i_k) * np.sin(omega_k)
        y = plane_x * np.sin(omega_k) + plane_y * np.cos(i_k) * np.cos(omega_k)
        z = plane_y * np.sin(i_k)

        position = np.array([x, y, z], dtype=np.float64)

        # eccentric_anomaly_rate
        e_dot_k = n / (1.0 - e * np.cos(e_k))
        # true_anomaly_rate
        v_dot_k = e_dot_k * (np.sqrt(1.0 - e ** 2) / (1.0 - e * np.cos(e_k)))

        # corrected_inclination_angle_rate
        di_k_dt = idot + 2.0 * v_dot_k * (c_is * np.cos(phi_k_2)- c_ic * np.sin(phi_k_2))
        # corrected_argument_latitude_rate
        u_dot_k = v_dot_k + 2.0 * v_dot_k * (c_us * np.cos(phi_k_2)- c_uc * np.sin(phi_k_2))
        # corrected_argument_latitude_rate
        r_dot_k = e * a * e_dot_k * np.sin(e_k) + 2.0 * v_dot_k * (c_rs * np.cos(phi_k_2)- c_rc * np.sin(phi_k_2))

        # longitude_ascending_node_rate
        omega_dot_k = omega_dot - omega_dot_e

        # Plane velocity
        x_dot_prime = r_dot_k * np.cos(u_k) - r_k * u_dot_k * np.sin(u_k)
        y_dot_prime = r_dot_k * np.sin(u_k) + r_k * u_dot_k * np.cos(u_k)

        x_velocity = (
            - plane_x * omega_dot_k * np.sin(omega_k)
            + x_dot_prime * np.cos(omega_k)
            - y_dot_prime * np.sin(omega_k) * np.cos(i_k)
            - plane_y * (omega_dot_k * np.cos(omega_k) * np.cos(i_k)
                        - di_k_dt * np.sin(omega_k) * np.sin(i_k))
        )
        y_velocity = (
            plane_x * omega_dot_k * np.cos(omega_k)
            + x_dot_prime * np.sin(omega_k)
            + y_dot_prime * np.cos(omega_k) * np.cos(i_k)
            - plane_y * (omega_dot_k * np.sin(omega_k) * np.cos(i_k)
                        + di_k_dt * np.cos(omega_k) * np.sin(i_k))
        )
        z_velocity = y_dot_prime * np.sin(i_k) + plane_y * di_k_dt * np.cos(i_k)

        velocity = np.array([x_velocity, y_velocity, z_velocity], dtype=np.float64)

        return position, velocity


    def parameters(self) -> GpsOrbitalParameters:
        return self.orbit_parameters