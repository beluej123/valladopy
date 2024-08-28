# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 22:50:18 2016, @author: Alex
Edits 2024-08-21 +, Jeff Belue.

Notes:
----------
    TODO, This file is organized ...
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally angles are saved in [rad], distance [km].
    
    Supporting functions for the test functions below, may be found in other
        files, for example vallad_func.py, astro_time.py, kepler.py etc...
        Also note, the test examples are collected right after this document
        block.  However, the example test functions are defined/enabled at the
        end of this file.  Each example function is designed to be stand-alone,
        but, if you use the function as stand alone you will need to copy the
        imports...

References
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""
import math
import unittest
from test import support

import numpy as np

import solarsys as ssys  # holds constants/parameters
from kepler import coe2rv, kepler


class KeplerExamplesFromBookTestCase(unittest.TestCase):

    def test_example_2_1_Keplers_Equation(self):
        M = 235.4
        e = 0.4
        M_rad = math.radians(M)
        actual = math.degrees(kepler.kep_eqtnE(M_rad, e))
        expected = 220.512074767522
        self.assertAlmostEqual(actual, expected, places=8)

    def test_example_2_2_Keplers_Equation_Parabolic(self):
        del_t = 53.7874
        p = 25512.0

        del_t_sec = del_t * 60.0
        actual = kepler.kep_eqtnP(del_t_sec, p)
        expected = 0.817751
        self.assertAlmostEqual(actual, expected, places=6)

    def test_example_2_3_Keplers_Equation_Hyperbolic(self):
        M = 235.4
        e = 2.4
        M_rad = math.radians(M)
        actual = kepler.kep_eqtnH(M_rad, e)
        expected = 1.601376144
        self.assertAlmostEqual(actual, expected, places=8)

    def test_example_2_4_Keplers_Problem(self):
        r_ijk = np.matrix((1131.340, -2282.343, 6672.423)).T
        v_ijk = np.matrix((-5.64305, 4.30333, 2.42879)).T
        del_t = 40.0

        del_t_sec = del_t * 60.0

        (actual_pos, actual_vel) = kepler.kepler(r_ijk, v_ijk, del_t_sec)
        expected_pos = np.matrix((-4219.7527, 4363.0292, -3958.7666)).T
        expected_vel = np.matrix((3.689866, -1.916735, -6.112511)).T
        self.assertAlmostEqual(actual_pos.item(0), expected_pos.item(0), places=3)
        self.assertAlmostEqual(actual_pos.item(1), expected_pos.item(1), places=3)
        self.assertAlmostEqual(actual_pos.item(2), expected_pos.item(2), places=3)
        self.assertAlmostEqual(actual_vel.item(0), expected_vel.item(0), places=5)
        self.assertAlmostEqual(actual_vel.item(1), expected_vel.item(1), places=5)
        self.assertAlmostEqual(actual_vel.item(2), expected_vel.item(2), places=5)

    def test_example_2_5_find_orbital_elements(self):
        r_ijk = np.matrix((6524.834, 6862.875, 6448.296)).T
        v_ijk = np.matrix((4.901327, 5.533756, -1.976341)).T
        (pe, ae, ee, ie, raane, aope, tanome) = (
            11067.790,
            36127.343,
            0.832853,
            math.radians(87.870),
            math.radians(227.898),
            math.radians(53.38),
            math.radians(92.335),
        )
        (pa, aa, ea, ia, raana, aopa, tanoma) = kepler.rv2coe(r_ijk, v_ijk)
        self.assertAlmostEqual(pa, pe, places=1)
        self.assertAlmostEqual(aa, ae, places=1)
        self.assertAlmostEqual(ea, ee, places=6)
        self.assertAlmostEqual(ia, ie, places=3)
        self.assertAlmostEqual(raana, raane, places=3)
        self.assertAlmostEqual(aopa, aope, places=3)
        self.assertAlmostEqual(tanoma, tanome, places=3)
        return


# ----------
def test_coe2rv():
    """
    Vallado pp.119, example 2-6.
    TODO update test goal

    Given:

    Find:

    Notes:
    ----------
        Implements Vallado pp.118, coe2rv(), algorithm 10.
        Note Vallado functions: pp. 296, planetRV(), algotithm 33,

        For my code generally, angles in [rad], distances in [km]..

        Orbital elements identifiers (*=not used this function):
            sp     = [km] semi-parameter (aka p, parameter)
            *sma   = [km] semi-major axis (aka a)
            ecc    = [--] eccentricity
            incl   = [deg] inclination angle; to the ecliptic
            RAAN   = [deg] right ascension of ascending node (aka capital W)
            aop    = [deg] arguement of periapsis
            *w_hat = [deg] longitude of periapsis
            anom   = [deg] true angle/anomaly
            L      = [deg] mean longitude

        References: see list at file beginning.
    """
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    mu_mars_km = 4.305e4  # [km^3/s^2], Vallado p.1041, tbl.D-3

    # define inputs and convert to [rad]
    sp = 11067.98  # [km]
    ecc = 0.83285
    incl_deg = 87.87  # [deg]
    raan_deg = 227.89  # [deg]
    aop_deg = 53.38  # [deg]
    anom_deg = 92.335  # [deg]

    incl = incl_deg * math.pi / 180  # [rad]
    raan = raan_deg * math.pi / 180  # [rad]
    aop = aop_deg * math.pi / 180  # [rad]
    anom = anom_deg * math.pi / 180  # [rad]

    r_vec, v_vec = coe2rv(
        p=sp, ecc=ecc, inc=incl, raan=raan, aop=aop, anom=anom, mu=ssys.Earth.mu
    )
    print(f"r_vec= {r_vec} [km]")
    print(f"v_vec= {v_vec} [km]")

    return None


def test_main():
    # 2024-08-24, JBelue until I spend the time to learn python test,
    #   I will manually generate test.
    # support.run_unittest(KeplerExamplesFromBookTestCase)
    return None


if __name__ == "__main__":
    # test_main() # python test scripts; figure out sometime 2024-August
    test_coe2rv()  #
