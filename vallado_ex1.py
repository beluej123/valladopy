# Vallado_examples
# Cannot get black formatter to work within VSCode; so in terminal type; black vallado_func.py
import numpy as np

import vallado_func as vfunc  # Vallado functions


def hohmann_ex6_1():
    """Vallado, Hohmann Transfer, example 6-1, p326.
    Hohmann uses one central body for the transfer ellipse.
    Interplanetary missions use the patched conic; 3 orbit types:
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival

    Interplanetary missions with patched conic in chapter 12.
    """
    # define constants
    r_earth = 6378.137  # [km]
    mu_earth = 3.986012e5  # [km^3 / s^2] gravatational constant, earth
    mu_sun = 1.327e11  # [km^3 / s^2] gravatational constant, sun

    # define inputs for hohmann transfer
    r1 = r_earth + 191.34411  # [km]
    r2 = r_earth + 35781.34857  # [km]
    mu_trans = mu_earth  # [km^3 / s^2] gravatational constant

    vfunc.val_hohmann(r1, r2, mu_trans)  # vallado hohmann transfer


def bielliptic_ex6_2():
    """Vallado, Bi-Elliptic Transfer, example 6-2, p327.
    Bi-Elliptic uses one central body for the transfer ellipse.
    Interplanetary missions use the patched conic; 3 orbit types:
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival

    Interplanetary missions with patched conic in chapter 12.
    """
    # define constants
    r_earth = 6378.137  # [km]
    mu_earth = 3.986012e5  # [km^3 / s^2] gravatational constant, earth
    mu_sun = 1.327e11  # [km^3 / s^2] gravatational constant, sun

    # define inputs; bi-elliptic transfer
    r1 = r_earth + 191.34411  # [km]
    rb = r_earth + 503873  # [km] point b, beyond r2
    r2 = r_earth + 376310  # [km]
    mu_trans = mu_earth  # [km^3 / s^2] gravatational constant

    vfunc.val_bielliptic(r1, rb, r2, mu_trans)  # vallado bi-elliptic transfer


def one_tan_burn_ex6_3():
    """Vallado One-Tangent Burn Transfer, example 6-3, p334.
    One-Tangent Burn uses one central body for the transfer ellipse.
    Interplanetary missions with patched conic in chapter 12.
    """
    # define constants
    r_earth = 6378.137  # [km]
    mu_earth = 3.986012e5  # [km^3 / s^2] gravatational constant, earth
    mu_sun = 1.327e11  # [km^3 / s^2] gravatational constant, sun

    # define inputs; one-tangent transfer (two burns/impulses)
    r1 = r_earth + 191.34411  # [km]
    # r2 = r_earth + 35781.34857  # [km], example 6-3
    r2 = r_earth + 376310  # [km], table 6-1, moon
    # nu_trans_b = 160  # [degrees], example 6-3
    nu_trans_b = 175  # [degrees], table 6-1, moon
    mu_trans = mu_earth  # [km^3 / s^2] gravatational constant

    vfunc.val_one_tan_burn(r1, r2, nu_trans_b, mu_trans)


import math

from kepler import findTOF


def test_tof_prob2_7() -> None:
    """
    Find time of flight. Vallado section 2, problem 2.7, p.128.
    Note:
    The problem gives the value for sp (semi-parameter, aka p), but in practice
    it is not clear, to me, how to chose sp because ecc (eccentricity) and or
    sma (semi-major axis) are parameter I can intuitively grasp.  And in that case
    the tof routine does not need to internally calculate sma...
    """
    print(f"\nVallado test time-of-flight, prob 2.7:")
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    r0_vec = np.array([-2574.9533, 4267.0671, 4431.5026])  # [km]
    r1_vec = np.array([2700.6738, -4303.5378, -4358.2499])  # [km/s]
    sp = 6681.571  # [km] semi-parameter (aka p, also, aka semi-latus rectum)
    r0_mag = np.linalg.norm(r0_vec)
    r1_mag = np.linalg.norm(r1_vec)
    # TODO calculate delta true anomalies...
    cosdv = np.dot(r0_vec.T, r1_vec) / (
        r0_mag * r1_mag
    )  # note r0_vec.T = transpose of r0_vec
    print(f"delta true anomaly's, {math.acos(cosdv)*180/math.pi:.6g} [deg]")

    tof = findTOF(r0=r0_vec, r=r1_vec, p=sp, mu=mu_earth_km)
    print(f"time of flight, tof= {tof:.8g} [s]")
    return


# Main code. Functions and class methods are called from main.
if __name__ == "__main__":
    # hohmann_ex6_1()  # hohmann transfer, vallado example 6-1
    # bielliptic_ex6_2()  # bi-elliptic transfer, vallado example 6-2
    # one_tan_burn_ex6_3()  # one-tangent transfer, vallado example 6-3
    test_tof_prob2_7()  #
