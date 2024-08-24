"""
Collection of Vallado [2] solutions to examples and problems.

Notes:
----------
    Reminder to me; cannot get black formatter to work within VSCode,
        so in terminal type; black *.py
    Reminder to me; VSCode DocString, Keyboard shortcut: ctrl+shift+2

References
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

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
    """
    Test Vallado One-Tangent Burn Transfer, example 6-3, p334.
    
    Input Parameters:
    ----------
        None
    Returns:
    -------
        None
    Notes:
    -------
        One-Tangent Burn uses one central body for the transfer ellipse.
        Interplanetary missions with patched conic in chapter 12.

        References: see list at file beginning.
    """
    print(f"\nTest one-tangent burn, Vallado example 6-3:")
    # constants
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    mu_sun_au = mu_sun_km / (au**3)  # unit conversion
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    # define inputs; one-tangent transfer (two burns/impulses)
    r0_mag = r_earth + 191.34411  # [km], example 6-3
    r1_mag = r_earth + 35781.34857  # [km], example 6-3
    nu_trans_b = 160  # [deg], example 6-3

    # r1 = r_earth + 376310  # [km], table 6-1, moon
    # nu_trans_b = 175  # [deg], table 6-1, moon

    # val_one_tan_burn(r_init: float, r_final: float, nu_trans_b: float, mu_trans: float)
    ecc_tx, sma_tx, tof_tx, dv_0, dv_1 = vfunc.val_one_tan_burn(
        r_init=r0_mag, r_final=r1_mag, nu_trans_b=nu_trans_b, mu_trans=mu_earth_km
    )
    print(f"transfer eccentricity, ecc_tx= {ecc_tx:.6g}")
    print(f"transfer semi-major axis, sma_tx= {sma_tx:.6g}")
    print(f"delta velocity at v0, dv_0= {dv_0:.6g}")
    print(f"delta velocity at v1, dv_1= {dv_1:.6g}")
    print(
        f"tof_trans= {tof_tx:0.8g} [sec], {tof_tx/60:0.8g} [min], {(tof_tx/(60*60)):0.8g} [hr]"
    )
    return None  # one_tan_burn_ex6_3()


import math

from kepler import findTOF, findTOF_a


def test_tof_prob2_7() -> None:
    """
    Find time of flight (tof). Vallado, problem 2.7, p.128.
    
    Input Parameters:
    ----------
        None
    Returns:
    -------
        None
    Notes:
    -------
        Interplanetary missions with patched conic in Vallado chapter 12.
        Note Vallado, tof, section 2.8, p.126, algorithm 11.
        It is useful to understand the limits on orbit definition; see
            test_tof_prob2_7a.

        Reference Details: see list at file beginning.
    """
    print(f"\nVallado time-of-flight, prob 2.7:")
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
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


def test_tof_prob2_7a(plot_sp=False) -> None:
    """
    Find time of flight (tof) and orbit parameter limits.
    Note Vallado [2], problem 2.7, p.128; tof, section 2.8, p.126, algorithm 11.
    Note BMWS [1], sma as a function of sp, section 5.4.2, p.204.

    Notes:
    ----------
        Problem statement gives a value for sp (semi-parameter, aka p), thus
        defining orbital energy.  Since sp is given ths routine explores the
        limits of orbit definition by looking at ellipse limits.

    Assume r0 in the vicinity of earth; thus assume
    Choose v0
    """
    from vallado_func import plot_sp_vs_sma

    print(f"\nVallado time-of-flight, prob 2.7a:")
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    r0_vec = np.array([-2574.9533, 4267.0671, 4431.5026])  # [km]
    r1_vec = np.array([2700.6738, -4303.5378, -4358.2499])  # [km/s]
    sp = 6681.571  # [km] semi-parameter (aka p, also, aka semi-latus rectum)

    r0_mag = np.linalg.norm(r0_vec)
    r1_mag = np.linalg.norm(r1_vec)
    # calculate delta true anomalies...
    # note r0_vec.T = transpose of r0_vec
    cosdv = np.dot(r0_vec.T, r1_vec) / (r0_mag * r1_mag)
    print(f"semi-parameter, sp= {sp:.6g} [km]")
    delta_nu = math.acos(cosdv)
    print(f"delta true anomaly's, {delta_nu*180/math.pi:.6g} [deg]")

    tof, sma, sp_i, sp_ii = findTOF_a(r0=r0_vec, r1=r1_vec, p=sp, mu=mu_earth_km)
    ecc = math.sqrt(1 - sp / sma)
    print(f"semi-major axis, sma= {sma:.8g}")
    print(f"eccemtricity, ecc= {ecc:.8g}")
    print(f"time of flight, tof= {tof:.8g} [s]")

    # inform user of sp limits
    print_text1 = f"sp limits; sp_i= {sp_i:.6g}, sp= {sp:.6g}, sp_ii= {sp_ii:.6g}"
    if sp > sp_i and sp < sp_ii:
        print(f"ellipse, {print_text1}")
    elif sp == sp_ii or sp == sp_i:
        print(f"parabola, {print_text1}")
    elif sp > sp_ii:
        print(f"hyperbola, {print_text1}")
    else:
        print(f"sp < sp_i; not sure orbit type.")

    if plot_sp == True:
        # plot_sp=True, to see possible range of orbit parameters plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker.
        # note, since sma may be near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(
            r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=delta_nu, sp=sp, clip1=True
        )
    return  # test_tof_prob2_7a()


# Main code. Functions and class methods are called from main.
if __name__ == "__main__":
    # hohmann_ex6_1()  # test hohmann transfer, example 6-1
    # bielliptic_ex6_2()  # test bi-elliptic transfer, example 6-2
    one_tan_burn_ex6_3()  # test one-tangent transfer, example 6-3
    # test_tof_prob2_7()  # test tof, problem 2-7
    # test_tof_prob2_7a(plot_sp=False)  # test tof; plot sma vs. sp
