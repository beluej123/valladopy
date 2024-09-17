"""
Collection of Vallado solutions to examples and problems.

Notes:
----------
    This file is organized with each example as a function; i.e. function name:
        test_ex6_3_one_tan_burn().
    
    Supporting functions for the test functions below, may be found in other
        files, for example vallad_func.py, astro_time.py, kepler.py etc...
        Also note, the test examples are collected right after this document
        block.  However, the example test functions are defined/enabled at the
        end of this file.  Each example function is designed to be stand-alone,
        but, if you use the function as stand alone you will need to copy the
        imports...

    Reminder to me; cannot get black formatter to work within VSCode,
        so in terminal type; black *.py.
    Reminder to me; VSCode DocString, Keyboard shortcut: ctrl+shift+2.
    
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally angles are saved in [rad], distance [km].

References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
    [4] Vallado, David A., (2022, 5th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
"""

import math

import numpy as np

import astro_time
import vallado_func as vfunc  # Vallado functions
from kepler import coe2rv, eccentric_to_true, findTOF, findTOF_a, kep_eqtnE


def test_prb2_7_tof() -> None:
    """
    Find time of flight (tof). Vallado [2], problem 2.7, p.128.

    Input Parameters:
    ----------
        None
    Returns:
    -------
        None
    Notes:
    -------
        Interplanetary missions with patched conic in Vallado [2] chapter 12.
        Note Vallado [2], tof, section 2.8, p.126, algorithm 11.
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


def test_prb2_7a_tof(plot_sp=False) -> None:
    """
    Find time of flight (tof) and orbit parameter limits.
    Note Vallado, problem 2.7, p.128; tof, section 2.8, p.126, algorithm 11.
    Note BMWS, sma as a function of sp, section 5.4.2, p.204.

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


def test_ex5_5_planetLocation():
    """
    Find planet location. Vallado [2], example 5-5, pp.297, algorithm 33, pp.296.
    Find planet location. Vallado [4], example 5-5, pp.304, algorithm 33, pp.303.
    TODO finish exercise
    Notes:
    ----------
    See kepler.py Notes for list of orbital element nameing definitions.
    https://ssd.jpl.nasa.gov/planets/approx_pos.html
    Horizons on-line look-up https://ssd.jpl.nasa.gov/horizons/app.html#/
    From Horizons on-line:
    2449493.333333333 = A.D. 1994-May-20 20:00:00.0000 TDB
    EC= 4.831844662701981E-02 QR= 4.951499586021582E+00 IN= 1.304648490975239E+00
    OM= 1.004706325510906E+02 W = 2.751997729775426E+02 Tp=  2451319.928584063426
    N = 8.308901661461704E-02 MA= 2.082299968639316E+02 TA= 2.057443313085129E+02
    A = 5.202895410205566E+00 AD= 5.454291234389550E+00 PR= 4.332702620248230E+03
    """
    deg2rad = math.pi / 180  # used multiple times
    rad2deg = 180 / math.pi  # used multiple times

    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    mu_sun_au = mu_sun_km / (au**3)  # [au^3/s^2], unit conversion
    # print(f"mu_sun_au= {mu_sun_au}")

    # Vallado [2] equivilent of Curtis [3] p.276, eqn 5.48:
    # parameters of julian_date(yr, mo, d, hr, minute, sec, leap_sec=False)
    year, month, day, hour, minute, second = 1994, 5, 20, 20, 0, 0
    jd = astro_time.julian_date(
        yr=year, mo=month, d=day, hr=hour, minute=minute, sec=second
    )

    # centuries since J2000, Curtis p.471, eqn 8.93a
    # julian centuries of tdb (barycentric dynamic time)
    jd_tdb = (jd - 2451545) / 36525
    # print(f"jd= {jd}, jd_cent={jd_tbd}")

    # 2024-09-15, only jupiter (planet_id=4) in coefficients table so far
    J2000_coefs = vfunc.planet_ele_1(planet_id=4, au_units=True, rad_units=False)
    # coeffs format; x0*t_TDB^0 + x1*t_TDB^1 + x2*t_TDB^2 + ...
    #   time, t_tdb = julian centuries of barycentric dynamical time
    x1 = np.arange(5)  # exponent values; power series
    x2 = np.full(5, jd_tdb)  # base time value
    x3 = x2**x1  # time multiplier series

    sma = np.sum(J2000_coefs[0, :] * x3)  # [au]
    ecc = np.sum(J2000_coefs[1, :] * x3)  # [--]
    incl_deg = np.sum(J2000_coefs[2, :] * x3)  # [deg]
    raan_deg = np.sum(J2000_coefs[3, :] * x3)  # [deg]
    w_bar_deg = np.sum(J2000_coefs[4, :] * x3)  # [deg]
    L_bar_deg = np.sum(J2000_coefs[5, :] * x3)  # [deg]

    incl_rad = incl_deg * deg2rad
    raan_rad = raan_deg * deg2rad
    w_bar_rad = w_bar_deg * deg2rad
    L_bar_rad = L_bar_deg * deg2rad

    print(f"sma= {sma} [au]")
    print(f"ecc= {ecc}")
    print(f"incl= {incl_deg} [deg]")
    print(f"raan= {raan_deg} [deg]")
    print(f"w_bar= {w_bar_deg} [deg]")  # longitude of periapsis
    print(f"L_bar= {L_bar_deg} [deg]")

    M_deg = L_bar_deg - w_bar_deg  # [deg] mean angle/anomaly
    M_rad = M_deg * deg2rad
    w_deg = w_bar_deg - raan_deg  # [deg] argument of periapsis (aka aop, or arg_p)
    w_rad = w_deg * deg2rad
    print(f"\nM_deg= {M_deg} [deg]")
    print(f"w_deg= {w_deg} [deg]")

    E_rad = kep_eqtnE(M=M_rad, e=ecc)
    E_deg = E_rad * rad2deg
    TA_rad = eccentric_to_true(E=E_rad, e=ecc) # no quadrent ambiguity
    # below, Curtis [3], p.160, eqn 3.13b.
    # TA_rad = 2 * math.atan(math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E_rad / 2))
    TA_deg = TA_rad * rad2deg
    
    print(f"E_deg= {E_deg} [deg]")
    print(f"TA_deg= {TA_deg} [deg]")

    sp = sma * (1 - ecc**2)
    print(f"sp= {sp} [au]")

    # function inputs, coe2rv(p, ecc, inc, raan, aop, anom, mu=Earth.mu)
    r_vec, v_vec = coe2rv(
        p=sp,
        ecc=ecc,
        inc=incl_rad,
        raan=raan_rad,
        aop=w_rad,
        anom=TA_rad,
        mu=mu_sun_au,
    )
    r_vec = np.ravel(r_vec)  # convert column array to row vector
    v_vec = np.ravel(v_vec)  # convert column array to row vector
    print(f"r_vec= {r_vec} [au]")
    print(f"v_vec= {v_vec*86400} [au/day]")


def test_ex6_1_hohmann():
    """
    Hohmann Transfer, Vallado, example 6-1, p326; uses p.325, algirithm 36.
    Hohmann uses one central body for the transfer ellipse.
    Interplanetary missions use the patched conic; 3 orbit types:
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival

    Interplanetary missions with patched conic in chapter 12.
    """
    print(f"\nTest Hohmann Transfer, Vallado, example 6.1")
    # define constants
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    # define inputs for hohmann transfer
    r1 = r_earth + 191.34411  # [km]
    r2 = r_earth + 35781.34857  # [km]

    # vallado hohmann transfer
    tof_hoh, ecc_trans, dv_total = vfunc.val_hohmann(
        r_init=r1, r_final=r2, mu_trans=mu_earth_km
    )
    print(f"Hohmann transfer time: {(tof_hoh/2):0.8g} [s], {tof_hoh/(2*60):0.8g} [m]")
    print(f"Transfer eccentricity, ecc_trans= {ecc_trans}")
    print(f"Total transfer delta-v, dv_total= {dv_total}")


def test_ex6_2_bielliptic():
    """
    Bi-Elliptic Transfer, Vallado, example 6-2, p327.
    Bi-Elliptic uses one central body for the transfer ellipse.
    Two transfer ellipses; one at periapsis, the other at apoapsis.
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival

    Notes:
    ----------
        Note, interplanetary missions use the patched conic; 3 orbit types:
        For interplanetary missions with patched conic see chapter 12.
    """
    print("\nTest Vallado Bi-Elliptic Transfer, example 6-2")  # temporary print
    # define constants
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    # define inputs; bi-elliptic transfer
    r_1 = r_earth + 191.34411  # [km]
    r_b = r_earth + 503873  # [km] point b, beyond r2
    r_2 = r_earth + 376310  # [km]

    # vallado bi-elliptic transfer
    # val_bielliptic(r_init: float, r_b: float, r_final: float, mu_trans: float):
    a_trans1, a_trans2, t_trans1, t_trans2, dv_total = vfunc.val_bielliptic(
        r_init=r_1, r_b=r_b, r_final=r_2, mu_trans=mu_earth_km
    )
    # extra: eccentricity not part of Vallado example:
    #   calculate eccentricity of the two transfer ellipses
    ecc_1 = abs(r_b - r_1) / (r_1 + r_b)  # elliptical orbits only
    ecc_2 = abs(r_b - r_2) / (r_2 + r_b)  # elliptical orbits only

    print(f"Ellipse1 transfer; semi-major axis, a_trans1= {a_trans1:.8g} [km]")
    print(f"Ellipse1 eccentricity, ecc_1= {ecc_1:.6g}")
    print(f"Ellipse2 transfer; semi-major axis, a_trans2= {a_trans2:.8g} [km]")
    print(f"Ellipse2 eccentricity, ecc_2= {ecc_2:.6g}")

    tof_total = t_trans1 + t_trans2  # tof=time of flight
    print(
        f"Bi-Elliptic tof: {tof_total:0.8g} [s], "
        f"{tof_total/(60):.8g} [min], "
        f"{tof_total/(3600):0.8g} [hr]"
    )


def test_ex6_3_one_tan_burn():
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
    r1_mag = r_earth + 35781.34857  # [km], example 6-3, geosynchronous
    nu_trans_b = 160  # [deg], example 6-3
    # uncomment below to test moon trajectory
    # r1_mag = r_earth + 376310  # [km], vallado p.336, tbl 6-1, moon
    # nu_trans_b = 175  # [deg], vallado p.336, tbl 6-1, moon

    ecc_tx, sma_tx, tof_tx, dv_0, dv_1 = vfunc.val_one_tan_burn(
        r_init=r0_mag, r_final=r1_mag, nu_trans_b=nu_trans_b, mu_trans=mu_earth_km
    )
    print(f"transfer eccentricity, ecc_tx= {ecc_tx:.6g}")
    print(f"transfer semi-major axis, sma_tx= {sma_tx:.6g}")
    print(f"delta velocity at v0, dv_0= {dv_0:.6g}")
    print(f"delta velocity at v1, dv_1= {dv_1:.6g}")
    print(f"one-tangent, total delta velocity, {dv_0+dv_1:.6g}")
    print(
        f"tof_trans= {tof_tx:0.8g} [sec], {tof_tx/60:0.8g} [min], {(tof_tx/(60*60)):0.8g} [hr]"
    )
    return None  # one_tan_burn_ex6_3()


# Test functions and class methods are called here.
if __name__ == "__main__":
    # test_prb2_7_tof()  # test tof, problem 2-7
    # test_prb2_7a_tof(plot_sp=False)  # test tof; plot sma vs. sp
    test_ex5_5_planetLocation()  # test planet location
    # test_ex6_1_hohmann()  # test hohmann transfer, example 6-1
    # test_ex6_2_bielliptic()  # test bi-elliptic transfer, example 6-2
    # test_ex6_3_one_tan_burn()  # test one-tangent transfer, example 6-3
