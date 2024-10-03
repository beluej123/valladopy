"""
Collection of Vallado python solutions to examples and problems; called as test_...

Notes:
----------
    This file is organized with each example as a test_xxx() function; i.e. function name:
        test_ex6_3_one_tan_burn().
    
    Supporting functions for the test functions below, may be found in other
        files, for example vallad_func.py, astro_time.py, kepler.py etc...
        Also note, the test examples are collected right after this document
        block.  However, the example test functions are defined/enabled at the
        end of this file.  Each example function is designed to be stand-alone,
        but, if you use the function as stand alone you will need to copy the
        imports...

    Reminder to me; cannot get black formatter to work within VSCode,
        so in VSCode terminal type; "black *.py" or "black filename.py"
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
from datetime import datetime as dt

import numpy as np

import astro_time
import vallado_func as vfunc  # Vallado functions collection.
from kepler import coe2rv, eccentric_to_true, findTOF, findTOF_a, kep_eqtnE, keplerUni
from vallado_func import ecliptic_angle, planet_rv, plot_sp_vs_sma


def test_ex2_1KeplerE():
    """
    Kepler equation eclipse solution; mean anomaly, eccentricity -> E
        Vallado [4], example 2-1, p.66, algorithm 2, pp.65.
    Given:
        M   : float, [rad] mean anomaly; must, -2*pi <= M <= +2*pi
        e   : float, [--] eccentricity
        default Newton-Raphson convergence tolerance

    Returns:
    -------
    none
    """
    print(f"\nTest Kepler ellipse solution for E:")
    # given
    M_rad = 235.4 * (math.pi / 180)  # [rad]
    ecc = 0.4
    E_ = kep_eqtnE(M=M_rad, e=ecc)
    print(f"Eccentric anomaly, E_= {E_:.6g} [rad], {E_*180/math.pi:.6g} [deg]")

    return None


def test_ex2_4_keplerUni():
    """
    Kepler propagation, given  r0_vec, v0_vec, dt -> r1_vec, v1_vec.
        Universal formulation.  Vallado [4], example 2-4, p.96,
        from algorithm 8, pp.94.

    Notes:
    ----------
    """
    print(
        f"\nTest Kepler, universal formulation, Vallado [4] example 2-4; r0, v0, dt -> r1, v1:"
    )
    np.set_printoptions(precision=6)  # numpy, set vector printing size

    # given
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    r0_vec = np.array([1131.340, -2282.343, 6672.423])  # [km]
    v0_vec = np.array([-5.64305, 4.30333, 2.42879])  # [km/s]
    delta_t = 40 * 60  # [s]
    # kepler universal formulation
    # function parameters, keplerUni(r0, v0, dt, mu=Earth.mu, tol=1e-6):
    r1_vec, v1_vec = keplerUni(r0_vec, v0_vec, dt=delta_t, mu=mu_earth_km, tol=1e-6)
    print(f"r1_vec= {r1_vec} [km]")
    print(f"v1_vec= {v1_vec} [km]")

    return


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
        Interplanetary missions with patched conic in Vallado [2], chapter 12.
        Note Vallado [2], tof, section 2.8, p.126, algorithm 11.
        It is useful to understand the limits on orbit definition; see
            test_tof_prob2_7a.

        Reference Details: see list at file beginning.
    """
    print(f"\nTest Vallado [2] time-of-flight, prob 2.7:")
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado [2] p.1041, tbl.D-3

    r0_vec = np.array([-2574.9533, 4267.0671, 4431.5026])  # [km]
    r1_vec = np.array([2700.6738, -4303.5378, -4358.2499])  # [km]
    sp = 6681.571  # [km] semi-parameter (aka p, also, aka semi-latus rectum)
    r0_mag = np.linalg.norm(r0_vec)
    r1_mag = np.linalg.norm(r1_vec)
    # TODO calculate delta true anomalies...
    cosdv = np.dot(r0_vec.T, r1_vec) / (
        r0_mag * r1_mag
    )  # note r0_vec.T = transpose of r0_vec
    print(f"Delta true anomaly's, {math.acos(cosdv)*180/math.pi:.6g} [deg]")

    tof = findTOF(r0=r0_vec, r=r1_vec, p=sp, mu=mu_earth_km)
    print(f"Time of flight, tof= {tof:.8g} [s]")
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

    print(f"\nTest Vallado [2] time-of-flight, prob 2.7a:")
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado [2] p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado [2] p.1041, tbl.D-3

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


def test_ex5_1_sunPosition():
    """
    Find geocentric position of sun vector.
    Vallado [4] sunPosition(), example 5-1, pp.286.
    Vallado [4], algorithm 29, pp.285

    Notes:
    ----------
        This is a low precision solution; see Vallado [4] section 5.1.1, p283.
        This effort debugged the function sunPosition() saved in vallado_func.py.
    """
    print(f"\nTest Vallado [4] example 5-1, sun position: ")
    # the following date is UTC, not inclusive of delta_UT1~0.265s, and delta_AT~33s
    yr, mo, day, hr, min, sec = 2006, 4, 1, 23, 58, 54.816
    # thus
    yr, mo, day, hr, min, sec = 2006, 4, 2, 0, 0, 0
    print(f"\nSun Position. Vallado example 5-1:")
    sun_vec = vfunc.sunPosition(yr=yr, mo=mo, day=day, hr=hr, min=min, sec=sec)
    print(f"sun_vec= {sun_vec} [au]")

    return None


def test_ex5_2_sunRiseSet():
    """
    Find sunrise-sunset. Vallado [4] sunRiseSet(), example 5-2, pp.290.
    Vallado [4], algorithm 30, pp.289

    Notes:
    ----------
    This sunrise/sunset function is a low precision solution;
        note Vallado [4] section 5.1.1, p283.
    https://leancrew.com/all-this/2023/11/more-general-sunrise-sunset-plots/
    https://www.sunrisesunset.com/USA/Texas/
    This effort debugged the function sunRiseSet() saved in vallado_func.py.
    """
    print(f"\nTest Vallado [4] sunrise-sunset, example 5-2:")
    np.set_printoptions(precision=6)  # numpy, set vector printing size
    deg2rad = math.pi / 180  # used multiple times
    rad2deg = 180 / math.pi  # used multiple times

    # two choices for julian date; julian_date() and convTime().
    #   julian_date() calculates only the julian date, while
    #   convTime() calculates both julian date, julian centuries since J2000.0.
    yr, mo, day = 1996, 3, 23
    lat = 40.0  # site lattitude
    lon = 0.0  # site longitude

    UT_sunRise, UT_sunSet = vfunc.sunRiseSet(yr=yr, mo=mo, day=day, lat=lat, lon=lon)
    print(f"\nUT_sunRise= {UT_sunRise} [deg]")
    hour, min, sec = astro_time.dec_deg2hms(dec_deg=UT_sunRise)
    print(f"{hour}:{min}:{sec} [hr:min:sec]")

    print(f"UT_sunSet= {UT_sunSet} [deg]")
    hour, min, sec = astro_time.dec_deg2hms(dec_deg=UT_sunSet)
    print(f"{hour}:{min}:{sec} [hr:min:sec]")

    return None


def test_ex5_5_planetLocation():
    """
    Find planet location; essentially Vallado's PlanetRV(), algorithm 33.
        Vallado [2], example 5-5, pp.297, algorithm 33, pp.296.
        Vallado [4], example 5-5, pp.304, algorithm 33, pp.303.

    Notes:
    ----------
    NOTE Vallado [2] and [4] appendix D.4 tables DONOT correlate well with
        JPL Horizons outputs for Jupiter; well kind-da same ball park.
            1994, 5, 20, 20, 0, 0 # ex.5-5, reviewed equatorial, not ecliptic
            1979, 3, 5, 12, 5, 26 # ex.12-8, reviewed equatorial, not ecliptic
    This effort debugged the function planet_rv() saved in vallado_func.py.
    See kepler.py Notes for list of orbital element nameing definitions.

    https://ssd.jpl.nasa.gov/planets/approx_pos.html
    Horizons on-line look-up https://ssd.jpl.nasa.gov/horizons/app.html#/
    From Horizons on-line:
    Jupiter; heliocentric, equatorial, International Celestial Reference Frame (ICRF)
    2449493.333333333 = A.D. 1994-May-20 20:00:00.0000 TDB
    units below: [deg], [au], [days]
    EC= 4.831844662701981E-02, QR= 4.951499586021582E+00, IN= 1.304648490975239E+00
    OM= 1.004706325510906E+02, W = 2.751997729775426E+02, Tp=  2451319.928584063426
    N = 8.308901661461704E-02, MA= 2.082299968639316E+02, TA= 2.057443313085129E+02
    A = 5.202895410205566E+00, AD= 5.454291234389550E+00, PR= 4.332702620248230E+03
    EC=Eccentricity; QR=Periapsis distance (au); IN=Inclination w.r.t X-Y plane (deg);
    OM=Longitude of Ascending Node (deg); W=Argument of Perifocus (deg);
    Tp=Time of periapsis (Julian Day Number); N=Mean motion (deg/day);
    MA=Mean anomaly (deg); TA=True anomaly (deg); A=Semi-major axis (au);
    AD=Apoapsis distance (au); PR=Sidereal orbit period (day)

    units below: x,y,z [au]; vx,vy,vz [au/day]
    X= -4.064448682130308E+00, Y= -3.337082238471498E+00, Z=-1.331927194090224E+00
    VX= 5.342571281658752E-03, VY= -3.857849397745642E-03, VZ=-1.791799534510893E-03
    LT= 3.133179623650530E-02, RG= 5.424932350393856E+00, RR=-1.189710609786379E-03
    """
    print(f"\nTest planet position, Vallado [4] example 5-5:")
    print(f"  Jupiter:")
    np.set_printoptions(precision=6)  # numpy, set vector printing size
    deg2rad = math.pi / 180  # used multiple times
    rad2deg = 180 / math.pi  # used multiple times

    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    mu_sun_au = mu_sun_km / (au**3)  # [au^3/s^2], unit conversion
    # print(f"mu_sun_au= {mu_sun_au}")

    # Vallado [2] equivilent of Curtis [3] p.276, eqn 5.48:
    # convTime() convert time; always calculates julian date, but also
    #   calculates other time conversions; i.e. julian centuries since J2000.0.
    #   In this example the julian date is not used in calculations; just to print.
    #   Centuries since J2000, Curtis p.471, eqn 8.93a.
    year, month, day, hour, minute, second = 1994, 5, 20, 20, 0, 0  # ex.5-5; Jupiter
    # year, month, day, hour, minute, second = 1979, 3, 5, 12, 5, 26 # see ex.12-8; Jupiter
    # year, month, day, hour, minute, second = 1977, 9, 8, 9, 8, 17 # see ex.12-8; Earth
    jd, jd_cJ2000 = astro_time.jd_convTime(
        year, month, day, hour, minute, second, c_type=0
    )
    print(f"Ephemeris date, {year}-{month}-{day} {hour}:{minute}:{second}")
    print(f"jd= {jd:.10g}, jd_cent={jd_cJ2000:.10g}")

    """
    2024-09-20, not all planets in coefficients table so far
    planet_id : int, 0=mercury, 1=venus. 2=earth, 3=mars, 4=jupiter
                     5=saturn, 6=uranus, 7=neptune, 8=pluto
    """
    J2000_coefs = vfunc.planet_ele_1(planet_id=4)  # returns [deg] and [au]
    # coeffs format; x0*t_TDB^0 + x1*t_TDB^1 + x2*t_TDB^2 + ...
    #   time, t_tdb = julian centuries of barycentric dynamical time
    x1 = np.arange(5)  # exponent values; power series
    x2 = np.full(5, jd_cJ2000)  # base time value
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

    print(f"sma= {sma:.8g} [au]")
    print(f"ecc= {ecc:.8g}")
    print(f"incl= {incl_deg:.8g} [deg]")
    print(f"raan= {raan_deg:.8g} [deg]")
    print(f"w_bar= {w_bar_deg:.8g} [deg]")  # longitude of periapsis
    print(f"L_bar= {L_bar_deg:.8g} [deg]")

    M_deg = L_bar_deg - w_bar_deg  # [deg] mean angle/anomaly
    M_rad = M_deg * deg2rad
    w_deg = w_bar_deg - raan_deg  # [deg] argument of periapsis (aka aop, or arg_p)
    w_rad = w_deg * deg2rad
    print(f"\nM_deg= {M_deg:.8g} [deg]")
    print(f"w_deg= {w_deg:.8g} [deg]")

    E_rad = kep_eqtnE(M=M_rad, e=ecc)
    E_deg = E_rad * rad2deg
    # TA_rad below, no quadrent ambiguity; addresses near pi values
    TA_rad = eccentric_to_true(E=E_rad, e=ecc)
    # below, commented out, Curtis [3] solution, p.160, eqn 3.13b.
    # TA_rad = 2 * math.atan(math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E_rad / 2))
    TA_deg = TA_rad * rad2deg

    print(f"E_deg= {E_deg:.8g} [deg]")
    print(f"TA_deg= {TA_deg:.8g} [deg]")

    sp = sma * (1 - ecc**2)
    print(f"sp= {sp:.8g} [au]")

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
    v_vec = np.ravel(v_vec) * 86400  # convert, seconds to days
    print(f"\nEquatorial, Heliocentric, XYZ")
    print(f"r_vec= {r_vec} [au]")
    print(f"v_vec= {v_vec} [au/day]")
    print(f"r_vec= {r_vec*au} [km]")
    print(f"v_vec= {v_vec*au/86400} [km/s]")

    # rotate r_vec and v_vec from equatorial to ecliptic/heliocentric
    e_angle = vfunc.ecliptic_angle(jd_cJ2000)  # ecliptic angle
    r1_vec = r_vec @ vfunc.rot_matrix(angle=-e_angle * deg2rad, axis=0)
    v1_vec = v_vec @ vfunc.rot_matrix(angle=-e_angle * deg2rad, axis=0)
    print(f"ecliptic angle, e_angle= {e_angle:.8g} [deg]")
    print(f"\nEcliptic/Heliocentric, XYZ")
    print(f"r1_vec= {r1_vec} [au]")
    print(f"v1_vec= {v1_vec} [au/day]")


def test_ex6_1_hohmann():
    """
    Hohmann Transfer, Vallado [2], example 6-1, p.326; uses p.325, algirithm 36.
    Hohmann, one central body for the transfer ellipse.

    Notes:
    ----------
        Note, interplanetary patched conic missions in Vallado [4] chapter 12, pp.959.
        Interplanetary missions use the patched conic; 3 orbit types:
            1) initial body - departure
            2) transfer body - transfer
            3) final body - arrival
    """
    print(f"\nTest Hohmann Transfer, Vallado [2], example 6.1")
    # define constants
    au = 149597870.7  # [km/au] Vallado[2] p.1042, tbl.D-5
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] earth radius; Vallado [2] p.1041, tbl.D-3

    # define inputs for hohmann transfer
    r1 = r_earth + 191.34411  # [km]
    r2 = r_earth + 35781.34857  # [km]

    # vallado hohmann transfer
    tof_hoh, ecc_trans, dv_total = vfunc.val_hohmann(
        r_init=r1, r_final=r2, mu_trans=mu_earth_km
    )
    print(f"Hohmann transfer time: {(tof_hoh/2):0.8g} [s], {tof_hoh/(2*60):0.8g} [min]")
    print(f"Transfer eccentricity, ecc_trans= {ecc_trans:.6g}")
    print(f"Total transfer delta-v, dv_total= {dv_total:.6g} [km/s]")


def test_ex6_2_bielliptic():
    """
    Bi-Elliptic Transfer, Vallado [2], example 6-2, p327.
    Bi-Elliptic uses one central body for the transfer ellipse.
    Two transfer ellipses; one at periapsis, the other at apoapsis.
        1) initial body - departure
        2) transfer body - transfer
        3) final body - arrival

    Notes:
    ----------
        Note, interplanetary patched conic missions in Vallado [4] chapter 12, pp.959.
        Interplanetary missions use the patched conic; 3 orbit types:
            1) initial body - departure
            2) transfer body - transfer
            3) final body - arrival
    """
    print("\nTest Vallado [2] Bi-Elliptic Transfer, example 6-2")  # temporary print
    # define constants
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] earth radius; Vallado [2] p.1041, tbl.D-3

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
    Test Vallado [2] One-Tangent Burn Transfer, example 6-3, p334.

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
    print(f"\nTest one-tangent burn, Vallado [2] example 6-3:")
    # constants
    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    mu_sun_au = mu_sun_km / (au**3)  # unit conversion
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] earth radius; Vallado [2] p.1041, tbl.D-3

    # define inputs; one-tangent transfer (two burns/impulses)
    r0_mag = r_earth + 191.34411  # [km], example 6-3
    r1_mag = r_earth + 35781.34857  # [km], example 6-3, geosynchronous
    nu_trans_b = 160  # [deg], example 6-3
    # uncomment below to test moon trajectory
    # r1_mag = r_earth + 376310  # [km], Vallado [2] p.336, tbl 6-1, moon
    # nu_trans_b = 175  # [deg], Vallado [2] p.336, tbl 6-1, moon

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


def test_planet_rv():  # test various dates & planets
    """
    Test Vallado [4], example 5-5, pp.304; using algorithm 33, pp.303.
    Test Vallado [4], example 12-8, pp.978; using algorithm 33, pp.303.

    Given:
    ----------
        Earth(1977-09-08 09:08:17), Vallado [4] ex 5-5, pp.304.
        Jupiter(1977-09-08 09:08:17), Vallado [4], ex 12-8, pp.978
    Find:
    ----------
        Equatorial & Ecliptic XYZ
    Returns:
    -------
        None
    Notes:
    -------
        Use date/time as object for ease of user reading string date/time.
        References: see list at file beginning.
    """
    print(f"\nTest planet_rv(), various planets & dates:")
    # constants
    np.set_printoptions(precision=6)  # numpy, set vector printing size
    deg2rad = math.pi / 180  # used multiple times
    rad2deg = 180 / math.pi  # used multiple times

    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5

    # dates to retrieve planet positions
    date_list = [
        "1994-5-20 20:0:0",  # Jupiter, ex5-5
        "1977-09-08 09:08:17",  # earth, ex12-8
        "1979-03-05 12:05:26",  # jupiter, ex12-8
        "1980-11-12 23:46:30",  # saturn, ex12-8
    ]
    "1994-5-20 20:0:0"
    # generate list of datetime objects
    x = [dt.strptime(s, "%Y-%m-%d %H:%M:%S") for s in date_list]
    # for i in range(len(x)):
    #     print(f"{x[i]}")
    # prepare variables for julian date calculation
    year, month, day, hour, minute, second = (
        x[2].year,
        x[2].month,
        x[2].day,
        x[2].hour,
        x[2].minute,
        x[2].second,
    )
    # Jupiter; x[0], just test planet position
    print(f"\nEphemeris date, Jupiter, ex5-5, x[0]= {x[0]}")
    r_vec, v_vec, r1_vec, v1_vec = planet_rv(planet_id=4, date_=x[0])
    print(f"Equatorial frame:")
    print(f"Earth, r_vec= {r_vec} [au]\nEarth, v_vec= {v_vec} [au/day]")
    print(f"Ecliptic frame:")
    print(f"Earth, r1_vec= {r1_vec} [au]\nEarth, v1_vec= {v1_vec} [au/day]")

    # Earth; x[1], planet_id=2
    print(f"\nEphemeris date, Earth, ex12-8, x[1]= {x[1]}")
    r_vec, v_vec, r1_vec, v1_vec = planet_rv(planet_id=2, date_=x[1])
    print(f"Equatorial frame:")
    print(f"Earth, r_vec= {r_vec*au} [km]\nEarth, v_vec= {v_vec*au/86400} [km/s]")

    # Jupiter; x[2], planet_id=4
    print(f"\nEphemeris date, Jupiter, ex12-8, x[2]= {x[2]}")
    r_vec, v_vec, r1_vec, v1_vec = planet_rv(planet_id=4, date_=x[2])
    print(f"Equatorial frame:")
    print(f"Jupiter, r_vec= {r_vec*au} [km]\nJupiter, v_vec= {v_vec*au/86400} [km/s]")
    return


def test_lambert_izzo():
    """
    Solve Lambert's problem with Izzo's devised algorithm; circa 2015.
        Test successful against Braeuning problems 5.3, 5.4; Earth->Mars.
        Test close enough Vallado [4] example 12-8; Earth->Jupiter:
    Returns
    -------
        None
    """
    from izzo_1 import izzo2015

    print(f"\nTest isso2015() LambertSolver:")
    # Solar system constants
    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    GM_sun_au = GM_sun_km / (au**3)
    mu = GM_sun_au
    
    # *********** Braeuning example 5.4 *********
    print(f"** Test of Braeunig's examples, ex5.3, ex5.4; Earth->Mars: **")
    tof = 207 * 24 * 60 * 60  # [s] given, time of flight; 207 days
    # Ecliptic coordinates; Braeunig ex5.3, ex5.4
    r1_vec = np.array([0.473265, -0.899215, 0])  # [au]
    r1_mag = np.linalg.norm(r1_vec)
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # [au]
    r2_mag = np.linalg.norm(r2_vec)

    v1_vec, v2_vec = izzo2015(
        mu=mu,
        r1=r1_vec,
        r2=r2_vec,
        tof=tof,
        M=0,
        prograde=True,
        low_path=True,
        maxiter=35,
        atol=1e-5,
        rtol=1e-7,
    )
    v1_mag, v2_mag = [np.linalg.norm(v) for v in [v1_vec, v2_vec]]

    np.set_printoptions(precision=5)  # numpy has spectial print provisions
    print(f"v1_vec= {v1_vec*au} [km/s]")  # note conversion au->km
    print(f"v2_vec= {v2_vec*au} [km/s]")  # note conversion au->km
    
    orbit_energy = ((v1_mag**2) / 2) - (mu / r1_mag)
    sma = -mu / (2 * orbit_energy)
    print(f"transfer semimajor axis, sma= {sma*au:.8g} [km]")

    h_vec = np.cross(r1_vec, v1_vec)
    h_mag = np.linalg.norm(h_vec)
    # print(f"h_vec= {h_vec} [au^2/s], h_mag= {h_mag:.6g} [au^2/s]")

    p = (h_mag**2) / mu
    print(
        f"p= {p:.6g} [au], sma= {sma:.6g} [au], tof= {tof/(24*3600):.6g} [day]"
    )

    ecc_vec = ((np.cross(v1_vec, h_vec)) / mu) - (r1_vec / r1_mag)
    ecc_mag = np.linalg.norm(ecc_vec)
    print(f"ecc_mag= {ecc_mag:.6g}")
    
    # **************** Vallado 12-8 *************
    print(f"\n** Test of Vallado [4] example 12-8; Earth->Jupiter: **")
    tof = 1.487*365 * 24 * 60 * 60  # [s] given, time of flight; 1.487 years
    # Ecliptic coordinates; Vallado [4], p.978, ex.12-8
    # r1_vec = np.array([146169549, -36666991, -644])  # [km]
    r1_vec = np.array([146169549, -36666991, -644])/au  # [au]; @ Earth
    r1_mag = np.linalg.norm(r1_vec)
    # r2_vec = np.array([-482178605, 627481965, 8221250])  # [km]
    r2_vec = np.array([-482178605, 627481695, 8221250])/au  # [au]; @ Jupiter
    r2_mag = np.linalg.norm(r2_vec)

    v1_vec, v2_vec = izzo2015(
        mu=mu,
        r1=r1_vec,
        r2=r2_vec,
        tof=tof,
        M=0,
        prograde=True,
        low_path=True,
        maxiter=35,
        atol=1e-5,
        rtol=1e-7,
    )
    v1_mag, v2_mag = [np.linalg.norm(v) for v in [v1_vec, v2_vec]]

    np.set_printoptions(precision=5)  # numpy has spectial print provisions
    print(f"v1_vec= {v1_vec*au} [km/s]")  # note conversion au->km
    print(f"v2_vec= {v2_vec*au} [km/s]")  # note conversion au->km
    
    # print(f"# of iterations {numiter}, time per iteration, tpi= {tpi:.6g} [s]")
    
    orbit_energy = ((v1_mag**2) / 2) - (mu / r1_mag)
    sma = -mu / (2 * orbit_energy)
    print(f"transfer semimajor axis, sma= {sma:.8g} [au]")

    h_vec = np.cross(r1_vec, v1_vec)
    h_mag = np.linalg.norm(h_vec)
    # print(f"h_vec= {h_vec} [au^2/s], h_mag= {h_mag:.6g} [au^2/s]")

    p = (h_mag**2) / mu
    print(
        f"p= {p:.6g} [au], sma= {sma:.6g} [au], tof= {tof/(24*3600):.6g} [day]"
    )

    ecc_vec = ((np.cross(v1_vec, h_vec)) / mu) - (r1_vec / r1_mag)
    ecc_mag = np.linalg.norm(ecc_vec)
    print(f"ecc_mag= {ecc_mag:.6g}")

    return None  # test_lambert_izzo()


def test_ex12_8_patchedConic():
    """
    Test Jupiter fly-by. Vallado [4] example 12-8, p.978.

    Given:
    ----------
        Depart earth 1977-09-08 09:08:17 UTC
        Fly-by Jupiter 1979-03-05 12:05:26 UTC
        Fly-by Saturn 1980-11-12 23:46:30 UTC
    Find:
    ----------
        delta heliocentric velocity vector for Jupiter fly-by
        turning angle
        closest radius of Jupiter fly-by
    Returns:
    -------
        None
    Notes:
    -------
        Voyager Jupiter gravity assist/fly-by enroute to Saturn fly-by.
        Use date/time as object for ease of user reading string date/time.
        References: see list at file beginning.
    """
    print(f"\nTest Jupiter fly-by, Vallado [4] example 12-8:")
    # constants
    np.set_printoptions(precision=6)  # numpy, set vector printing size
    deg2rad = math.pi / 180  # used multiple times
    rad2deg = 180 / math.pi  # used multiple times

    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5

    # dates to retrieve planet positions
    date_list = [
        "1977-09-08 09:08:17",  # UTC, depart earth
        "1979-03-05 12:05:26",  # fly-by jupiter
        "1980-11-12 23:46:30",  # fly-by saturn
    ]
    # generate list of datetime objects
    x = [dt.strptime(s, "%Y-%m-%d %H:%M:%S") for s in date_list]
    # for i in range(len(x)):
    #     print(f"{x[i]}")
    # prepare variables for julian date calculation
    year, month, day, hour, minute, second = (
        x[0].year,
        x[0].month,
        x[0].day,
        x[0].hour,
        x[0].minute,
        x[0].second,
    )
    date1 = x[0]
    print(f"Ephemeris date, x[0]= {x[0]}")
    r_vec, v_vec, r1_vec, v1_vec = planet_rv(planet_id=2, date_=date1)
    print(f"Equatorial frame:")
    print(f"Earth, r_vec= {r_vec} [au]\nEarth, v_vec= {v_vec} [au/dau]")
    print(f"Ecliptic frame:")
    print(f"Earth, r1_vec= {r1_vec} [au]\nEarth, v1_vec= {v1_vec} [au/day]")


def Main():  # helps editor navigation :--)
    return


# Test functions and class methods are called here.
if __name__ == "__main__":
    # test_ex2_1KeplerE()  # kepler ellipse solve for eccentric anomaly
    # test_ex2_4_keplerUni()  # kepler propagation; Kepler universal variables
    # test_prb2_7_tof()  # time of flight, problem 2-7
    # test_prb2_7a_tof(plot_sp=False)  # time-of-flight; plot sma vs. sp
    # test_ex5_1_sunPosition()  # sun position
    # test_ex5_2_sunRiseSet()  # sunrise sunset
    # test_ex5_5_planetLocation()  # planet location
    # test_ex6_1_hohmann()  # hohmann transfer, example 6-1
    # test_ex6_2_bielliptic()  # bi-elliptic transfer, example 6-2
    # test_ex6_3_one_tan_burn()  # one-tangent transfer, example 6-3
    # test_planet_rv()  # test various dates and planets for planet_rv()
    test_lambert_izzo()  # lambert solver by izzo
    # test_ex12_8_patchedConic()  # NOT Finished, gravity assist, Jupiter fly-by
