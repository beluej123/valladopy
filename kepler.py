"""
Created Sun Jan 31 16:41:34 2016, @author: Alex
Edits 2024-08-21 +, Jeff Belue.

Notes:
----------
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally, angles are saved in [rad], distance [km].
    
    Supporting functions for the test functions below, may be found in other
        files, for example vallad_func.py, astro_time.py, kepler.py etc...
        Also note, the test examples are collected right after this document
        block.  However, the example test functions are defined/enabled at the
        end of this file.  Each example function is designed to be stand-alone,
        but, if you use the function as stand alone you will need to copy the
        imports...

References:
----------
    See references.py for references list.

Orbital Elements Naming Collection:
Start with Kepler coe (classic orbital elements).
    https://ssd.jpl.nasa.gov/planets/approx_pos.html

    o_type : int  , [-] orbit type (python dictionary list below)
                    0:"circular", 1:"circular inclined", 2:"circular equatorial"
                    3:"elliptical", 4:"elliptical equatorial"
                    5:"parabolic", 6:"parabolic equatorial"
                    7:"hyperbolic", 8:"hyperbolic equatorial"
    sp     : float, [km or au] semi-parameter (aka p)
    sma    : float, [km or au] semi-major axis (aka a)
    ecc    : float, [--] eccentricity
    incl   : float, [rad] inclination
    raan   : float, [rad] right ascension of ascending node (aka capital W)
    w_     : float, [rad] arguement of periapsis (aka aop, or arg_p)
    TA     : float, [rad] true angle/anomaly (aka t_anom, or theta)

    alternative coe's for circular & equatirial:
    Lt0    : float, [rad] true longitude at epoch, circular equatorial
                    when incl=0, ecc=0
    w_bar  : float, [rad] longitude of periapsis (aka II), equatorial
                NOT argument of periapsis, w_
                Note, w_bar = w + RAAN
    u_     : float, [rad] argument of lattitude (aka ), circular inclined

    Other coe Elements:
        w_p    : float [rad] longitude of periapsis (aka w_bar) ??
        L_     : float, [deg] mean longitude
                    NOT mean anomaly, M
                    L_ = w_bar + M
        wt_bar : float, [rad] true longitude of periapsis
                    measured in one plane
        M_     : mean anomaly, often replaces TA
        t_peri : float, [jd] time of periapsis passage

    circular, e=0: w_ and TA = undefined;
        use argument of latitude, u_; u_=acos((n_vec X r_vec)/(n_mag * r_mag))
        If r_vec[2] < 0 then 180 < u < 360 degree

    equatorial, i=0 or 180 [deg]: raan and w_ = undefined
        use longitude of periapsis, II (aka w_bar); II=acos(e_vec[0]/e_mag)
        If e_vec[1] < 0 then 180 < II < 360 degree

    circular & equatorial, e=0 and i=0 or i=180: w_ and raan and TA = undefined;
        use true longitude, Lt0 = angle between r0 & I-axis; Lt0=acos(r_vec[1]/r_mag)
        If r_mag[1] < 0 then 180 < Lt0 < 360 degree

From JPL Horizizons, osculating elements:
    Symbol & meaning [1 au= 149597870.700 km, 1 day= 86400.0 s]:
    JDTDB  Julian Day Number, Barycentric Dynamical Time
    EC     Eccentricity, e
    QR     Periapsis distance, q (au)
    IN     Inclination w.r.t X-Y plane, i (degrees)
    OM     Longitude of Ascending Node, OMEGA, (degrees)
    W      Argument of Perifocus, w (degrees)
    Tp     Time of periapsis (Julian Day Number)
    N      Mean motion, n (degrees/day)
    MA     Mean anomaly, M (degrees)
    TA     True anomaly, nu (degrees)
    A      Semi-major axis, a (au)
    AD     Apoapsis distance (au)
    PR     Sidereal orbit period (day)
"""

import math

import numpy as np

from solarsys import *


def find_c2c3(psi: float):
    """
    c(:math:`psi`) functions for the universal formulation (Algorithm 1)

    Trigonometric implementation of the :math:`c(psi)` functions needed in
    the universal formulation of Kepler's Equation.  Reference Vallado [2],
    section 2.2, p.63, algorithm 1.

    Parameters:
    ----------
        psi: double
            :math:`psi = chi^2/a` where :math:`chi` is universal
            variable and a is the semi-major axis
    Returns:
    -------
        c2: double
            c2 coefficient in universal formulation of Kepler's Equation
        c3: double
            c3 coefficient in universal formulation of Kepler's Equation
    Notes:
    ----------
        References: see list at file beginning.
    """

    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        psi3 = psi * psi * psi
        sqrt_psi3 = np.sqrt(psi3)
        c2 = (1.0 - np.cos(sqrt_psi)) / psi
        c3 = (sqrt_psi - np.sin(sqrt_psi)) / sqrt_psi3
    else:
        if psi < -1e-6:
            sqrt_psi = np.sqrt(-psi)
            sqrt_psi3 = np.sqrt((-psi) * (-psi) * (-psi))
            c2 = (1.0 - np.cosh(sqrt_psi)) / psi
            c3 = (np.sinh(sqrt_psi) - sqrt_psi) / sqrt_psi3
        else:
            c2 = 0.5
            c3 = 1.0 / 6.0
    return c2, c3


def kep_eqtnE(M, e, tol=1e-8):
    """
    Elliptical solution to Kepler's Equation.
    Vallado [2] or [4], section 2.2, algorithm 2, p.65.

    Newton-Raphson iterative approach solving Kepler's Equation for
    elliptical orbits.

    Input Parameters:
    ----------
        M   : float, [rad] mean anomaly; must, -2*pi <= M <= +2*pi
        e   : float, [--] eccentricity
        tol : float, optional, default=1E-8
            Newton-Raphson convergence tolerance

    Returns:
    -------
        E   : float, [rad] eccentric anomaly
    """
    # keep M between +- 2*pi
    # if M < 0:
    #     M = M % (2 * math.pi)
    #     M = -M
    # else:
    #     M = M % (2 * math.pi)
    M = M % (2 * math.pi)

    if e < 1 and e > 0:  # make sure elliptical orbit
        if ((M > -np.pi) and (M < 0)) or ((M > np.pi) and (M < 2 * np.pi)):
            E = M - e
        else:
            E = M + e

        diff = np.inf
        while diff > tol:
            E_new = E + ((M - E + e * np.sin(E)) / (1.0 - e * np.cos(E)))
            diff = np.abs(E_new - E)
            E = E_new
    else:
        print(f"Ellipse test fails for function kep_eqtnE().")
        raise NameError("kep_eqtnE(), ellipse test fails.")

    return E


def kep_eqtnP(del_t, p, mu=Earth.mu):
    """
    Parabolic solution to Kepler's Equation (Algorithm 3)

    A trigonometric approach to solving Kepler's Equation for
    parabolic orbits. Reference Vallado [2], section 2.2, p.69, algorithm 3.

    Parameters
    ----------
    del_t: double
        Change in time
    p: double
        Semi-parameter
    mu: double, optional, default = Earth.mu
        Gravitational parameter; defaults to Earth

    Returns
    -------
    B: double
        Parabolic Anomaly (radians)
    """
    p3 = p**3
    n_p = 2.0 * np.sqrt(mu / p3)
    s = 0.5 * np.arctan(2.0 / (3.0 * n_p * del_t))
    w = np.arctan((np.tan(s)) ** (1 / 3.0))
    B = 2.0 / np.tan(2.0 * w)
    return B


def kep_eqtnH(M, ecc, tol=1e-8):
    """
    Hyperbolic solution to Kepler's Equation (Algorithm 4)

    A Newton-Raphson iterative approach to solving Kepler's Equation for
    hyperbolic orbits. Reference Vallado [2], section 2.2, p.71, algorithm 4.

    Parameters
    ----------
    M: double
        Mean Anomaly (radians)
    e: double
        Eccentricity
    tol: double, optional, default=1E-8
        Convergence tolerance used in Newton-Raphson method

    Returns
    -------
    H: double
        Hyperbolic Anomaly (radians)
    """
    if ecc < 1.6:
        if (M > -np.pi and M < 0) or (M > np.pi):
            H = M - ecc
        else:
            H = M + ecc
    else:
        if ecc < 3.6 and (np.abs(M) > np.pi):
            H = M - np.sign(M) * ecc
        else:
            H = M / (ecc - 1)

    diff = np.inf
    while diff > tol:
        H_new = H + (M - ecc * np.sinh(H) + H) / (ecc * np.cosh(H) - 1)
        diff = np.abs(H_new - H)
        H = H_new
    return H


def true_to_anom(true_anom, e):
    """
    Converts true anomaly to the proper orbit anomaly (Algorithm 5)

    Converts true anomaly to eccentric (E) anomaly for elliptical orbits,
    parabolic (B) anomaly for parabolic orbits, or hyperbolic anomaly (H) for
    hyperbolic orbits.  Reference Vallado [2], section 2.2, p.77, algorithm 5.

    Parameters
    ----------
    true_anom: double
        True anomaly (radians)
    e: double
        Eccentricity

    Returns
    -------
    E/B/H: double
        Eccentric, Parabolic, or Hyperbolic anomaly (radians)
    """
    if e < 1.0:
        num = np.sin(true_anom) * np.sqrt(1.0 - e**2)
        denom = 1 + e * np.cos(true_anom)
        E = np.arcsin(num / denom)
        return E
    elif e == 1.0:
        B = np.tan(0.5 * true_anom)
        return B
    else:
        num = np.sin(true_anom) * np.sqrt(e**2 - 1)
        denom = 1 + e * np.cos(true_anom)
        H = np.arcsinh(num / denom)
        return H


def eccentric_to_true(E, e):
    """
    Convert eccentric angle/anomaly (E) to true anomaly (TA) for elliptical orbits.
    Vallado [2], section 2.2, p.77, algorithm 6; quadrant checks, see Notes below.

    Input Parameters:
    ----------
        E  : float, [rad] eccentric angle/anomaly
        e  : float, [--] eccentricity
    Returns:
    -------
        TA : float, [rad] true angle/anomaly
    Notes:
    ----------
        2024-09-16, JBelue edits to account for quadrant checks; using arctan().
            https://en.wikipedia.org/wiki/True_anomaly
            Avoid numerical issues when the arguments are near ± pi, as the tangents
            become infinite. Also, E and TA are always in the same quadrant, so
            quadrant checks are not needed.  I did not verify the calculation claim;
            my basis for the calculation is on paper by Broucke, R. and Cefola, P.
            (1973); "A Note on the Relations between True and Eccentric Anomalies in
            the Two-Body Problem".
    """
    # num = np.cos(E) - e
    # denom = 1 - e * np.cos(E)
    # TA = np.arccos(num / denom)

    beta = e / (1 + np.sqrt(1 - e**2))
    TA = E + 2 * np.arctan((beta * np.sin(E)) / (1 - beta * np.cos(E)))

    return TA


def parabolic_to_true(B):  # , p, r):
    """
    Converts parabolic anomaly (B) to true anomaly (Algorithm 6 (b))

    Converts parabolic anomaly (B) to true anomaly for parabolic orbits.
    Reference Vallado [2], section 2.2, p.77, algorithm 6.

    Parameters
    ----------
    B: double
        Parabolic anomaly (radians)
    p: double
        semi-parameter (km)
    r: double
        distance to the focus (km)

    Returns
    -------
    true_anom: double
        True anomaly (radians)
    """
    # true_anom = np.arcsin(p*B/r)
    true_anom = np.arctan(B)
    return true_anom


def hyperbolic_to_true(H, e):
    """
    Converts hyperbolic anomaly (H) to true anomaly (Algorithm 6 (c))

    Converts hyperbolic anomaly (H) to true anomaly for hyperbolic orbits.
    Reference Vallado [2], section 2.2, p.77, algorithm 6.

    Parameters
    ----------
    H: double
        Hyperbolic anomaly (radians)
    e: double
        eccentricity

    Returns
    -------
    true_anom: double
        True anomaly (radians)
    """

    num = np.cosh(H) - e
    denom = 1 - e * np.cosh(H)
    true_anom = np.arccos(num / denom)
    return true_anom


def rv2coe(r_vec, v_vec, mu):
    """
    Convert position/velocity vectors to Keplerian orbital elements (coe).
    Vallado [4] section 2.5, pp.114, algorithm 9 pp.115, rv2cov(), example 2-5 pp.116.
    See Curtis [3] algorithm 4.2 pp.209, example 4.3, pp.212, in Example4_x.py.

    TODO: 2024-Sept, improve efficiency by elliminating redundant calculations
    TODO: 2024-Nov, finish test cases
    TODO: 2024-Nov, calculate alternative elements even when not required.
            For comparisons.  Maybe add switch if performance is needed...

    Input Parameters:
    ----------
        r_vec  : numpy.array, [km] row vector, position
        v_vec  : numpy.array, [km/s] row vector, velocity
        mu     : float, [km^3/s^2], gravitational parameter

    Returns:
    ----------
        o_type : int  , [-] orbit type:
                        1=circular, 2=circular equatorial
                        3=elliptical, 4=elliptical equatorial
                        5=parabolic, 6=parabolic equatorial
                        7=hyperbolic, 8=hyperbolic equatorial
        sp     : float, [km or au] semi-parameter (aka p)
        sma    : float, [km or au] semi-major axis (aka a)
        ecc    : float, [--] eccentricity
        incl   : float, [rad] inclination
        raan   : float, [rad] right ascension of ascending node (aka capital W)
        w_     : float, [rad] arguement of periapsis (aka aop, or arg_p)
        TA     : float, [rad] true angle/anomaly (aka t_anom, or theta)
        
        alternative orbital elements for circular & equatorial:
        Lt0    : float, [rad] true longitude at epoch, circular equatorial
                        when incl=0, ecc=0
        w_bar  : float, [rad] longitude of periapsis (aka II), equatorial
                    NOT argument of periapsis, w_
                    Note, w_bar = w + RAAN
        u_     : float, [rad] argument of lattitude (aka ), circular inclined

    Other potential elements:
        L_     : float, [deg] mean longitude
                    NOT mean anomaly, M
                    L_ = w_bar + M
        wt_bar : float, [rad] true longitude of periapsis
                    measured in one plane
        M_     : mean anomaly, often replaces TA
        t_peri : float, [jd] time of periapsis passage

    circular, e=0: w_ and TA = undefined;
        use argument of latitude, u_; u_=acos((n_vec X r_vec)/(n_mag * r_mag))
        If r_vec[2] < 0 then 180 < u < 360 degree

    equatorial, i=0 or 180 [deg]: raan and w_ = undefined
        use longitude of periapsis, II (aka w_bar); II=acos(e_vec[0]/e_mag)
        If e_vec[1] < 0 then 180 < II < 360 degree

    circular & equatorial, e=0 and i=0 or i=180: w_ and raan and TA = undefined;
        use true longitude, Lt0 = angle between r0 & I-axis; Lt0=acos(r_vec[1]/r_mag)
        If r_mag[1] < 0 then 180 < Lt0 < 360 degree

    Notes:
    ----------
        This algorithm handles special cases (circular, equatorial).
            The following hyper-link may be helpful but, be careful of some mis-representations;
            https://colorado.pressbooks.pub/introorbitalmechanics/chapter/copy-of-chapter-3__editing/
    """
    rad2deg = 180 / math.pi
    SMALL_c = 0.00015  # circle threshold value, defines ecc=0
    SMALL_p = 0.00001  # parabolic threshold value, defines ecc=1
    # initialize values since some will not be calculated...
    sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_ = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    # if n_vec = 0 then equatorial orbit
    n_vec = np.cross([0, 0, 1], h_vec)
    n_mag = np.linalg.norm(n_vec)

    # eccentricity; if ecc = 0 then circular orbit
    r0_inv = 1.0 / r_mag  # for calculation efficiency
    ecc_inv = 1  # for calculation efficiency
    A = (v_mag * v_mag - mu * r0_inv) * r_vec
    B = -(np.dot(r_vec, v_vec)) * v_vec
    ecc_vec = (1 / mu) * (A + B)
    ecc_mag = np.linalg.norm(ecc_vec)
    if ecc_mag < SMALL_c:
        ecc_mag = 0.0
        ecc_vec=ecc_vec*0
        ecc_inv = np.inf
    elif ecc_mag > (1 - SMALL_p) and ecc_mag < (1 + SMALL_p):
        ecc_mag = 1
    else:
        ecc_inv = 1 / ecc_mag

    # xi=orbit energy
    xi = (0.5 * v_mag * v_mag) - mu * r0_inv

    if ecc_mag == 0:  # circular, avoids rounding issues
        sma = r_mag
        sp = r_mag
    elif ecc_mag != 0 and ecc_mag != 1:  # not parabolic
        sma = -0.5 * mu / xi
        sp = sma * (1.0 - ecc_mag * ecc_mag)
    elif ecc_mag == 1:  # parabolic orbit
        sma = np.inf
        sp = h_mag * h_mag / mu

    incl = np.arccos(h_vec[2] / h_mag)  # no quadrent check needed

    # examine orbit type (o-type)
    if n_mag == 0.0:  # Equatorial
        if ecc_mag < SMALL_c:  # circular equatorial
            # Lt0=true longitude at epoch (r0)
            Lt0 = math.acos(r_vec[0] * r0_inv)
            if r_vec[1] < 0:
                Lt0 = 2.0 * np.pi - Lt0
            raan = np.nan
            w_ = np.nan  # aka aop
            o_type = 1  # circular equatorial
        else:  # ecc > 0, thus equatorial -> ellipse, parabola, hyperbola
            # w_bar=longitude of periapsis (aka II)
            w_bar = math.acos(ecc_vec[0] * ecc_inv)
            if ecc_vec[1] < 0:
                w_bar = 2.0 * math.pi - w_bar
            raan = np.nan
            TA = math.acos(np.dot(ecc_vec, r_vec) / (ecc_mag * r_mag))
            if np.dot(r_vec, v_vec) < 0:
                TA = 2 * math.pi - TA
            # equatorial, for either ellipse, parabola, hyperbola
            if ecc_mag < 1:
                o_type = 4  # equatorial ellipse
            elif ecc_mag > 1:
                o_type = 8  # equatorial hyperbola
            else:
                o_type = 6  # equatorial parabola
    elif ecc_mag < SMALL_c:  # not equatorial, but circular (inclined)
        w_ = np.nan
        TA = np.nan

        n_inv = 1.0 / n_mag
        raan = math.acos(n_vec[0] * n_inv)
        # u_ = argument of lattitude
        u_ = math.acos(np.dot(n_vec, r_vec) * n_inv * r0_inv)
        if r_vec[2] < 0:
            u_ = 2.0 * math.pi - u_
        o_type = 1  # circular (inclined)
    else:  # elements: non-equatorial, non-circular
        n_inv = 1.0 / n_mag
        ecc_inv = 1 / ecc_mag

        raan = math.acos(n_vec[0] * n_inv)
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan

        # w_ = arguement of periapsis (aka aop, or arg_p)
        w_ = math.acos(np.dot(n_vec, ecc_vec) * n_inv * ecc_inv)
        if ecc_vec[2] < 0:
            w_ = 2 * math.pi - w_

        TA = math.acos(np.dot(ecc_vec, r_vec) / (ecc_mag * r_mag))
        if np.dot(r_vec, v_vec) < 0:
            TA = 2 * math.pi - TA

        # finish identifying orbit type
        if ecc_mag < 1:
            o_type = 3  # ellipse
        elif ecc_mag > 1:
            o_type = 7  # hyperbola
        else:
            o_type = 5  # parabola

    elements = np.array([sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_])
    return o_type, elements


def o_type_decode(o_type):
    """
    Print orbit type based on input value
    o_type orbit type list:
        1:"circular", 2:"circular equatorial"
        3:"elliptical", 4:"elliptical equatorial"
        5:"parabolic", 6:"parabolic equatorial"
        7:"hyperbolic", 8:"hyperbolic equatorial"
    Returns
    ----------
    """
    o_type_list = {
        1: "circular",
        2: "circular equatorial",
        3: "elliptical",
        4: "elliptical equatorial",
        5: "parabolic",
        6: "parabolic equatorial",
        7: "hyperbolic",
        8: "hyperbolic equatorial",
    }
    print(f"{o_type_list.get(o_type)}")
    return None


def print_coe(o_type, elements):
    rad2deg = 180 / math.pi
    sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_ = elements

    o_type_decode(o_type=o_type)  # prints orbit type
    print(f"sp= {sp} [km]")
    print(f"sma= {sma} [km]")
    print(f"ecc_mag= {ecc_mag}")
    print(f"incl= {incl} [rad], {incl*rad2deg} [deg]")
    print(f"raan= {raan} [rad], {raan*rad2deg} [deg]")
    print(f"w_= {raan} [rad], {w_*rad2deg} [deg]")
    print(f"TA= {TA} [rad], {TA*rad2deg} [deg]")
    print(f"Lt0= {Lt0} [rad], {Lt0*rad2deg} [deg]")
    print(f"w_bar= {w_bar} [rad], {w_bar*rad2deg} [deg]")
    print(f"u_= {u_} [rad], {u_*rad2deg} [deg]")
    return None


def coe2rv(p, ecc, inc, raan, aop, anom, mu):
    """
    Convert Keplerian orbital elements to position/velocity vectors; IJK frame.
    Vallado [2], section 2.6, algorithm 10, pp.118
    Vallado [4], section 2.6, algorithm 10, pp.120

    Input Parameters:
    ----------
        p    : float, [km] semi-parameter
                p=h^2 / mu
        ecc  : float, [--] eccentricity
        inc  : float, [rad] inclination
        raan : float, [rad] right ascension of the ascending node
        aop  : float, [rad] argument of periapsis (aka w, or omega)
                w=w_bar-RAAN; undefined for RAAN=0, undefined for circular
        anom : float, [rad] true angle/anomaly (aka TA)
        mu   : float, [km^3/s^2] gravitational parameter
    Returns:
    -------
        r_ijk : numpy.array, [km] position vector in IJK frame
        v_ijk : numpy.array, [km/s] velocity vector in IJK frame
    Notes:
    ----
        Algorithm assumes raan, aop, and anom have been set to account for
        special cases (circular, equatorial, etc.) as in rv2coe (Algorithm 9)
        Also see Curtis, p.473 example 8.7.
    """
    # saved trig computations save computing time
    cosv = np.cos(anom)
    sinv = np.sin(anom)
    cosi = np.cos(inc)
    sini = np.sin(inc)
    cosw = np.cos(aop)
    sinw = np.sin(aop)
    coso = np.cos(raan)
    sino = np.sin(raan)

    r_pqw = np.matrix(
        [p * cosv / (1.0 + ecc * cosv), p * sinv / (1.0 + ecc * cosv), 0.0]
    )
    r_pqw = r_pqw.T  # Make column vector

    v_pqw = np.matrix([-np.sqrt(mu / p) * sinv, np.sqrt(mu / p) * (ecc + cosv), 0.0])
    v_pqw = v_pqw.T  # Make column vector

    m_pqw2ijk = [
        [
            coso * cosw - sino * sinw * cosi,
            -coso * sinw - sino * cosw * cosi,
            sino * sini,
        ],
        [
            sino * cosw + coso * sinw * cosi,
            -sino * sinw + coso * cosw * cosi,
            -coso * sini,
        ],
        [sinw * sini, cosw * sini, cosi],
    ]
    m_pqw2ijk = np.matrix(m_pqw2ijk)
    #    m_pqw2ijk = np.matrix([[row1], [row2], [row3]])
    # Convert to IJK frame
    r_ijk = m_pqw2ijk * r_pqw
    v_ijk = m_pqw2ijk * v_pqw
    return r_ijk, v_ijk


def findTOF(r0_vec, r1_vec, sp, mu):
    """
    Find tof (time-of-flight) between two position vectors.
        Now there are three tof functions; 2024-10-14: tof(), tof_a(), tof_b().
        Vallado [2], section 2.8, algorithm 11, p.126.
        Vallado [4], section 2.8, algorithm 11, pp.128; problem 2.7, p.130.

    Input Parameters:
    ----------
        r0_vec : numpy.array, initial position vector
        r1_vec : numpy.array, final position vector
        sp     : float, [] semi-parameter (aka p)
        mu     : float, [km^3/s^2] gravitational parameter
    Returns:
    -------
        TOF    : float, [s] time-of-flight
    Note:
    ----------
        Numeric precision plays a role in orbit type decisions; especially
            for parabolic orbits; this routine deals with it...
        This routine requires a value for sp (semi-parameter, aka p), but in
            practice ecc (eccentricity) and or sma (semi-major axis) must be
            chosen to completely define the orbit -  to understand the sp limits
            see findTOF_a() which returns sp limits on ellipse orbit, noting
            that parabolic and hyperbolic relations come out of the ellipse limits.

        As long as consistant units are passed to this routine unit definitions
            are not required.
    """

    r0_mag = np.linalg.norm(r0_vec)
    r1_mag = np.linalg.norm(r1_vec)

    cosdv = np.dot(r0_vec.T, r1_vec) / (r0_mag * r1_mag)  # note r0.T = transpose
    del_anom = np.arccos(cosdv)
    k = r0_mag * r1_mag * (1.0 - cosdv)
    l = r0_mag + r1_mag
    m = r0_mag * r1_mag * (1.0 + cosdv)

    f = 1.0 - (r1_mag / sp) * (1.0 - cosdv)
    g = r0_mag * r1_mag * np.sin(del_anom) / (np.sqrt(mu * sp))

    # Look for parabolic and near parabolic orbits; numeric precision
    #   plays a critical role in orbit-type.
    # The following eqn is brokend down into a1 & a2; to prevent divide by 0.
    # a = m * k * sp / (((2.0 * m - l * l) * sp * sp) + (2.0 * k * l * sp) - k * k)
    a1 = m * k * sp
    a2 = ((2.0 * m - l * l) * sp * sp) + (2.0 * k * l * sp) - k * k
    # note, large a1/a2 ratios are very near parabolic, define as parabolic
    if a2 == 0:
        a = np.inf
    else:
        a = a1 / a2
        if abs(a) > 1e16:
            a = np.inf

    if a > 0.0:
        if a == np.inf:
            # parabolic
            c = np.sqrt(r0_mag**2 + r1_mag**2 - 2.0 * r0_mag * r1_mag * cosdv)
            s = (r0_mag + r1_mag + c) * 0.5
            TOF = (2.0 / 3.0) * np.sqrt(0.5 * s**3 / mu) * (1.0 - ((s - c) / s) ** 1.5)
            # return TOF
        else:
            # elliptic
            f_dot = (
                np.sqrt(mu / sp)
                * np.tan(0.5 * del_anom)
                * ((1.0 - cosdv) / sp - 1.0 / r0_mag - 1 / r1_mag)
            )
            cosde = 1.0 - (r0_mag / a) * (1.0 - f)
            sinde = (-r0_mag * r1_mag * f_dot) / (np.sqrt(mu * a))
            del_E = np.arccos(cosde)
            TOF = g + np.sqrt(a**3 / mu) * (del_E - sinde)
            # return TOF
    elif a < 0.0:
        # hyperbolic
        coshdh = 1.0 + (f - 1.0) * (r0_mag / a)
        del_H = np.arccosh(coshdh)
        TOF = g + np.sqrt((-a) ** 3 / mu) * (np.sinh(del_H) - del_H)
        # return TOF
    else:
        # should never get here
        TOF = None
    return TOF


def findTOF_a(r0, r1, sp, mu):
    """
    Same as findTOF() except this function returns internal calculations
        allowing user to understand orbit type; range of sp (semi-parameter).
        Now there are three tof functions; 2024-10-14: tof(), tof_a(), tof_b().
        Vallado [2], section 2.8, algorithm 11, p.126.
        Vallado [4], section 2.8, algorithm 11, pp.128; problem 2.7, p.130.

    Note:
    ----------
        Numeric precision plays a role in orbit type decisions; especially
            for parabolic orbits; this routine deals with it...
        This routine requires a value for sp (semi-parameter, aka p), but in
            practice ecc (eccentricity) and or sma (semi-major axis) must be
            chosen to completely define the orbit -  to understand the sp limits
            see findTOF_a() which returns sp limits on ellipse orbit, noting
            that parabolic and hyperbolic relations come out of the ellipse limits.

        As long as consistant units are passed to this routine unit definitions
            are not required.

    Parameters:
    ----------
        r0  : numpy.matrix (3x1), initial position vector
        r1  : numpy.matrix (3x1), final position vector
        sp  : float, [km] Semi-parameter (aka p)
        mu  : float, [km^3/s^2] gravitational parameter

    Returns
    -------
        TOF  : float, [s] time-of-flight
        a    : float, semi-major axis (aka sma)
        sp_i : float, minimum ellipse semi-parameter
        sp_ii: float, maximum ellipse semi-parameter
    """

    r0_mag = np.linalg.norm(r0)
    r1_mag = np.linalg.norm(r1)

    cosdv = np.dot(r0.T, r1) / (r0_mag * r1_mag)  # [rad] note r0.T = transpose
    del_anom = np.arccos(cosdv)  # [rad]

    # constants; given r0, r1, angle
    k = r0_mag * r1_mag * (1.0 - cosdv)
    l = r0_mag + r1_mag
    m = r0_mag * r1_mag * (1.0 + cosdv)

    # bracket p values for ellipse, p_i & p_ii; BMWS [2], p.205
    # values > sp_ii will be hyperbolic trajectories
    # value at sp_ii will be parabolic trajectory
    # minimum sp for ellipse; calculated value maybe degenerate...
    sp_i = k / (l + np.sqrt(2 * m))  # BMWS [2], p.208, eqn 5-52
    # maximum sp for ellipse; calculated value is actually a parabola
    sp_ii = k / (l - np.sqrt(2 * m))  # BMWS [2], p.208, eqn 5-53

    f = 1.0 - (r1_mag / sp) * (1.0 - cosdv)
    g = r0_mag * r1_mag * np.sin(del_anom) / (np.sqrt(mu * sp))
    # Look for parabolic and near parabolic orbits; numeric precision
    #   plays a critical role in orbit-type.
    # The following eqn is brokend down into a1 & a2; to prevent divide by 0.
    # a = m * k * sp / (((2.0 * m - l * l) * sp * sp) + (2.0 * k * l * sp) - k * k)
    a1 = m * k * sp
    a2 = ((2.0 * m - l * l) * sp * sp) + (2.0 * k * l * sp) - k * k
    # note, large a1/a2 ratios are very near parabolic, define as parabolic
    if a2 == 0:
        a = np.inf
    else:
        a = a1 / a2
        if abs(a) > 1e16:
            a = np.inf

    if a > 0.0:
        if a == np.inf:
            # parabola
            c = np.sqrt(r0_mag**2 + r1_mag**2 - 2.0 * r0_mag * r1_mag * cosdv)
            s = (r0_mag + r1_mag + c) * 0.5
            TOF = (2.0 / 3.0) * np.sqrt(0.5 * s**3 / mu) * (1.0 - ((s - c) / s) ** 1.5)
            # return TOF, a, sp_i, sp_ii
        else:
            # ellipse
            f_dot = (
                np.sqrt(mu / sp)
                * np.tan(0.5 * del_anom)
                * ((1.0 - cosdv) / sp - 1.0 / r0_mag - 1 / r1_mag)
            )
            cosde = 1.0 - (r0_mag / a) * (1.0 - f)
            sinde = (-r0_mag * r1_mag * f_dot) / (np.sqrt(mu * a))
            del_E = np.arccos(cosde)
            TOF = g + np.sqrt(a**3 / mu) * (del_E - sinde)
            # return TOF, a, sp_i, sp_ii
    elif a < 0.0:
        # hyperbola
        coshdh = 1.0 + (f - 1.0) * (r0_mag / a)
        del_H = np.arccosh(coshdh)
        TOF = g + np.sqrt((-a) ** 3 / mu) * (np.sinh(del_H) - del_H)
        # return TOF, a, sp_i, sp_ii
    else:
        # should never get here
        TOF = None
    return TOF, a, sp_i, sp_ii


def findTOF_b(r0_mag, r1_mag, delta_ta, sp, mu):
    """
    Find tof (time-of-flight) between two position magnitudes, delta_ta, sp.
        Now there are three tof functions; 2024-10-14: tof(), tof_a(), tof_b().
        This function origionally designed to find tof to soi (sphere of influence).
        Vallado [2], section 2.8, algorithm 11, p.126.
        Vallado [4], section 2.8, algorithm 11, pp.128; problem 2.7, p.130.

    Parameters:
    ----------
        r0_mag   : float, [km] initial position magnitude
        r1_mag   : float, [km] final position magnitude
        delta_ta : float, [rad] delta true anomaly/angle
        sp       : float, [km] semi-parameter (aka p)
        mu       : float, [km^3/s^2] gravitational parameter
    Returns:
    -------
        TOF      : float, [sec] time-of-flight
        o_type   : str, orbit type; e=ellipse, p=parabola, h=hyperbola
    Note:
    ----------
        Numeric precision plays a role in orbit type decisions; especially
            for parabolic orbits; this routine deals with it...
        This routine requires a value for sp (semi-parameter, aka p), but in
            practice ecc (eccentricity) and or sma (semi-major axis) must be
            chosen to completely define the orbit -  to understand the sp limits
            see findTOF_a() which returns sp limits on ellipse orbit, noting
            that parabolic and hyperbolic relations come out of the ellipse limits.

        As long as consistant units are passed to this routine, unit definitions
            are not required.
    """
    # numerical precision can be problematic, especially near parabolic orbits.
    # import decimal # explore numerical precision
    # from decimal import Decimal, getcontext
    # getcontext().prec = 28

    del_anom = delta_ta
    cosdv = math.cos(delta_ta)
    k = r0_mag * r1_mag * (1.0 - cosdv)
    l = r0_mag + r1_mag
    m = r0_mag * r1_mag * (1.0 + cosdv)

    f = 1.0 - (r1_mag / sp) * (1.0 - cosdv)
    g = r0_mag * r1_mag * math.sin(del_anom) / (math.sqrt(mu * sp))
    # the following eqn is brokend down into a1 & a2; to prevent divide by 0
    # a = m * k * sp / (((2.0 * m - l * l) * sp * sp) + (2.0 * k * l * sp) - k * k)
    a1 = m * k * sp
    a2 = ((2.0 * m - l * l) * sp * sp) + (2.0 * k * l * sp) - k * k
    # define very large a ratios as very near parabolic, as parabolic
    if a2 == 0:
        a = np.inf
    else:
        a = a1 / a2
        if abs(a) > 1e16:
            a = np.inf
    if a > 0.0:
        # parabolic
        if a == np.inf:
            c = math.sqrt(r0_mag**2 + r1_mag**2 - 2.0 * r0_mag * r1_mag * cosdv)
            s = (r0_mag + r1_mag + c) * 0.5
            TOF = (
                (2.0 / 3.0) * math.sqrt(0.5 * s**3 / mu) * (1.0 - ((s - c) / s) ** 1.5)
            )
            o_type = "p"  # orbit type, parabolic
        else:
            # elliptic
            f_dot = (
                math.sqrt(mu / sp)
                * math.tan(0.5 * del_anom)
                * ((1.0 - cosdv) / sp - 1.0 / r0_mag - 1 / r1_mag)
            )
            cosde = 1.0 - (r0_mag / a) * (1.0 - f)
            sinde = (-r0_mag * r1_mag * f_dot) / (np.sqrt(mu * a))
            del_E = math.acos(cosde)
            TOF = g + math.sqrt(a**3 / mu) * (del_E - sinde)
            o_type = "e"  # orbit type, eliptic
    elif a < 0.0:
        # hyperbolic
        coshdh = 1.0 + (f - 1.0) * (r0_mag / a)
        del_H = math.acosh(coshdh)
        TOF = g + math.sqrt((-a) ** 3 / mu) * (math.sinh(del_H) - del_H)
        o_type = "h"  # orbit type, hyperbolic
    # troubleshooting print statements; commented out at this point
    # print(f"orbit type, {o_type}")
    # print(f"cosdv= {cosdv:.8g}")  # cos(delta ta)
    # print(f"k= {k:.8g}")
    # print(f"l= {l:.8g}")
    # print(f"m= {m:.8g}")
    # print(f"f= {f:.8g}")
    # print(f"g= {g:.8g}")
    # print(f"a1= {a1:.8g}")
    # print(f"a2= {a2:.8g}")
    # print(f"a= {a:.8g}")
    # if a < 0.0:
    #     print(f"coshdh= {coshdh:.8g}")
    #     print(f"del_h= {del_H:.8g}")

    return TOF, o_type  # findTOF_b()


def keplerCOE(r0, v0, dt, mu):
    """
    Two body orbit propagation using classical orbital elements (Algorithm 7)

    Two body orbit propagation that uses a change to classical orbital elements
    and an update to true anomaly to find the new position and velocity
    vectors. For reference, see Algorithm 7 in Vallado Section 2.3.1 (pg 81).

    Parameters
    ----------
    r0: list, numpy.array, or numpyt.matrix (length 3)
        Initial position vector (km)
    v0: list, numpy.array, or numpyt.matrix (length 3)
        Initial velocity vector (km/s)
    dt: double
        Time to propagate (seconds)
    mu: double, optional, default=3.986004415E5 (Earth.mu in solarsys.py)
        Gravitational parameter (km^3/s^2)

    Returns
    -------
    r: numpy.matrix (3x1)
        New position vector (km)
    v: numpy.matrix (3x1)
        New velocity vector (km/s)
    """
    # [p, a, ecc, inc, raan, aop, t_anom, o_type] = rv2coe(r0, v0, mu)
    o_type, elements = rv2coe(r_vec=r0, v_vec=v0, mu=mu)
    sp, sma, ecc_mag, incl, raan, aop, t_anom, Lt0, w_bar, u_ = elements

    n = np.sqrt(mu / sma**3)

    if ecc_mag != 0.0:
        anom0 = true_to_anom(t_anom, ecc_mag)
    else:
        anom0 = t_anom

    if ecc_mag < 1.0:  # ecc < 1 -> elliptical or circular
        M0 = anom0 - ecc_mag * np.sin(anom0)
        M = M0 + n * dt
        E = kep_eqtnE(M, ecc_mag)
        if ecc_mag != 0.0:
            t_anom = eccentric_to_true(E, ecc_mag)
        else:  # circular
            t_anom = E
    elif ecc_mag == 1.0:  # ecc = 1.0 -> Parabolic
        M0 = anom0 + anom0**3 / 3
        B = kep_eqtnP(dt, sp)
        t_anom = parabolic_to_true(B)
    else:  # ecc > 1.0 -> hyperbolic
        M0 = ecc_mag * np.sinh(anom0) - anom0
        M = M0 + n * dt
        H = kep_eqtnH(M, ecc_mag)
        t_anom = hyperbolic_to_true(H, ecc_mag)

    r, v = coe2rv(sp, ecc_mag, incl, raan, aop, t_anom, mu)
    return r, v


def keplerUni(r0, v0, dt, mu=Earth.mu, tol=1e-6):
    """
    Two body orbit propagation with universal formulation.
        Kepler propagation, given  r0_vec, v0_vec, dt -> r1_vec, v1_vec.
        Universal formulation.  Vallado [4], example 2-4, p.96, algorithm 8, pp.94.

    Input Parameters:
    ----------
        r0  : numpy.array, float [km], initial position vector
        v0  : numpy.array, float [km/s], initial velocity vector
        dt  : float [s], time to propagate
        mu  : float, [km^3/s^2] optional, default=Earth.mu in solarsys.py
            Gravitational parameter
        tol : float, optional, default=1E-6
            Convergence criterion (iteration difference)
    Returns:
    -------
        r   : numpy.array, float [km], new position vector
        v   : numpy.array, float [km/s], new velocity vector
    Notes:
    ----------
        Vallado [2] Section 2.3, algorithm 8, p.93.
        Vallado [4] Section 2.3, algorithm 8, p.95.
    """

    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    alpha = -(v_mag**2) / mu + 2.0 / r_mag

    if alpha > 0.000001:  # elliptical or circular
        # initial chi estimate; ellipse
        chi0 = np.sqrt(mu) * dt * alpha

    elif alpha < -0.000001:  # hyperbolic
        a = 1 / alpha
        # initial chi estimate; hyperbolic
        chi0 = (
            np.sign(dt)
            * np.sqrt(-a)
            * np.log(
                -2.0
                * mu
                * alpha
                * dt
                / (
                    np.dot(r0.T, v0)
                    + np.sign(dt) * np.sqrt(-mu * a) * (a - r_mag * alpha)
                )
            )
        )
    else:  # parabolic
        h = np.cross(r0, v0, axis=0)
        h_mag = np.linalg.norm(h)
        p = h_mag * h_mag / mu
        s = 0.5 * np.arctan(1.0 / (3.0 * np.sqrt(mu / p) * dt))
        w = np.arctan((np.tan(s)) ** (1 / 3.0))
        # initial chi estimate; parabolic
        chi0 = np.sqrt(p) * 2.0 / np.tan(2.0 * w)

    diff = np.inf
    chi = chi0
    psi = 0  # initialize psi
    while np.abs(diff) > tol:
        psi = chi * chi * alpha
        c2, c3 = find_c2c3(psi)
        r = (
            chi**2 * c2
            + (np.dot(r0.T, v0) / np.sqrt(mu)) * chi * (1.0 - psi * c3)
            + r_mag * (1.0 - psi * c2)
        )
        chi_new = (
            chi
            + (
                np.sqrt(mu) * dt
                - chi**3 * c3
                - (np.dot(r0.T, v0) / np.sqrt(mu)) * chi**2 * c2
                - r_mag * chi * (1.0 - psi * c3)
            )
            / r
        )
        diff = np.abs(chi_new - chi)
        chi = chi_new

    f = 1.0 - (chi**2 / r_mag) * c2
    g = dt - (chi**3 / np.sqrt(mu)) * c3

    g_dot = 1.0 - (chi**2 / r) * c2
    f_dot = (np.sqrt(mu) / (r * r_mag)) * chi * (psi * c3 - 1.0)
    r_vec = f * r0 + g * v0
    v_vec = f_dot * r0 + g_dot * v0

    return r_vec, v_vec


def test_rv2coe():
    """
    Test cases:
        https://colorado.pressbooks.pub/introorbitalmechanics/chapter/copy-of-chapter-3__editing/
        Curtis [3] pp.212, example 4.3
        BMWS [1] pp.54, 2 examples
        Vallado [4] pp.116

    Returns
    -------
        None
    """
    rad2deg = 180 / math.pi
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    print(f"** Colorado Pressbooks; test rv2coe(): **")
    print(f"** Example 1: **")
    r0_vec = np.array([0, 0, 10000])  # [km]
    v0_vec = np.array([6, 0, 0])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)

    print(f"** Example 2: **")
    r0_vec = np.array([10000, 0, 0])  # [km]
    v0_vec = np.array([0, 4.464, -4.464])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)

    print(f"** Example 3: **")
    r0_vec = np.array([0, -7000, 0])  # [km]
    v0_vec = np.array([9, 0, 0])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)

    print(f"\n** Curtis [3]; test rv2coe(): **")
    print(f"** Example 4.3, pp.212: **")
    r0_vec = np.array([-6045, -3490, 2500])  # [km]
    v0_vec = np.array([-3.457, 6.618, 2.533])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)

    print(f"\n** BMWS [1]; test rv2coe(): **")
    print(f"** Example pp.54: **")
    r0_vec = np.array([12756.2, 0, 0])  # [km]
    v0_vec = np.array([0, 7.90537, 0])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)

    print(f"** Example pp.56: **")
    r0_vec = np.array([8750, 5100, 0])  # [km]
    v0_vec = np.array([-3.0, 5.2, 5.9])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)
    
    print(f"\n** Vallado [4]; test rv2coe(): **")
    print(f"** Example 2-5, pp.116: **")
    r0_vec = np.array([6524.834, 6862.875, 6448.296])  # [km]
    v0_vec = np.array([4.901327, 5.533756, -1.976341])  # [km/s]
    o_type, elements = rv2coe(r_vec=r0_vec, v_vec=v0_vec, mu=mu_earth_km)
    print_coe(o_type=o_type, elements=elements)

    return None


def Main():  # helps with my code editor navigation :--)
    return


# Test functions and class methods are called here.
if __name__ == "__main__":
    test_rv2coe()  #
