"""
Created on Sat Aug 20 17:34:45 2016, @author: Alex
Edits 2024-August +, by Jeff Belue.
Support various time functions.  Be mindful of the various scales
    to measure time; see comments in Notes: below.


References:
----------
    See references.py for references list
Notes:
----------
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally, angles are saved in [rad], distance [km].
    
    Note some common timescales; Vallado [4] chap.3, & skyfield:
        From skyfield, https://rhodesmill.org/skyfield/time.html):
        UTC — Coordinated Universal Time (“Greenwich Time”)
        UT1 — Universal Time, usually a Julian date
        TAI — International Atomic Time
        TT — Terrestrial Time
        TDB — Barycentric Dynamical Time (JPL's T_eph)
"""

import math

import numpy as np


def g_date2jd(yr, mo, d, hr=0, minute=0, sec=0.0, leap_sec=False) -> float:
    """
    Convert Gregorian/Julian date & time (yr, month, day, hour, second) to julian date.
    This function accomadates both Julian and Gregorian calendars and allows
        negative years (BCE).
    To me, the computer implementation details in the general literature need to
        be more specific.  Vallado [4] algorithm 14, does not address the
        complete julian range including BCE.  For details on addressing the full
        Julian date range note; Wertz [5], https://en.wikipedia.org/wiki/Julian_day,
        Meeus [6], and Duffett-Smith [7] for a good computer step-by-step implementation.
    Valid for any time system (UT1, UTC, AT, etc.) but should be identified to
        avoid confusion.  This routine superceeds Vallado [4], algorithm 14.
    Input Parameters:
    ----------
        yr       : int, four digit year
        mo       : int, month
        d        : int, day of month
        hr       : int, hour (24-hr based)
        minute   : int, minute
        sec      : float, seconds
        leap_sec : boolean, optional, default = False
                   Flag if time is during leap second
    Returns:
    -------
        jd       : float date/time as julian date
    Notes:
    ----------
        Remember, the Gregorian calendar starts 1582-10-15 (Friday); skips 10
        days...  Also note, The Gregorian calendar is off by 26 seconds per
        year.  By 4909 it will be a day ahead of the solar year.
    """
    import math as ma

    # commented code, below, is superceeded.
    #   the new code allows the "full" julian range; negative dates.
    # x = (7 * (yr + np.trunc((mo + 9) / 12))) / 4.0
    # y = (275 * mo) / 9.0
    # if leap_sec:
    #     t = 61.0
    # else:
    #     t = 60.0
    # z = (sec / t + minute) / 60.0 + hr
    # jd = 367 * yr - np.trunc(x) + np.trunc(y) + d + 1721013.5 + z / 24.0
    # verify year is > -4712
    if yr <= (-4713):
        print(f"** Year must be > -4712 for this algorithm; g_date2jd(). **")
        raise ValueError("Year must be > -4712.")

    # verify hr, minute, seconds are in bounds
    if (hr >= 24) or (minute > 60) or (sec >= 60):
        print(f"** Error in g_date2jd() function. **")
        print(f"** hours, minutes, or seconds out of bounds. **")
        raise ValueError("hours, minutes, or seconds out of bounds; g_date2jd().")

    yr_d = yr
    if mo < 3:
        yr_d = yr - 1
    mo_d = mo
    if mo < 3:
        mo_d = mo + 12

    a_ = ma.trunc(yr_d / 100)
    b_ = 0.0  # in the Julian calendar, b=0
    # check for gregorian calendar date
    if (
        (yr > 1582)
        or ((yr == 1582) and (mo > 10))
        or ((yr == 1582) and (mo == 10) and (d > 15))
    ):
        b_ = 2 - a_ + ma.trunc(a_ / 4)
    if yr_d < 0:
        c_ = ma.trunc((365.25 * yr_d) - 0.75)
    else:
        c_ = ma.trunc(365.25 * yr_d)

    d_ = ma.trunc(30.6001 * (mo_d + 1))
    d1 = d + (hr / 24) + (minute / 1440) + (sec / 86400)
    jd = b_ + c_ + d_ + d1 + 1720994.5

    return jd


def jd_convTime(yr, mo, d, hr=0, min=0, sec=0.0, c_type=0):
    """
    (1) Calculate julian date (jd) from date & time (yr, month, day, hour, second).
    (2) Converts jd to other time fromats:
        if c_type=0, find julian centuries from J2000.0 TT.
    Easy to add other conversion types, as needed.  Vallado [4], section 3.5,
        p.196, algorithm 16.
    Valid for any time system (UT1, UTC, AT, etc.) but should be identified to
        avoid confusion.
    Input Parameters:
    ----------
        yr     : int, four digit year
        mo     : int, month
        d      : int, day of month
        hr     : int, hour (24-hr based)
        minute : int, minute
        sec    : float, seconds
        c_type : int, conversion type
                c_type=0, julian centuries from J2000.0 TT
    Returns:
    -------
        jd        : float, date/time as julian date
        jd_cJ2000 : float, julian centuries from J2000.0 TT
    """
    jd = g_date2jd(yr, mo, d, hr=hr, minute=min, sec=sec)
    jd_cJ2000 = 0.0  # make sure variable is assigned
    if c_type == 0:
        jd_cJ2000 = (jd - 2451545.0) / 36525.0
    else:
        print(f"Unknown time conversion type; function jd_convTime().")
    return jd, jd_cJ2000


def jd2g_date(jd):
    """
    Convert julian date to gregorian Date (y, mo, d, h, m, s).
        Algorithm 22 in Vallado (Fourth Edition) Section 3.6.6, pg 202.

    Input Parameters:
    ----------
        jd: float, Julian Date
    Returns:
    ----------
        (year, month, day, hours, minutes, seconds): tuple
    """
    t1900 = (jd - 2415019.5) / 365.25
    year = 1900 + np.trunc(t1900)
    leap_years = np.trunc((year - 1900 - 1) * 0.25)
    days = (jd - 2415019.5) - ((year - 1900) * 365.0 + leap_years)

    if days < 1.0:
        year = year - 1
        leap_years = np.trunc((year - 1900 - 1) * 0.25)
        days = (jd - 2415019.5) - ((year - 1900) * 365.0 + leap_years)

    (month, day, hours, minutes, seconds) = days2ymdhms(days, year)
    return (year, month, day, hours, minutes, seconds)


def find_gmst(jd_ut1):
    """
    Find Greenwich Mean Sidereal Time (GMST), with supplied UT1 Julian date.
        Method 1, uses Julian calculation; method 2 has limited use.
        Vallado [4] algorithm 15, p.190. Associated example 3-5, pp.190.

    Input Parameters:
    ----------
        jd_ut1     : float [day], UT1 julian date

    Returns
    -------
        theta_gmst : float [deg], Greenwich Mean Sidereal Time
    """
    t_ut1 = (jd_ut1 - 2451545.0) / 36525.0
    # print(f"t_ut1=, {t_ut1}") # troubleshooting print()
    theta_gmst = (
        67310.54841
        + (876600 * 3600.0 + 8640184.812866) * t_ut1
        + 0.093104 * t_ut1 * t_ut1
        - 6.2e-6 * t_ut1 * t_ut1 * t_ut1
    )
    theta_gmst = math.fmod(theta_gmst, 86400.0)
    theta_gmst /= 240.0
    if theta_gmst < 0:
        theta_gmst += 360.0
    return theta_gmst


def find_lst(theta_gmst, lon):
    """
    Find the Local Sidereal Time (LST) for a supplied GMST and Longitude.
    Vallado [4] algorithm 15, p.190. Associated example 3-5, pp.190.

    Input Parameters:
    ----------
        theta_gmst : float [deg] Greenwich Mean Sidereal Time (GMST)
        longitude  : float [deg] Site longitude

    Returns
    -------
        theta_lst      : float [deg] local sidereal time (lst)
    """

    theta_lst = theta_gmst + lon
    return theta_lst


def dms2rad(degrees, minutes, seconds):
    """
    Converts degrees, minutes, seconds to radians.

    Converts degrees, minutes, seconds to radians. For reference, see Algorithm
    17 in Vallado (Fourth Edition), Section 3.5 pg 197

    Parameters
    ----------
    degrees: double
        degrees part of angle
    minutes: double
        minutes part of angle
    seconds: double
        seconds part of angle

    Returns
    -------
    rad: double
        angle in radians
    """
    rad = (degrees + minutes / 60.0 + seconds / 3600.0) * (math.pi / 180.0)
    return rad


def rad2dms(rad):
    """Converts an angle in radians to Degrees, Minutes, Seconds

    Converts and angle in radians to Degrees, Minutes, Seconds. For reference,
    see Algorithm 18 in Vallado (Fourth Edition), Section 3.5 pg 197

    Parameters
    ----------
    rad: double
        Angle in radians

    Returns
    -------
    (degrees, minutes, seconds): tuple
        Angle parts in degrees, minutes, and seconds
    """
    temp = rad * (180.0 / math.pi)
    degrees = np.trunc(temp)
    minutes = np.trunc((temp - degrees) * 60.0)
    seconds = (temp - degrees - minutes / 60.0) * 3600.0
    return (degrees, minutes, seconds)


def hms2rad(hours, minutes, seconds):
    """Converts a time (hours, minutes, and seconds) to an angle (radians).

    Converts a time (hours, minutes, and seconds) to an angle in radians. For
    reference, see Algorithm 19 in Vallado (Fourth Edition), Section 3.5 pg 198

    Parameters
    ----------
    hours: double
        hours portion of time
    minutes: double
        minutes portion of time
    seconds: double
        seconds portion of time

    Returns
    -------
    rad: double
        angle representation in radians
    """
    rad = 15 * (hours + minutes / 60.0 + seconds / 3600.0) * math.pi / 180.0
    return rad


def rad2hms(rad):
    """
    Converts an angle (in radians) to a time (in hours, minutes, seconds)

    Converts an angle (radians) to a time (hours, minutes, seconds). For
    refrence, see Algorithm 20 in Vallado (Fourth Edition), Section 3.5 pg 198

    Parameters
    ----------
    rad: double
        angle in radians

    Returns
    -------
    (hours, minutes, seconds): tuple
        time in hours, minutes, and seconds
    """
    temp = rad * 180.0 / (15.0 * math.pi)
    hours = np.trunc(temp)
    minutes = np.trunc((temp - hours) * 60.0)
    seconds = (temp - hours - minutes / 60.0) * 3600.0
    return (hours, minutes, seconds)


def dec_deg2hms(dec_deg):
    """
    Convert decimal degrees to hours, minutes, seconds.

    Input Parameters:
    ----------
        dec_deg: float, decimal degrees

    Returns:
    -------
        hours, minutes, seconds : tuple float
    """
    hours = int(dec_deg / 15)
    minutes = int((dec_deg % 15) * 4)
    seconds = ((dec_deg % 15) * 4 - minutes) * 60
    return (hours, minutes, seconds)


def is_leap_year(yr):
    """
    Determines if the year is a leap year

    Determines if the year is a leap year. For reference, see Section 3.6.4 of
    Vallado (Fourth Edition), pg 200

    Parameters
    ----------
    yr: int
        Four digit year

    Returns
    -------
    is_leap: boolean
        Flag indicating if the year is a leap year
    """
    if np.remainder(yr, 4) != 0:
        return False
    else:
        if np.remainder(yr, 100) == 0:
            if np.remainder(yr, 400) == 0:
                return True
            else:
                return False
        else:
            return True


def time2hms(time_in_seconds):
    """
    Computes the time in Hours, Minutes, Seconds from seconds into a day

    Computes the time in Hours, Minutes, and Seconds from the time expressed
    as seconds into a day.  For reference, see Algorithm 21 in Vallado
    (Fourth Edition), Section 3.6.3 pg 199

    Parameters
    ----------
    time_in_seconds: double
        time expessed as seconds into a day

    Returns
    -------
    (hours, minutes, seconds): tuple
        time expressed as hours, minutes, seconds
    """
    temp = time_in_seconds / 3600.0
    hours = np.trunc(temp)
    minutes = np.trunc((temp - hours) * 60)
    seconds = (temp - hours - minutes / 60) * 3600

    return (hours, minutes, seconds)


def hms2time(hours, minutes, seconds):
    """
    Computes the time in seconds into the day from hours, minutes, seconds

    Computes the time in seconds into the day from the time expressed as
    hours, minutes, seconds. For reference, see Section 3.6.3 in Vallado
    (Foiurth Edition), pg 199

    Parameters
    ----------
    hours: double
        hours part of the time
    minutes: double
        minutes part of the time
    seconds: double
        seconds part of the time

    Returns
    -------
    tau: double
        the time as seconds into the day
    """
    tau = 3600.0 * hours + 60.0 * minutes + seconds
    return tau


def ymd2doy(year, month, day):
    """
    Computes the day of the year from the year, month, and day

    Computes the day of the year from the year, month, and day of a date. For
    reference, see Section 3.6.4 of Vallado (Fourth Edition) pg 200

    Parameters
    ----------
    year: int
        year (needed for leap year test)
    month: int
        month
    day: int
        day

    Returns
    -------
    doy: int
        day of the year
    """
    mos = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_year(year):
        mos[1] = 29

    idx = month - 1

    doy = np.sum(mos[:idx]) + day
    return doy


def doy2ymd(day_of_year, year):
    """
    NOT TESTED !! not sure about proper exit/break of while loop.
    Computes the month and day, given the year and day of the year.
    Vallado (Fourth Edition), section 3.6.4, pg 200.

    Input Parameters:
    ----------
        day_of_year: int, the day of the year
        year: int, the year (needed for leap year test)

    Returns:
    ----------
        (month, day): tuple, the month and day of the date
    """
    mos = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_year(year):
        mos[1] = 29

    temp = 0
    idx = 0
    while temp < day_of_year:
        temp += mos[idx]

        if temp >= day_of_year:
            month = idx + 1
            if month == 1:
                day = day_of_year
            else:
                day = day_of_year - np.sum(mos[:idx])
            # commented out return, not great practice;
            #   does not allow proper linting
            # return (month, day)
            break

        idx += 1
    return (month, day)


def ymdhms2days(year, month, day, hour, minutes, seconds):
    """
    Computes the decimal day for a give date (y, m, d) and time (h, m, s)

    Computes the decimal day for a given date (years, months, days) and time
    (hours, minutes, seconds). For reference, see Section 3.6.5 in Vallado
    (Fourth Edition) pg 201

    Parameters
    ----------
    year: int
        year (YYYY)
    month: int
        month
    day: int
        day
    hour: int
        hour
    minutes: int
        minutes
    seconds: double
        seconds

    Returns
    -------
    days: double
        decimal day (and partial day) of the year
    """
    doy = ymd2doy(year, month, day)
    days = doy + hour / 24.0 + minutes / 1440.0 + seconds / 86400.0
    return days


def days2ymdhms(days, year):
    """
    Computes the month, day, hrs, mins, and secs from the year and decimal day

    Computes the month, day, hours, minutes, and seconds from the year and
    decimal day of the year (including partial day). For reference, see
    Section 3.6.5 in Vallado (Fourth Edition), pg 201

    Parameters
    ----------
    days: double
        decimal day (and partial day) of the year
    year: int
        year (YYYY)

    Returns
    -------
    (month, day, hours, minutes, seconds): tuple
        the month, day, hours, minutes, and seconds of the date/time
    """
    doy = np.trunc(days)
    (month, day) = doy2ymd(doy, year)

    tau = (days - doy) * 86400.0
    (hours, minutes, seconds) = time2hms(tau)

    return (month, day, hours, minutes, seconds)


def test_julian_date():
    """
    Test julian date function called for in Curtis [3] p.277, example 5.4.
    The julian date is tricky if you need negative years (BCE).
        Neither Curtis [3] nor Vallado [4] implementations cover BCE.
        g_date2jd() superceeds Curtis [3] and Vallado [4], algorithm 14.
    Given:
        calendar date : 2004-05-12 14:45:30 UT
    Find:
        julian date
    Notes:
    ----------
        References: (see references.py for references list)
    Return:
    -------
        None
    """
    print(f"\nTest g_date2jd() function, Curtis example 5.4:")
    date_UT = [2004, 5, 12, 14, 45, 30]  # [UT] date/time python list
    date_UT1 = [1600, 12, 31, 0, 0, 0]  # [UT] date/time python list
    date_UT2 = [837, 4, 10, 7, 12, 0]  # [UT] date/time python list
    date_UT3 = [0, 1, 1, 0, 0, 0]  # [UT] date/time python list
    date_UT4 = [-1001, 8, 17, 21, 36, 0]  # [UT] date/time python list
    date_UT5 = [-4712, 1, 1, 12, 0, 0]  # [UT] date/time python list

    yr, mo, d, hr, minute, sec = date_UT
    JD = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec)
    print(f"date_UT= {date_UT}")
    print(f"julian_date= {JD:.10g}")

    yr, mo, d, hr, minute, sec = date_UT1
    JD1 = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec)
    print(f"\ndate_UT= {date_UT1}")
    print(f"julian_date= {JD1:.8g}")

    yr, mo, d, hr, minute, sec = date_UT2
    JD2 = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec)
    print(f"date_UT= {date_UT2}")
    print(f"julian_date= {JD2:.8g}")

    yr, mo, d, hr, minute, sec = date_UT3
    JD3 = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec)
    print(f"date_UT= {date_UT3}")
    print(f"julian_date= {JD3:.8g}")

    yr, mo, d, hr, minute, sec = date_UT4
    JD4 = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec)
    print(f"date_UT= {date_UT4}")
    print(f"julian_date= {JD4:.8g}")

    yr, mo, d, hr, minute, sec = date_UT5
    JD5 = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec)
    print(f"date_UT= {date_UT5}")
    print(f"julian_date= {JD5:.8g}")

    return None


def main():
    # placeholder at the end of the file; helps my edit navigation
    return None


# use the following to test/examine functions
if __name__ == "__main__":

    test_julian_date()  # test
