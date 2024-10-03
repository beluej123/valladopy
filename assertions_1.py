"""Parameter validation collection."""

import numpy as np


def assert_valid_parameters(mu, r1, r2, tof, M):
    """
    Check for safe solver input parameters.
    Input Parameters:
    ----------
        mu  : float, gravitational parameter.
        r1  : np.array, initial position vector.
        r2  : np.array, final position vector.
        tof : float, tof (time of flight).
        M   : int, number of revolutions
    Return:
    ----------
        True if all parameters are safe;
            else raise exception; terminates routine.
    """
    assert_positive_gravitational_parameter(mu)
    assert_valid_position_vectors(r1, r2)
    assert_positive_tof(tof)
    assert_positive_number_of_revolutions(M)

    return True


def assert_positive_gravitational_parameter(mu):
    """
    Verify positive gravitational parameter.
    Input Parameters:
    ----------
        mu: float, gravitational parameter
    Raises:
    ----------
        ValueError
    """
    # Check positive gravitational parameter
    if mu <= 0:
        raise ValueError("Gravitational parameter must be positive!")
    else:
        return True


def assert_valid_position_vector(r):
    """
    Verify position vector has proper dimensions and is not the null one.
    Input Parameters:
    ----------
        r: np.array, initial position vector.
    Raises:
    ----------
        ValueError
    """
    if r.shape != (3,):
        raise ValueError("Vector must be three-dimensional!")

    if np.all(r == 0):
        raise ValueError("Position vector cannot be the null vector [0,0,0]!")

    return True


def assert_valid_position_vectors(r1, r2):
    """
    Verify position vectors are safe in dimension and values.
    Input Parameters:
    ----------
        r1: np.array, initial position vector.
        r2: np.array, final position vector.
    Raises:
    ----------
        ValueError
    """
    # Check if position vectors have proper dimensions
    for r in [r1, r2]:
        assert_valid_position_vector(r1)

    # Check both vectors are different
    if np.all(np.equal(r1, r2)):
        raise ValueError("Initial and final position vectors cannot be equal!")

    return True


def assert_positive_tof(tof):
    """
    Check if time of flight is positive.
    Input Parameters:
    ----------
        tof: float, tof (time of flight).
    Raises:
    ----------
        ValueError
    """
    if tof <= 0:
        raise ValueError("Time of flight must be positive!")
    else:
        return True


def assert_positive_number_of_revolutions(M):
    """
    Verify positive number of revolutions (or zero).
    Input Parameters:
    ----------
        M: int, number of revolutions
    Raises:
    ----------
        ValueError
    """
    if M < 0:
        raise ValueError("Number of revolutions must be equal or greater than zero!")
    else:
        return True


def assert_nonzero_transfer_angle(dtheta):
    """
    Verify nonzero transfer angle.
    Input Parameters:
    ----------
        dtheta: float, transfer angle value.
    Raises:
    ----------
        ValueError
    """
    if dtheta == 0:
        raise ValueError("Transfer angle was found to be zero!")
    else:
        return True


def assert_transfer_angle_not_pi(dtheta):
    """
    Verify the transfer angle is not 180 degrees.
    Input Parameters:
    ----------
        dtheta: float, [rad] transfer angle value.
    Raises:
    ----------
        ValueError

    """
    if dtheta == np.pi:
        raise ValueError("Transfer angle was found to be 180 degrees!")
    else:
        return True
