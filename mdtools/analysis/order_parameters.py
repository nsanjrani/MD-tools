import numpy as np


def fraction_of_contacts(cm: np.ndarray, ref_cm: np.ndarray) -> float:
    r"""Compute the fraction of contact between contact maps.

    Given two contact matices of equal dimensions, computes the
    fraction of entries which are equal. This is comonoly refered
    to as the fraction of contacts and in the case where ref_cm
    represents the native state this is the fraction of native contacts.

    Parameters
    ----------
    cm : np.ndarray
        A contact matrix.
    ref_cm : np.ndarray
        The reference contact matrix for comparison.

    Returns
    -------
    float
        The fraction of contacts (between 0 and 1).

    Notes
    -----
    This function can act on coo sparse matrices, so long as both
    `cm` and `ref_cm` are sparse.
    """
    return 1 - np.mean(cm != ref_cm)
