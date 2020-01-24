import numpy as np


def jaccard_coef(A, B):
    """
    calculate the jaccard coefficient of A and B .
    A and B should be of the same shape.
    :param A: binary image
    :param B: binary image
    :return: jaccard coefficient
    """

    if A.shape != B.shape:
        raise Exception("A and B should have the same shape")

    return np.logical_and(A, B).sum() / np.logical_or(A, B).sum()


def jaccard_coef_shift_invariant(A, B):
    """
    calculate the similarity under all possible relative translations.
    These two  images are aligned at the top-left corner (x =0, y = 0).
    B is fixed while A is being translated .
    :param A: binary image
    :param B: binary image
    :return: similarity value and A's relative translation to B
    """

    A_shape_y, A_shape_x = A.shape[: 2]
    B_shape_y, B_shape_x = B.shape[: 2]

    B_expanded = np.pad(B,
                        ((A_shape_y, A_shape_y), (A_shape_x, A_shape_x)),
                        constant_values = False)
    A_expanded = np.full_like(B_expanded, False)

    cartesian_prod = [(x, y)
                      for x in np.arange(A_shape_x + B_shape_x)[int(A_shape_x / 2) : -int(A_shape_x / 2)]
                      for y in np.arange(A_shape_y + B_shape_y)[int(A_shape_y / 2) : -int(A_shape_y / 2)]]

    j_coefs = np.zeros(len(cartesian_prod))
    for (x, y), ii in zip(cartesian_prod, np.arange(len(j_coefs))):
        A_expanded.fill(False)
        A_expanded[y: y + A_shape_y, x: x + A_shape_x] = A
        j_coefs[ii] = jaccard_coef(A_expanded, B_expanded)

    j_coefs_max = j_coefs.max()

    max_id = np.array(cartesian_prod)[np.where(j_coefs == j_coefs_max)]
    max_id_x = max_id[:, 0] - A_shape_x
    max_id_y = max_id[:, 1] - A_shape_y
    return j_coefs_max, max_id_x, max_id_y

