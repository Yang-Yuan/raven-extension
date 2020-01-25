import numpy as np


def jaccard_coef_same_shape(A, B):
    """
    calculate the jaccard coefficient of A and B .
    A and B should be of the same shape.
    :param A: binary image
    :param B: binary image
    :return: jaccard coefficient
    """

    if A.shape != B.shape:
        raise Exception("A and B should have the same shape")

    union = np.logical_or(A, B).sum()

    if 0 == union:
        return 1
    else:
        return np.logical_and(A, B).sum() / union


def jaccard_coef_naive(A, B):
    """

    :param A:
    :param B:
    :return: as jaccard_coef, but return all the translations of A
    """
    A_shape_y, A_shape_x = A.shape[: 2]
    B_shape_y, B_shape_x = B.shape[: 2]

    B_expanded = np.pad(B,
                        ((A_shape_y, A_shape_y), (A_shape_x, A_shape_x)),
                        constant_values = False)
    A_expanded = np.full_like(B_expanded, False)

    cartesian_prod = [(x, y)
                      for x in np.arange(1, A_shape_x + B_shape_x)
                      for y in np.arange(1, A_shape_y + B_shape_y)]

    j_coefs = np.zeros(len(cartesian_prod))
    for (x, y), ii in zip(cartesian_prod, np.arange(len(j_coefs))):
        A_expanded.fill(False)
        A_expanded[y: y + A_shape_y, x: x + A_shape_x] = A
        j_coefs[ii] = jaccard_coef_same_shape(A_expanded, B_expanded)

    j_coefs_max = np.max(j_coefs)

    max_id = np.array(cartesian_prod)[np.where(j_coefs == j_coefs_max)]
    max_id_x = max_id[:, 0] - A_shape_x
    max_id_y = max_id[:, 1] - A_shape_y

    return j_coefs_max, max_id_x, max_id_y


def jaccard_coef(A, B):
    """

    :param A: binary image
    :param B: binary image
    :return: maximum jaccard coefficient over all possible relative translations.
             And how A should be moved if we fixed by B, and consider the most top-left
             pixel of B as the origin of axes.
             If multiple optimal translations exists, choose the one that corresponds to
             the smallest translation.
             x is the second dim and y is the first dim
    """

    A_y, A_x = np.where(A)
    B_y, B_x = np.where(B)
    if 0 == len(A_y):
        if 0 == len(B_y):
            return 1, 0, 0  # A and B are all white images.
        else:
            return 0, 0, 0  # A is all white, but B is not.
    else:
        if 0 == len(B_y):
            return 0, 0, 0  # B is all white, but A is not.
        else:
            A_y_min = A_y.min()
            A_x_min = A_x.min()
            A_y_max = A_y.max() + 1
            A_x_max = A_x.max() + 1
            A_trimmed = A[A_y_min: A_y_max, A_x_min: A_x_max]

            B_y_min = B_y.min()
            B_x_min = B_x.min()
            B_y_max = B_y.max() + 1
            B_x_max = B_x.max() + 1
            B_trimmed = B[B_y_min: B_y_max, B_x_min: B_x_max]

    sim, x_trimmed, y_trimmed = jaccard_coef_naive(A_trimmed, B_trimmed)
    x = x_trimmed - A_x_min + B_x_min
    y = y_trimmed - A_y_min + B_y_min

    if 1 == len(x):
        return sim, x[0], y[0]
    else:
        smallest = np.argmin(abs(x) + abs(y))
        return sim, x[smallest], y[smallest]
