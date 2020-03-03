import numpy as np
from os.path import join
from matplotlib import pyplot as plt

jaccard_cache_folder = "./precomputed-similarities/jaccard"
jaccard_similarities = None
jaccard_x = None
jaccard_y = None
jaccard_images = None
jaccard_similarities_initial_size = 150
jaccard_similarities_increment = 50


def load_jaccard_cache(problem_name):
    global jaccard_cache_folder
    global jaccard_similarities
    global jaccard_images
    global jaccard_x
    global jaccard_y
    global jaccard_similarities_initial_size

    cache_file_name = join(jaccard_cache_folder, problem_name + ".npz")

    try:
        cache_file = np.load(cache_file_name, allow_pickle = True)
    except FileNotFoundError:
        cache_file = None

    if cache_file is not None:
        jaccard_similarities = cache_file["similarities"]
        cache_file.files.remove("similarities")

        jaccard_x = cache_file["jaccard_x"]
        cache_file.files.remove("jaccard_x")

        jaccard_y = cache_file["jaccard_y"]
        cache_file.files.remove("jaccard_y")

        jaccard_images = []
        for img_arr_n in cache_file.files:
            jaccard_images.append(cache_file[img_arr_n])
    else:
        jaccard_similarities = np.full((jaccard_similarities_initial_size, jaccard_similarities_initial_size),
                                       np.nan, dtype = float)
        jaccard_x = np.full((jaccard_similarities_initial_size, jaccard_similarities_initial_size),
                            np.nan, dtype = int)
        jaccard_y = np.full((jaccard_similarities_initial_size, jaccard_similarities_initial_size),
                            np.nan, dtype = int)
        jaccard_images = []


def save_jaccard_cache(problem_name):
    global jaccard_cache_folder
    global jaccard_similarities
    global jaccard_images
    global jaccard_x
    global jaccard_y

    cache_file_name = join(jaccard_cache_folder, problem_name + ".npz")
    np.savez(cache_file_name,
             similarities = jaccard_similarities,
             jaccard_x = jaccard_x,
             jaccard_y = jaccard_y,
             *jaccard_images)


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

    A_B_sum = np.logical_or(A, B).sum()

    if 0 == A_B_sum:
        return 1
    else:
        return np.logical_and(A, B).sum() / A_B_sum


def jaccard_coef_naive(A, B):
    """

    :param A:
    :param B:
    :return: j_coef, A_to_B_x, A_to_B_y
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

    j_coef = np.max(j_coefs)
    coef_argmax = np.where(j_coefs == j_coef)[0]

    if 1 == len(coef_argmax):
        ii = coef_argmax[0]
        x, y = cartesian_prod[ii]
    else:
        center_y = (A_shape_y * 2 + B_shape_y) / 2
        center_x = (A_shape_x * 2 + B_shape_x) / 2

        most_centered_ii = -1
        smallest_dist2center = np.inf
        for ii in coef_argmax:
            x_tmp, y_tmp = cartesian_prod[ii]
            dist2center = abs(x_tmp + A_shape_x / 2 - center_x) + abs(y_tmp + A_shape_y / 2 - center_y)
            if dist2center < smallest_dist2center:
                smallest_dist2center = dist2center
                most_centered_ii = ii

        x, y = cartesian_prod[most_centered_ii]

    A_to_B_x = x - A_shape_x
    A_to_B_y = y - A_shape_y

    return j_coef, int(A_to_B_x), int(A_to_B_y)


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

    global jaccard_similarities
    global jaccard_x
    global jaccard_y

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

    B_id = jaccard_image2index(B_trimmed)
    A_id = jaccard_image2index(A_trimmed)

    sim = jaccard_similarities[A_id, B_id]

    if np.isnan(sim):
        sim, A_to_B_trimmed_x, A_to_B_trimmed_y = jaccard_coef_naive(A_trimmed, B_trimmed)
        jaccard_similarities[A_id, B_id] = sim
        jaccard_x[A_id, B_id] = A_to_B_trimmed_x
        jaccard_y[A_id, B_id] = A_to_B_trimmed_y
    else:
        A_to_B_trimmed_x = jaccard_x[A_id, B_id]
        A_to_B_trimmed_y = jaccard_y[A_id, B_id]

    A_to_B_x = A_to_B_trimmed_x - A_x_min + B_x_min
    A_to_B_y = A_to_B_trimmed_y - A_y_min + B_y_min

    return sim, A_to_B_x, A_to_B_y


def jaccard_image2index(img):
    """
    TODO need to be improved in the future using hash or creating indexing
    :param img:
    :return:
    """
    global jaccard_similarities
    global jaccard_images
    global jaccard_x
    global jaccard_y
    global jaccard_similarities_increment

    img_packed = np.packbits(img, axis = -1)
    ii = -1
    for ii in range(len(jaccard_images)):
        if img_packed.shape == jaccard_images[ii].shape and (img_packed == jaccard_images[ii]).all():
            return ii

    jaccard_images.append(img_packed)

    if len(jaccard_images) > jaccard_similarities.shape[0]:
        jaccard_similarities = np.pad(jaccard_similarities,
                                      ((0, jaccard_similarities_increment), (0, jaccard_similarities_increment)),
                                      constant_values = np.nan)
        jaccard_x = np.pad(jaccard_x,
                           ((0, jaccard_similarities_increment), (0, jaccard_similarities_increment)),
                           constant_values = np.nan)
        jaccard_y = np.pad(jaccard_y,
                           ((0, jaccard_similarities_increment), (0, jaccard_similarities_increment)),
                           constant_values = np.nan)

    return ii + 1
