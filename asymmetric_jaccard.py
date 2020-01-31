import numpy as np
from os.path import join
from matplotlib import pyplot as plt

asymmetric_jaccard_cache_folder = "./precomputed-similarities/asymmetric_jaccard"
asymmetric_jaccard_similarities = None
asymmetric_jaccard_x = None
asymmetric_jaccard_y = None
asymmetric_jaccard_diff = None
asymmetric_jaccard_diff_is_positive = None
asymmetric_jaccard_images = None
asymmetric_jaccard_similarities_initial_size = 150
asymmetric_jaccard_similarities_increment = 50


# The asymmetric jaccard cache follows the convention that
# all the cache entries, Cache[ii, jj], are from image[ii] to image[jj]
# which measures how much ii-th image is a subset of jj-th image.


def load_asymmetric_jaccard_cache(problem_name):
    global asymmetric_jaccard_cache_folder
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_images
    global asymmetric_jaccard_x
    global asymmetric_jaccard_y
    global asymmetric_jaccard_diff
    global asymmetric_jaccard_diff_is_positive
    global asymmetric_jaccard_similarities_initial_size

    cache_file_name = join(asymmetric_jaccard_cache_folder, problem_name + ".npz")

    try:
        cache_file = np.load(cache_file_name, allow_pickle = True)
    except FileNotFoundError:
        cache_file = None

    if cache_file is not None:
        asymmetric_jaccard_similarities = cache_file["asymmetric_jaccard_similarities"]
        cache_file.files.remove("asymmetric_similarities")

        asymmetric_jaccard_x = cache_file["asymmetric_jaccard_x"]
        cache_file.files.remove("asymmetric_jaccard_x")

        asymmetric_jaccard_y = cache_file["asymmetric_jaccard_y"]
        cache_file.files.remove("asymmetric_jaccard_y")

        asymmetric_jaccard_diff = cache_file["asymmetric_jaccard_diff"]
        cache_file.file.remove("asymmetric_jaccard_diff")

        asymmetric_jaccard_diff_is_positive = cache_file["asymmetric_jaccard_diff_is_positive"]
        cache_file.file.remove("asymmetric_jaccard_diff_is_positive")

        asymmetric_jaccard_images = []
        for img_arr_n in cache_file.files:
            asymmetric_jaccard_images.append(cache_file[img_arr_n])
    else:
        asymmetric_jaccard_similarities = np.full((asymmetric_jaccard_similarities_initial_size,
                                                   asymmetric_jaccard_similarities_initial_size),
                                                  np.nan, dtype = float)
        asymmetric_jaccard_x = np.full((asymmetric_jaccard_similarities_initial_size,
                                        asymmetric_jaccard_similarities_initial_size),
                                       None, dtype = object)
        asymmetric_jaccard_y = np.full((asymmetric_jaccard_similarities_initial_size,
                                        asymmetric_jaccard_similarities_initial_size),
                                       None, dtype = object)
        asymmetric_jaccard_diff = np.full((asymmetric_jaccard_similarities_initial_size,
                                           asymmetric_jaccard_similarities_initial_size),
                                          None, dtype = object)
        asymmetric_jaccard_diff_is_positive = np.full((asymmetric_jaccard_similarities_initial_size,
                                                       asymmetric_jaccard_similarities_initial_size),
                                                      None, dtype = object)
        asymmetric_jaccard_images = []


def save_asymmetric_jaccard_cache(problem_name):
    global asymmetric_jaccard_cache_folder
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_x
    global asymmetric_jaccard_y
    global asymmetric_jaccard_diff
    global asymmetric_jaccard_diff_is_positive
    global asymmetric_jaccard_images

    cache_file_name = join(asymmetric_jaccard_cache_folder, problem_name + ".npz")
    np.savez(cache_file_name,
             asymmetric_similarities = asymmetric_jaccard_similarities,
             asymmetric_jaccard_x = asymmetric_jaccard_x,
             asymmetric_jaccard_y = asymmetric_jaccard_y,
             asymmetric_jaccard_diff = asymmetric_jaccard_diff,
             asymmetric_jaccard_diff_is_positive = asymmetric_jaccard_diff_is_positive,
             *asymmetric_jaccard_images)


def asymmetric_jaccard_coef_same_shape(A, B):
    """
    calculate the asymmetric jaccard coefficient.
    sim_A = |A intersect B| / |A|
    sim_B = |A intersect B| / |B|
    sim= = max(sim_A, sim_B)
    A and B should be of the same shape.
    A and B can't be all white simultaneously.
    :param A: binary image
    :param B: binary image
    :return: asymmetric_jaccard coefficient, A_sum < B_sum
    """

    if A.shape != B.shape:
        raise Exception("A and B should have the same shape.")

    A_sum = A.sum()
    B_sum = B.sum()

    if 0 == A_sum or 0 == B_sum:
        raise Exception("A and B can't be all white simultaneously.")
    elif 0 == A_sum and 0 != B_sum:
        return 1, True
    elif 0 != A_sum and 0 == B_sum:
        return 1, False
    else:
        if A_sum < B_sum:
            return np.logical_and(A, B).sum() / A_sum, True
        else:
            return np.logical_and(A, B).sum() / B_sum, False


def asymmetric_jaccard_coef_naive(A, B):
    """
    max_diff is trimmed to its minimal box. (max_x, max_y) is the relative position of max_diff to A
    :param A:
    :param B:
    :return: a_j_coefs_max, max_x, max_y, max_diff
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

    a_j_coefs = np.zeros(len(cartesian_prod))
    is_A_smaller = np.full(len(cartesian_prod), False)
    for (x, y), ii in zip(cartesian_prod, np.arange(len(a_j_coefs))):
        A_expanded.fill(False)
        A_expanded[y: y + A_shape_y, x: x + A_shape_x] = A
        a_j_coefs[ii], is_A_smaller[ii] = asymmetric_jaccard_coef_same_shape(A_expanded, B_expanded)

    a_j_coefs_max = np.max(a_j_coefs)
    coef_argmax = np.where(a_j_coefs == a_j_coefs_max)[0]

    max_diff = []
    max_x = []
    max_y = []
    max_diff_is_positive = []
    for ii in coef_argmax:
        x, y = cartesian_prod[ii]
        A_is_smaller = is_A_smaller[ii]

        A_expanded.fill(False)
        A_expanded[y: y + A_shape_y, x: x + A_shape_x] = A

        if A_is_smaller:
            diff = np.logical_and(B_expanded,
                                  np.logical_not(A_expanded))
            diff_y, diff_x = np.where(diff)
            diff_x_min = diff_x.min()
            diff_x_max = diff_x.max() + 1
            diff_y_min = diff_y.min()
            diff_y_max = diff_y.max() + 1
            max_diff.append(diff[diff_y_min: diff_y_max, diff_x_min: diff_x_max])
            max_x.append(diff_x_min - x)
            max_y.append(diff_y_min - y)
            max_diff_is_positive.append(True)
        else:
            diff = np.logical_and(A_expanded,
                                  np.logical_not(B_expanded))
            diff_y, diff_x = np.where(diff)
            diff_x_min = diff_x.min()
            diff_x_max = diff_x.max() + 1
            diff_y_min = diff_y.min()
            diff_y_max = diff_y.max() + 1
            # This negation is important
            max_diff.append(np.logical_not(diff[diff_y_min: diff_y_max, diff_x_min: diff_x_max]))
            max_x.append(diff_x_min - x)
            max_y.append(diff_y_min - y)
            max_diff_is_positive.append(False)

    max_x = np.array(max_x)  # convert it to numpy array because want to broadcast
    max_y = np.array(max_y)  # across the entries in the list

    return a_j_coefs_max, max_x, max_y, max_diff, max_diff_is_positive


def asymmetric_jaccard_coef(A, B):
    """

    :param A: binary image
    :param B: binary image
    :return:
    """

    A_y, A_x = np.where(A)
    B_y, B_x = np.where(B)
    if 0 == len(A_y):
        if 0 == len(B_y):
            return 0, 0, 0, None  # A and B are all white images.
        else:
            return 1, 0, 0, B  # A is all white, but B is not.
    else:
        if 0 == len(B_y):
            return 0, 0, 0, None  # B is all white, but A is not.
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

    B_id = asymmetric_jaccard_image2index(B_trimmed)
    A_id = asymmetric_jaccard_image2index(A_trimmed)

    sim = asymmetric_jaccard_similarities[A_id, B_id]

    if np.isnan(sim):
        sim, x_trimmed, y_trimmed, diff_trimmed, diff_trimmed_is_positive = asymmetric_jaccard_coef_naive(A_trimmed,
                                                                                                          B_trimmed)
        asymmetric_jaccard_similarities[A_id, B_id] = sim
        asymmetric_jaccard_x[A_id, B_id] = x_trimmed
        asymmetric_jaccard_y[A_id, B_id] = y_trimmed
        asymmetric_jaccard_diff[A_id, B_id] = diff_trimmed
        asymmetric_jaccard_diff_is_positive[A_id, B_id] = diff_trimmed_is_positive
    else:
        x_trimmed = asymmetric_jaccard_x[A_id, B_id]
        y_trimmed = asymmetric_jaccard_y[A_id, B_id]
        diff_trimmed = asymmetric_jaccard_diff[A_id, B_id]
        diff_trimmed_is_positive = asymmetric_jaccard_diff_is_positive[A_id, B_id]

    x = x_trimmed + A_x_min  # the diff patch is relative to A's top-left corner
    y = y_trimmed + A_y_min  # which is different from jaccard_coef

    if 1 == len(x):
        return sim, x[0], y[0], diff_trimmed[0], diff_trimmed_is_positive[0]
    else:
        smallest = int(np.argmin(abs(x - A.shape[1] / 2) + abs(y - A.shape[0] / 2)))
        return sim, x[smallest], y[smallest], diff_trimmed[smallest], diff_trimmed_is_positive[smallest]


def asymmetric_jaccard_image2index(img):
    """
    TODO need to be improved in the future using hash or creating indexing
    :param img:
    :return:
    """
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_images
    global asymmetric_jaccard_x
    global asymmetric_jaccard_y
    global asymmetric_jaccard_diff
    global asymmetric_jaccard_diff_is_positive
    global asymmetric_jaccard_similarities_increment

    img_packed = np.packbits(img, axis = -1)
    ii = -1
    for ii in range(len(asymmetric_jaccard_images)):
        if img_packed.shape == asymmetric_jaccard_images[ii].shape and (
                img_packed == asymmetric_jaccard_images[ii]).all():
            return ii

    asymmetric_jaccard_images.append(img_packed)

    if len(asymmetric_jaccard_images) > asymmetric_jaccard_similarities.shape[0]:
        asymmetric_jaccard_similarities = np.pad(asymmetric_jaccard_similarities,
                                                 ((0, asymmetric_jaccard_similarities_increment),
                                                  (0, asymmetric_jaccard_similarities_increment)),
                                                 constant_values = np.nan)
        asymmetric_jaccard_x = np.pad(asymmetric_jaccard_x,
                                      ((0, asymmetric_jaccard_similarities_increment),
                                       (0, asymmetric_jaccard_similarities_increment)),
                                      constant_values = None)
        asymmetric_jaccard_y = np.pad(asymmetric_jaccard_y,
                                      ((0, asymmetric_jaccard_similarities_increment),
                                       (0, asymmetric_jaccard_similarities_increment)),
                                      constant_values = None)
        asymmetric_jaccard_diff = np.pad(asymmetric_jaccard_diff,
                                         ((0, asymmetric_jaccard_similarities_increment),
                                          (0, asymmetric_jaccard_similarities_increment)),
                                         constant_values = None)
        asymmetric_jaccard_diff_is_positive = np.pad(asymmetric_jaccard_diff_is_positive,
                                                     ((0, asymmetric_jaccard_similarities_increment),
                                                      (0, asymmetric_jaccard_similarities_increment)),
                                                     constant_values = None)

    return ii + 1
