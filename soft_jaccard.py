import numpy as np
import utils

A_x_max = 60
A_y_max = 60
B_x_max = 60
B_y_max = 60
master_distance = np.array([[[[???
                               for A_x in range(B_y_max)]
                              for A_y in range(B_x_max)]
                             for B_x in range(A_y_max)]
                            for B_y in range(A_x_max)])


def soft_jaccard_coef_internal(A_coords, B_coords):

    x_min = min(A_coords[:, 1].min(), B_coords[:, 1].min())
    y_min = min(A_coords[:, 0].min(), B_coords[:, 0].min())

    A_coords = A_coords - [y_min, x_min]
    B_coords = B_coords - [y_min, x_min]

    A_x = np.full((len(A_coords), len(B_coords)), A_coords[:, [0]])
    A_y = np.full((len(A_coords), len(B_coords)), A_coords[:, [1]])
    B_x = np.full((len(A_coords), len(B_coords)), B_coords[:, 0])
    B_y = np.full((len(A_coords), len(B_coords)), B_coords[:, 1])

    dist = master_distance[A_x, A_y, B_x, B_y]

    return None


def soft_jaccard_naive_cross(hrz, vtc):
    hrz_shape_y, hrz_shape_x = hrz.shape
    vtc_shape_y, vtc_shape_x = vtc.shape

    padding_y = int(vtc_shape_y * 0.25)
    padding_x = int(hrz_shape_x * 0.25)

    delta_xs = list(range(-padding_x, vtc_shape_x - hrz_shape_x + 1 + padding_x))
    delta_ys = list(range(-padding_y, hrz_shape_y - vtc_shape_y + 1 + padding_y))

    length = int(len(delta_xs) * len(delta_ys))
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    j_coefs = np.zeros(length, dtype = np.float)

    hrz_coords = np.argwhere(hrz)
    vtc_coords = np.argwhere(vtc)

    kk = 0
    for delta_x in delta_xs:
        hrz_coords_shifted = hrz_coords + [0, delta_x]
        for delta_y in delta_ys:
            vtc_coords_shifted = vtc_coords + [delta_y, 0]
            coords[kk] = [delta_x, delta_y]
            j_coefs[kk] = soft_jaccard_coef_internal(hrz_coords_shifted, vtc_coords_shifted)
            kk += 1

    return coords, j_coefs


def soft_jaccard_naive_embed(frgd, bkgd):
    bgd_shape_y, bgd_shape_x = bkgd.shape
    fgd_shape_y, fgd_shape_x = frgd.shape

    padding_y = int(fgd_shape_y * 0.25)
    padding_x = int(fgd_shape_x * 0.25)

    delta_xs = list(range(-padding_x, bgd_shape_x - fgd_shape_x + 1 + padding_x))
    delta_ys = list(range(-padding_y, bgd_shape_y - fgd_shape_y + 1 + padding_y))

    length = int(len(delta_xs) * len(delta_ys))
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    j_coefs = np.zeros(length, dtype = np.float)

    frgd_coords = np.argwhere(frgd)
    bkgd_coords = np.argwhere(bkgd)

    kk = 0
    for delta_x in delta_xs:
        for delta_y in delta_ys:
            coords[kk] = [delta_x, delta_y]
            frgd_coords_shifted = frgd_coords + [delta_y, delta_x]
            j_coefs[kk] = soft_jaccard_coef_internal(frgd_coords_shifted, bkgd_coords)
            kk += 1

    return coords, j_coefs


def soft_jaccard_naive(A, B):
    A_shape_y, A_shape_x = A.shape
    B_shape_y, B_shape_x = B.shape

    if A_shape_x <= B_shape_x and A_shape_y <= B_shape_y:
        sj_coefs, A_to_B_coords = soft_jaccard_naive_embed(A, B)
    elif A_shape_x > B_shape_x and A_shape_y > B_shape_y:
        sj_coefs, A_to_B_coords = soft_jaccard_naive_embed(B, A)
    elif A_shape_x <= B_shape_x and A_shape_y > B_shape_y:
        sj_coefs, A_to_B_coords = soft_jaccard_naive_cross(A, B)
    else:
        sj_coefs, A_to_B_coords = soft_jaccard_naive_cross(B, A)

    sj_coef = np.max(sj_coefs)
    coef_argmax = np.where(sj_coefs == sj_coef)[0]

    if 1 == len(coef_argmax):
        ii = coef_argmax[0]
        A_to_B_x, A_to_B_y = A_to_B_coords[ii]
    else:

        A_center_x = (A_shape_x - 1) / 2
        A_center_y = (A_shape_y - 1) / 2
        B_center_x = (B_shape_x - 1) / 2
        B_center_y = (B_shape_y - 1) / 2

        closest_center_ii = -1
        smallest_center_dist = np.inf
        for ii in coef_argmax:
            x, y = A_to_B_coords[ii]
            center_dist = abs(x + A_center_x - B_center_x) + abs(y + A_center_y - B_center_y)
            if center_dist < smallest_center_dist:
                closest_center_ii = ii
                smallest_center_dist = center_dist

        A_to_B_x, A_to_B_y = A_to_B_coords[closest_center_ii]

    return sj_coef, int(A_to_B_x), int(A_to_B_y)


def soft_jaccard(A, B):
    A_tr, tr_to_A_x, tr_to_A_y = utils.trim_binary_image(A, coord = True)
    B_tr, tr_to_B_x, tr_to_B_y = utils.trim_binary_image(B, coord = True)

    sj, A_tr_to_B_tr_x, A_tr_to_B_tr_x = soft_jaccard_naive(A_trimmed, B_trimmed)