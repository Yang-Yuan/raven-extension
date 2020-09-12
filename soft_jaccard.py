import matplotlib.pyplot as plt
import copy
import numpy as np
import utils
import norm

norm.load_norm_to_p()

# alpha = 0.007 for ACE problems

alpha = 0.1 # for UT


def soft_partition(PR, order = None):
    """
    For now, assume ascending order as the significant order.
    :param PR:
    :param order: for future use
    :return:
    """

    A_len, B_len = PR.shape

    A_to_B_argmin = [np.argwhere(row == np.min(row)).flatten().tolist() for row in PR]
    B_to_A_argmin = [np.argwhere(col == np.min(col)).flatten().tolist() for col in PR.transpose()]

    AB_A_ids = []
    AB_B_ids = []
    AA_1_ids = []
    AA_2_idss = []
    BB_1_ids = []
    BB_2_idss = []

    for A_id in range(A_len):
        for B_id in A_to_B_argmin[A_id]:
            if A_id in B_to_A_argmin[B_id]:
                AB_A_ids.append(A_id)
                AB_B_ids.append(B_id)

    for A_1_id in range(A_len):
        if A_1_id not in AB_A_ids:
            A_2_id = []
            for B_id in A_to_B_argmin[A_1_id]:
                A_2_id.extend(B_to_A_argmin[B_id])
            AA_1_ids.append(A_1_id)
            AA_2_idss.append(np.unique(A_2_id).tolist())

    for B_1_id in range(B_len):
        if B_1_id not in AB_B_ids:
            B_2_id = []
            for A_id in B_to_A_argmin[B_1_id]:
                B_2_id.extend(A_to_B_argmin[A_id])
            BB_1_ids.append(B_1_id)
            BB_2_idss.append(np.unique(B_2_id).tolist())

    return AB_A_ids, AB_B_ids, AA_1_ids, AA_2_idss, BB_1_ids, BB_2_idss


def soft_jaccard_coef_internal(A_coords, B_coords, asymmetric = False):
    x_min = min(A_coords[:, 1].min(), B_coords[:, 1].min())
    y_min = min(A_coords[:, 0].min(), B_coords[:, 0].min())

    A_coords = A_coords - [y_min, x_min]
    B_coords = B_coords - [y_min, x_min]

    dist_AB = norm.get_distance_matrix(A_coords, B_coords)

    AB_A_ids, AB_B_ids, AA_1_ids, AA_12_idss, BB_1_ids, BB_12_idss = soft_partition(dist_AB)

    AB_d = dist_AB[AB_A_ids, AB_B_ids].sum()

    if len(AA_1_ids) != 0:
        dist_AA = norm.get_distance_matrix(A_coords, A_coords)
        AA_2_ids = []
        for AA_1_id, AA_12_ids in zip(AA_1_ids, AA_12_idss):
            AA_2_ids.append(AA_12_ids[dist_AA[AA_1_id, AA_12_ids].argmin()])
        AA_d = dist_AA[AA_1_ids, AA_2_ids].sum()
    else:
        AA_d = 0

    if len(BB_1_ids) != 0:
        dist_BB = norm.get_distance_matrix(B_coords, B_coords)
        BB_2_ids = []
        for BB_1_id, BB_12_ids in zip(BB_1_ids, BB_12_idss):
            BB_2_ids.append(BB_12_ids[dist_BB[BB_1_id, BB_12_ids].argmin()])
        BB_d = dist_BB[BB_1_ids, BB_2_ids].sum()
    else:
        BB_d = 0

    if asymmetric:
        sim = np.exp(-alpha * (AB_d + AA_d) / (len(AB_A_ids) + len(AA_1_ids)))
    else:
        sim = np.exp(-alpha * (AB_d + AA_d + BB_d) / (len(AB_A_ids) + len(AA_1_ids) + len(BB_1_ids)))

    return sim, [AB_A_ids, AB_B_ids]


def soft_jaccard_naive_embed(frgd, bkgd, asymmetric = False):
    bgd_shape_y, bgd_shape_x = bkgd.shape
    fgd_shape_y, fgd_shape_x = frgd.shape

    padding_y = int(fgd_shape_y * 0.05)
    padding_x = int(fgd_shape_x * 0.05)

    delta_xs = list(range(-padding_x, bgd_shape_x - fgd_shape_x + 1 + padding_x))
    delta_ys = list(range(-padding_y, bgd_shape_y - fgd_shape_y + 1 + padding_y))

    length = int(len(delta_xs) * len(delta_ys))
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    sj_coefs = np.zeros(length, dtype = np.float)
    itsc_coord_idss = []

    frgd_coords = np.argwhere(frgd)
    bkgd_coords = np.argwhere(bkgd)

    kk = 0
    for delta_x in delta_xs:
        for delta_y in delta_ys:
            coords[kk] = [delta_x, delta_y]
            frgd_coords_shifted = frgd_coords + [delta_y, delta_x]
            sj_coefs[kk], itsc_coord_ids = soft_jaccard_coef_internal(frgd_coords_shifted, bkgd_coords, asymmetric)
            itsc_coord_idss.append(itsc_coord_ids)
            kk += 1

    return sj_coefs, coords, itsc_coord_idss


def soft_jaccard_naive_cross(hrz, vtc, asymmetric = False):
    hrz_shape_y, hrz_shape_x = hrz.shape
    vtc_shape_y, vtc_shape_x = vtc.shape

    padding_y = int(vtc_shape_y * 0.05)
    padding_x = int(hrz_shape_x * 0.05)

    delta_xs = list(range(-padding_x, vtc_shape_x - hrz_shape_x + 1 + padding_x))
    delta_ys = list(range(-padding_y, hrz_shape_y - vtc_shape_y + 1 + padding_y))

    length = int(len(delta_xs) * len(delta_ys))
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    sj_coefs = np.zeros(length, dtype = np.float)
    itsc_coord_idss = []

    hrz_coords = np.argwhere(hrz)
    vtc_coords = np.argwhere(vtc)

    kk = 0
    for delta_x in delta_xs:
        hrz_coords_shifted = hrz_coords + [0, delta_x]
        for delta_y in delta_ys:
            vtc_coords_shifted = vtc_coords + [delta_y, 0]
            coords[kk] = [delta_x, -delta_y]
            sj_coefs[kk], itsc_coord_ids = soft_jaccard_coef_internal(hrz_coords_shifted, vtc_coords_shifted, asymmetric)
            itsc_coord_idss.append(itsc_coord_ids)
            kk += 1

    return sj_coefs, coords, itsc_coord_idss


def soft_jaccard_naive(A, B, asymmetric = False):
    A_shape_y, A_shape_x = A.shape
    B_shape_y, B_shape_x = B.shape

    if A_shape_x <= B_shape_x and A_shape_y <= B_shape_y:
        sj_coefs, A_to_B_coords, AB_itsc_coord_idss = soft_jaccard_naive_embed(A, B, asymmetric)
    elif A_shape_x > B_shape_x and A_shape_y > B_shape_y:
        sj_coefs, B_to_A_coords, BA_itsc_coord_idss = soft_jaccard_naive_embed(B, A, asymmetric)
        A_to_B_coords = -B_to_A_coords
        AB_itsc_coord_idss = np.array(BA_itsc_coord_idss)[:, :: -1].tolist()
    elif A_shape_x <= B_shape_x and A_shape_y > B_shape_y:
        sj_coefs, A_to_B_coords, AB_itsc_coord_idss = soft_jaccard_naive_cross(A, B, asymmetric)
    else:
        sj_coefs, B_to_A_coords, BA_itsc_coord_idss = soft_jaccard_naive_cross(B, A, asymmetric)
        A_to_B_coords = -B_to_A_coords
        AB_itsc_coord_idss = np.array(BA_itsc_coord_idss)[:, :: -1].tolist()

    sj_coef = np.max(sj_coefs)
    coef_argmax = np.where(sj_coefs == sj_coef)[0]

    if 1 == len(coef_argmax):
        ii = coef_argmax[0]
        A_to_B_x, A_to_B_y = A_to_B_coords[ii]
        AB_itsc_coord_ids = AB_itsc_coord_idss[ii]
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
        AB_itsc_coord_ids = AB_itsc_coord_idss[closest_center_ii]

    # diff_A = copy.copy(A)
    diff_B = copy.copy(B)
    # A_coords = np.argwhere(A)
    B_coords = np.argwhere(B)
    # diff_A[tuple(A_coords[AB_itsc_coord_ids[0]].transpose())] = False
    diff_B[tuple(B_coords[AB_itsc_coord_ids[1]].transpose())] = False

    return sj_coef, int(A_to_B_x), int(A_to_B_y), diff_B


def soft_jaccard(A, B, asymmetric = False):
    A_tr, A_tr_to_A_x, A_tr_to_A_y = utils.trim_binary_image(A, coord = True)
    B_tr, B_tr_to_B_x, B_tr_to_B_y = utils.trim_binary_image(B, coord = True)

    sj, A_tr_to_B_tr_x, A_tr_to_B_tr_y, diff_B_tr = soft_jaccard_naive(A_tr, B_tr, asymmetric)

    if asymmetric:
        diff = utils.erase_noise_point(diff_B_tr, 4)
        diff_to_A_x = -A_tr_to_B_tr_x + A_tr_to_A_x
        diff_to_A_y = -A_tr_to_B_tr_y + A_tr_to_A_y
        diff_to_B_x = B_tr_to_B_x
        diff_to_B_y = B_tr_to_B_y

        return sj, int(diff_to_A_x), int(diff_to_A_y), int(diff_to_B_x), int(diff_to_B_y), diff
    else:
        A_to_B_x = -A_tr_to_A_x + A_tr_to_B_tr_x + B_tr_to_B_x
        A_to_B_y = -A_tr_to_A_y + A_tr_to_B_tr_y + B_tr_to_B_y
        return sj, int(A_to_B_x), int(A_to_B_y)
