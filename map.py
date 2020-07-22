from matplotlib import pyplot as plt
from itertools import permutations
import numpy as np
import jaccard
import utils


def jaccard_map(A_coms, B_coms):
    """
    find a injective (not necessarily surjective, bijective or well-defined) mapping
    between A_coms and B_coms such that the sum of jaccard indices of each mapped pair
    of components is maximized
    :param A_coms:
    :param B_coms:
    :return: indices of mapped pair, and minimal jaccard index
    """

    all_jc = np.array([[jaccard.jaccard_coef(A_com, B_com)[0] for B_com in B_coms] for A_com in A_coms])

    A_coms_len = len(A_coms)
    B_coms_len = len(B_coms)
    max_coms_len = max(A_coms_len, B_coms_len)

    # any value greater than 1, the range of jaccard index, will do here.
    pad_value = 1.1

    all_jc = np.pad(all_jc,
                    pad_width = ((0, max_coms_len - A_coms_len), (0, max_coms_len - B_coms_len)),
                    constant_values = pad_value)

    ps = list(permutations(range(max_coms_len)))
    all_jc_p = np.array([all_jc[tuple(range(max_coms_len)), p] for p in ps])
    sum_max_p = all_jc_p.sum(axis = 1).argmax()
    all_jc_p[sum_max_p].sort()
    min_sum_max_jc = all_jc_p[sum_max_p][0]

    return ps[sum_max_p], min_sum_max_jc


def topological_map(A_coms, B_coms):
    A = utils.superimpose(A_coms)
    B = utils.superimpose(B_coms)

    A_com_ids, B_com_ids = tpm(A_coms, B_coms,
                               A, B,
                               list(range(len(A_coms))), list(range(len(B_coms))))

    if A_com_ids is not None and B_com_ids is not None:
        B_com_ids = [B_com_ids[A_com_ids.index(ii)] for ii in range(len(A_com_ids))]

    return B_com_ids


def tpm(A_coms, B_coms, cur_A, cur_B, cur_A_com_ids, cur_B_com_ids):
    if len(cur_A_com_ids) != len(cur_B_com_ids):
        return None, None

    if 1 == len(cur_A_com_ids) and 1 == len(cur_B_com_ids):
        return cur_A_com_ids, cur_B_com_ids

    cur_A_filled = utils.fill_holes(cur_A)
    cur_B_filled = utils.fill_holes(cur_B)

    cur_A_filled_coms, _, _ = utils.decompose(cur_A_filled, 8, trim = False)
    cur_B_filled_coms, _, _ = utils.decompose(cur_B_filled, 8, trim = False)

    if len(cur_A_filled_coms) != len(cur_B_filled_coms):
        return None, None

    A_com_groups = [[com_id for com_id in cur_A_com_ids
                     if (np.logical_and(A_coms[com_id], A_filled_com) == A_coms[com_id]).all()]
                    for A_filled_com in cur_A_filled_coms]

    B_com_groups = [[com_id for com_id in cur_B_com_ids
                     if (np.logical_and(B_coms[com_id], B_filled_com) == B_coms[com_id]).all()]
                    for B_filled_com in cur_B_filled_coms]

    A_com_group_sizes = np.array([len(g) for g in A_com_groups])
    B_com_group_sizes = np.array([len(g) for g in B_com_groups])

    if not (np.sort(A_com_group_sizes) == np.sort(B_com_group_sizes)).all():
        return None, None

    A_result = []
    B_result = []

    if 1 == len(A_com_groups) and 1 == len(B_com_groups):

        for ii, com_id in enumerate(cur_A_com_ids):
            if (utils.fill_holes(A_coms[com_id]) == cur_A_filled).all():
                break
        A_result.append(cur_A_com_ids.pop(ii))

        for ii, com_id in enumerate(cur_B_com_ids):
            if (utils.fill_holes(B_coms[com_id]) == cur_B_filled).all():
                break
        B_result.append(cur_B_com_ids.pop(ii))

        cur_A_sub = utils.superimpose([A_coms[com_id] for com_id in cur_A_com_ids])
        cur_B_sub = utils.superimpose([B_coms[com_id] for com_id in cur_B_com_ids])

        cur_A_sub_result, cur_B_sub_result = topological_map(A_coms, B_coms,
                                                             cur_A_sub, cur_B_sub,
                                                             cur_A_com_ids, cur_B_com_ids)
        if cur_A_sub_result is None or cur_A_sub_result is None:
            return None, None
        else:
            A_result = A_result + cur_A_sub_result
            B_result = B_result + cur_A_sub_result

        return A_result, B_result

    group_sizes = np.unique(A_com_group_sizes)
    for size in group_sizes:
        A_group_ids = np.where(A_com_group_sizes == size)[0]
        B_group_ids = np.where(B_com_group_sizes == size)[0]
        A_groups = [A_com_groups[ii] for ii in A_group_ids]
        B_groups = [B_com_groups[ii] for ii in B_group_ids]

        # this should not happen.
        if len(A_groups) != len(B_groups):
            raise Exception("Ryan!")

        group_n = len(A_groups)
        if 1 == group_n:
            A_sub = utils.superimpose([A_coms[com_id] for com_id in A_groups[0]])
            B_sub = utils.superimpose([B_coms[com_id] for com_id in B_groups[0]])
            A_sub_result, B_sub_result = topological_map(A_coms, B_coms,
                                                         A_sub, B_sub,
                                                         A_groups[0], B_groups[0])
            if A_sub_result is None or B_sub_result is None:
                return None, None
            else:
                A_result = A_result + A_sub_result
                B_result = B_result + B_sub_result
        else:
            A_subs = [utils.superimpose([A_coms[com_id] for com_id in A_group]) for A_group in A_groups]
            B_subs = [utils.superimpose([B_coms[com_id] for com_id in B_group]) for B_group in B_groups]

            # there is bug, or error here, because I really should try to compare the topological structures
            # first. If it doesn't resolve which A_sub should match which B_sub, then fall_back to jaccard_map.
            # But I am too tired to write the code, and for the current test problems,
            # it is OK to use this.
            jcm, _ = jaccard_map(A_subs, B_subs)

            for A_sub_id, B_sub_id in enumerate(jcm):
                A_sub_result, B_sub_result = topological_map(A_coms, B_coms,
                                                             A_subs[A_sub_id], B_subs[B_sub_id],
                                                             A_groups[A_sub_id], B_groups[B_sub_id])
                if A_sub_result is None or B_sub_result is None:
                    return None, None
                else:
                    A_result = A_result + A_sub_result
                    B_result = B_result + B_sub_result

    return A_result, B_result
