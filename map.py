from matplotlib import pyplot as plt
from itertools import permutations
import numpy as np
import jaccard
import soft_jaccard
import utils


def size_first_injective_mapping(PR, order = None):
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
        for B_id in range(B_len):
            if A_id in B_to_A_argmin[B_id] and B_id in A_to_B_argmin[A_id]:
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


def significant_level_first_injective_mapping(PR, order):
    """
    This function tries to emulate how a human forms an injective mapping
    when given a quantitative description of how well/bad each element in the domain
    is related to each element in the codomain.
    The way human do this mapping is different from the formal/mathematical methods in
    planning or optimization problems, in particular when some "conflicts/ambiguity" exist
    and the decisions need to be made to resolve the "conflicts/ambiguity", and you don't
    how the decision is going to influence the final result until the final step is reached.
    Formal methods, pursuing the optimal solution with various techniques,
    deal with this situation much more comprehensively than humans (if humans don't
    go through these formal methods themselves), while humans mostly only identify the
    most perceptually significant correspondences, ignoring the elements they are not sure about.
    In the intelligence tests for humans, we tend to favor the human mapping method for two reasons:
    (1) it is designed by/for human.
    (2) it is computationally inexpensive.
    (3) if a correspondence not perceptually significant, how do you guarantee it can surpport
        the following reasoning steps?
    Therefore, we decide to implement a method that emulates human's mapping by considering only
    correspondences that are more significant than others, and at the same they can form an
    injective mapping that includes as many such correspondences as possible.
    :param PR: a matrix denoting significant level of the perceptual relations between each element
                in the domain and each element in codomain.
    :param order: a function taking as input two values in PR to decide whether one is more
                    significant than the other. Return True for more or equally significant, False for not.
    :return: a mapping
    """
    levels = np.unique(PR)
    PR_flat = PR.flatten()

    max_mapping = None
    max_mapping_level = None
    max_mapping_size = -np.inf
    for level in levels:
        mapping = np.array(list(map(order, PR_flat, np.full_like(PR_flat, level)))).reshape(PR.shape)

        if utils.is_injective(mapping):
            mapping_size = mapping.sum()
            if mapping_size >= max_mapping_size:
                max_mapping_size = mapping_size
                max_mapping = mapping
                max_mapping_level = level

    if max_mapping is not None:
        mapping_ids = np.where(max_mapping)
        return mapping_ids[0].tolist(), mapping_ids[1].tolist(), max_mapping_level
    else:
        return [], [], 0


def delta_location_map(A_coms, B_coms, C_coms, D_coms,
                       AB_A_com_ids, AB_B_com_ids,
                       CD_C_com_ids, CD_D_com_ids):
    if len(AB_A_com_ids) != len(CD_C_com_ids):
        return None, None, None, None, 0

    AB_loc_diff = utils.location_diff(A_coms, B_coms, AB_A_com_ids, AB_B_com_ids)
    CD_loc_diff = utils.location_diff(C_coms, D_coms, CD_C_com_ids, CD_D_com_ids)

    dist = np.array([[np.linalg.norm(AB_lcd - CD_lcd) for CD_lcd in CD_loc_diff] for AB_lcd in AB_loc_diff])

    AB_loc_diff_ids, CD_loc_diff_ids, level = significant_level_first_injective_mapping(dist, lambda a, b: a <= b)

    if AB_loc_diff_ids is not None and CD_loc_diff_ids is not None and len(AB_loc_diff_ids) == len(AB_A_com_ids):
        AC_A_com_ids = [AB_A_com_ids[AB_id] for AB_id in AB_loc_diff_ids]
        BD_B_com_ids = [AB_B_com_ids[AB_id] for AB_id in AB_loc_diff_ids]
        AC_C_com_ids = [CD_C_com_ids[CD_id] for CD_id in CD_loc_diff_ids]
        BD_D_com_ids = [CD_D_com_ids[CD_id] for CD_id in CD_loc_diff_ids]
        return AC_A_com_ids, AC_C_com_ids, BD_B_com_ids, BD_D_com_ids, 1 - level / max(A_coms[0].shape)
    else:
        return [], [], [], [], 0


def delta_jaccard_map(A_coms, B_coms, C_coms, D_coms,
                      AB_A_com_ids, AB_B_com_ids,
                      CD_C_com_ids, CD_D_com_ids):

    if len(AB_A_com_ids) != len(CD_C_com_ids):
        return None, None, None, None, 0

    AB_jaccard_diff = [jaccard.jaccard_coef(A_coms[A_com_id], B_coms[B_com_id])[0] for A_com_id, B_com_id in
                       zip(AB_A_com_ids, AB_B_com_ids)]
    CD_jaccard_diff = [jaccard.jaccard_coef(C_coms[C_com_id], D_coms[D_com_id])[0] for C_com_id, D_com_id in
                       zip(CD_C_com_ids, CD_D_com_ids)]

    dist = np.array([[abs(AB_jcd - CD_jcd) for CD_jcd in CD_jaccard_diff] for AB_jcd in AB_jaccard_diff])

    AB_jaccard_diff_ids, CD_jaccard_diff_ids, level = significant_level_first_injective_mapping(dist, lambda a, b: a <= b)

    if AB_jaccard_diff_ids is not None and CD_jaccard_diff_ids is not None and len(AB_jaccard_diff_ids) == len(AB_A_com_ids):
        AC_A_com_ids = [AB_A_com_ids[AB_id] for AB_id in AB_jaccard_diff_ids]
        BD_B_com_ids = [AB_B_com_ids[AB_id] for AB_id in AB_jaccard_diff_ids]
        AC_C_com_ids = [CD_C_com_ids[CD_id] for CD_id in CD_jaccard_diff_ids]
        BD_D_com_ids = [CD_D_com_ids[CD_id] for CD_id in CD_jaccard_diff_ids]
        return AC_A_com_ids, AC_C_com_ids, BD_B_com_ids, BD_D_com_ids, 1 - level
    else:
        return [], [], [], [], 0


def placeholder_map(A_coms, B_coms):
    """
    this mapping serves as a placeholder mapping derived from an
    original mapping that lies in the other direction.
    This mapping should be consistent with the original mapping in
    that the two elements mapped by the original mapping have the same
    computational role in the this new mapping, i.e. if there exists
    a link between two elements in this mapping (or its inverse mapping)
    , then their images in the original mapping should also have a link
    in this new mapping (or its inverse mapping).
    Since an arbitrary bijective mapping is used for this placeholder,
    how well this mapping holds is actually indicated by the score of its
    original mapping at different stages (digest and predict).
    Why bijective?
    Because bijective mapping is more rigorous than mappings of other kinds
    in that it involves all elements.
    If you use only injective or surjective mapping, you have to decide which
    ones are mapped and which one are not.
    Why not choose a single random map?
    Because this placeholder mapping should always be consistent with its
    original mapping, i.e. all the possible bijective mappings should be checked
    when predicting/make decision about each option.
    :param A_coms:
    :param B_coms:
    :return:
    """

    if len(A_coms) != len(B_coms):
        return None, None, 0
    else:
        return None, None, 1


def derive_isomorphic_mappings(A, B, C, D,
                               m1_AB_A_com_ids, m1_AB_B_com_ids,
                               m2_AB_A_com_ids, m2_AB_B_com_ids,
                               m1_CD_C_com_ids, m1_CD_D_com_ids,
                               m2_CD_C_com_ids, m2_CD_D_com_ids):
    if len(A) == len(C) and len(B) == len(D):

        isomorphic_mappings = []
        for p in permutations(C):

            AC_A_com_ids = A
            AC_C_com_ids = list(p)

            m1_BD_B_com_ids, m1_BD_D_com_ids = translate_mapping(AC_A_com_ids, AC_C_com_ids,
                                                                 m1_AB_A_com_ids, m1_AB_B_com_ids,
                                                                 m1_CD_C_com_ids, m1_CD_D_com_ids)

            m2_BD_B_com_ids, m2_BD_D_com_ids = translate_mapping(AC_A_com_ids, AC_C_com_ids,
                                                                 m2_AB_A_com_ids, m2_AB_B_com_ids,
                                                                 m2_CD_C_com_ids, m2_CD_D_com_ids)

            if same_mappings(B, D, m1_BD_B_com_ids, m1_BD_D_com_ids, m2_BD_B_com_ids, m2_BD_D_com_ids):
                isomorphic_mappings.append([m1_BD_B_com_ids, m1_BD_D_com_ids])

        return isomorphic_mappings

    else:
        return None


def same_mappings(A, B, m1_AB_A_com_ids, m1_AB_B_com_ids, m2_AB_A_com_ids, m2_AB_B_com_ids):
    for a in A:
        b1 = do_map(a, [m1_AB_A_com_ids, m1_AB_B_com_ids])
        b2 = do_map(a, [m2_AB_A_com_ids, m2_AB_B_com_ids])
        if b1 != b2:
            return False

    for b in B:
        a1 = do_map(b, [m1_AB_B_com_ids, m1_AB_A_com_ids])
        a2 = do_map(b, [m2_AB_B_com_ids, m2_AB_A_com_ids])
        if a1 != a2:
            return False

    return True


def translate_mapping(AC_A_com_ids, AC_C_com_ids,
                      AB_A_com_ids, AB_B_com_ids,
                      CD_C_com_ids, CD_D_com_ids):
    """
    derive the mapping from B to D
    :return:
    """
    if AC_A_com_ids is None or AC_C_com_ids is None:
        return None, None

    BD_B_com_ids = []
    BD_D_com_ids = []
    for A_com_id, C_com_id in zip(AC_A_com_ids, AC_C_com_ids):
        B_com_id = do_map(A_com_id, [AB_A_com_ids, AB_B_com_ids])
        D_com_id = do_map(C_com_id, [CD_C_com_ids, CD_D_com_ids])
        if B_com_id is not None and D_com_id is not None:
            BD_B_com_ids.append(B_com_id)
            BD_D_com_ids.append(D_com_id)

    return BD_B_com_ids, BD_D_com_ids


def location_map(A_coms, B_coms):
    A_centers = np.array([utils.center_of_mass(com) for com in A_coms])
    B_centers = np.array([utils.center_of_mass(com) for com in B_coms])

    dist = np.array([[np.linalg.norm(A_c - B_c) for B_c in B_centers] for A_c in A_centers])

    thresholds = np.unique(dist)[::-1]

    max_mapping_size = -np.inf
    max_mapping = None
    max_mapping_t = None
    for t in thresholds:
        mapping = dist <= t

        if utils.is_injective(mapping):
            mapping_size = mapping.sum()

            if mapping_size >= max_mapping_size:
                max_mapping_size = mapping_size
                max_mapping = mapping
                max_mapping_t = t

    if max_mapping is not None:
        mapping_ids = np.where(max_mapping)
        return mapping_ids[0].tolist(), mapping_ids[1].tolist(), 1 - max_mapping_t / max(A_coms[0].shape)
    else:
        return [], [], 0


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

    thresholds = np.unique(all_jc)

    max_mapping = None
    max_mapping_t = None
    max_mapping_size = -np.inf
    for t in thresholds:
        mapping = all_jc >= t

        if utils.is_injective(mapping):
            mapping_size = mapping.sum()
            if mapping_size >= max_mapping_size:
                max_mapping_size = mapping_size
                max_mapping = mapping
                max_mapping_t = t

    if max_mapping is not None:
        mapping_ids = np.where(max_mapping)
        return mapping_ids[0].tolist(), mapping_ids[1].tolist(), max_mapping_t
    else:
        return [], [], 0


def soft_jaccard_map(A_coms, B_coms):
    """
    find a injective (not necessarily surjective, bijective or well-defined) mapping
    between A_coms and B_coms such that the sum of jaccard indices of each mapped pair
    of components is maximized
    :param A_coms:
    :param B_coms:
    :return: indices of mapped pair, and minimal jaccard index
    """
    all_jc = np.array([[soft_jaccard.soft_jaccard(A_com, B_com)[0] for B_com in B_coms] for A_com in A_coms])

    A_ids, B_ids, level = significant_level_first_injective_mapping(all_jc, lambda a, b: a >= b)

    return A_ids, B_ids, level


def topological_map(A_coms, B_coms):
    A = utils.superimpose(A_coms)
    B = utils.superimpose(B_coms)

    A_com_ids, B_com_ids = tpm(A_coms, B_coms, A, B,
                               list(range(len(A_coms))), list(range(len(B_coms))))

    if 0 != len(A_com_ids) and 0 != len(B_com_ids):
        return A_com_ids, B_com_ids, 1
    else:
        return [], [], 0


def tpm(A_coms, B_coms, cur_A, cur_B, cur_A_com_ids, cur_B_com_ids):
    if len(cur_A_com_ids) != len(cur_B_com_ids):
        return [], []

    if 1 == len(cur_A_com_ids) and 1 == len(cur_B_com_ids):
        return cur_A_com_ids, cur_B_com_ids

    cur_A_filled = utils.fill_holes(cur_A)
    cur_B_filled = utils.fill_holes(cur_B)

    cur_A_filled_coms, _, _ = utils.decompose(cur_A_filled, 8, trim = False)
    cur_B_filled_coms, _, _ = utils.decompose(cur_B_filled, 8, trim = False)

    if len(cur_A_filled_coms) != len(cur_B_filled_coms):
        return [], []

    if len(cur_A_filled_coms) == len(cur_A_com_ids):
        # TODO inside-outside topo mapping failed here.
        # TODO fallback mapping method can be applied here, but for the current problems, it is unnecessary.
        return [], []

    A_com_groups = [[com_id for com_id in cur_A_com_ids
                     if (np.logical_and(A_coms[com_id], A_filled_com) == A_coms[com_id]).all()]
                    for A_filled_com in cur_A_filled_coms]

    B_com_groups = [[com_id for com_id in cur_B_com_ids
                     if (np.logical_and(B_coms[com_id], B_filled_com) == B_coms[com_id]).all()]
                    for B_filled_com in cur_B_filled_coms]

    if 1 == len(A_com_groups) and 1 == len(B_com_groups):
        return tpm_go_deeper(A_coms, B_coms, cur_A_com_ids, cur_A_filled, cur_B_com_ids, cur_B_filled)
    else:
        return tpm_go_wider(A_coms, B_coms, A_com_groups, B_com_groups)


def tpm_go_wider(A_coms, B_coms, A_com_groups, B_com_groups):
    A_com_group_sizes = np.array([len(g) for g in A_com_groups])
    B_com_group_sizes = np.array([len(g) for g in B_com_groups])

    if not (np.sort(A_com_group_sizes) == np.sort(B_com_group_sizes)).all():
        return [], []

    A_result = []
    B_result = []

    group_sizes = np.unique(A_com_group_sizes)
    for size in group_sizes:
        A_size_group_ids = np.where(A_com_group_sizes == size)[0]
        B_size_group_ids = np.where(B_com_group_sizes == size)[0]
        A_size_groups = [A_com_groups[ii] for ii in A_size_group_ids]
        B_size_groups = [B_com_groups[ii] for ii in B_size_group_ids]

        A_size_result = []
        B_size_result = []
        for p in permutations(range(len(B_size_groups))):
            A_size_group_ids = list(range(len(A_size_groups)))
            B_size_group_ids = list(p)

            A_size_p_result = []
            B_size_p_result = []
            for A_group_id, B_group_id in zip(A_size_group_ids, B_size_group_ids):
                A_sub = utils.superimpose([A_coms[com_id] for com_id in A_size_groups[A_group_id]])
                B_sub = utils.superimpose([B_coms[com_id] for com_id in B_size_groups[B_group_id]])

                A_sub_result, B_sub_result = tpm(A_coms, B_coms, A_sub, B_sub,
                                                 A_size_groups[A_group_id], B_size_groups[B_group_id])

                if 0 == len(A_sub_result) or 0 == len(B_sub_result):
                    A_size_p_result = []
                    B_size_p_result = []
                    break
                else:
                    A_size_p_result = A_size_p_result + A_sub_result
                    B_size_p_result = B_size_p_result + B_sub_result

            if 0 != len(A_size_p_result) and 0 != len(B_size_p_result):
                A_size_result = A_size_p_result
                B_size_result = B_size_p_result
                break

        if 0 == len(A_size_result) or 0 == len(B_size_result):
            return [], []
        else:
            A_result = A_result + A_size_result
            B_result = B_result + B_size_result

    return A_result, B_result


def tpm_go_deeper(A_coms, B_coms, cur_A_com_ids, cur_A_filled, cur_B_com_ids, cur_B_filled):
    A_result = []
    B_result = []

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
    cur_A_sub_result, cur_B_sub_result = tpm(A_coms, B_coms,
                                             cur_A_sub, cur_B_sub,
                                             cur_A_com_ids, cur_B_com_ids)
    if 0 == len(cur_A_sub_result) or 0 == len(cur_A_sub_result):
        return [], []
    else:
        A_result = A_result + cur_A_sub_result
        B_result = B_result + cur_B_sub_result

    return A_result, B_result


def are_consistent(A, B, C, D,
                   AB_A_ids, AB_B_ids,
                   CD_C_ids, CD_D_ids,
                   AC_A_ids, AC_C_ids,
                   BD_B_ids, BD_D_ids):
    if AB_A_ids is None \
            or AB_B_ids is None \
            or CD_C_ids is None \
            or CD_D_ids is None \
            or AC_A_ids is None \
            or AC_C_ids is None \
            or BD_B_ids is None \
            or BD_D_ids is None:
        return False

    for a in A:
        d_b = do_map(a, [AB_A_ids, AB_B_ids], [BD_B_ids, BD_D_ids])
        d_c = do_map(a, [AC_A_ids, AC_C_ids], [CD_C_ids, CD_D_ids])
        if d_b != d_c:
            return False

    for b in B:
        c_a = do_map(b, [AB_B_ids, AB_A_ids], [AC_A_ids, AC_C_ids])
        c_d = do_map(b, [BD_B_ids, BD_D_ids], [CD_D_ids, CD_C_ids])
        if c_a != c_d:
            return False

    for c in C:
        b_a = do_map(c, [AC_C_ids, AC_A_ids], [AB_A_ids, AB_B_ids])
        b_d = do_map(c, [CD_C_ids, CD_D_ids], [BD_D_ids, BD_B_ids])
        if b_a != b_d:
            return False

    for d in D:
        a_b = do_map(d, [BD_D_ids, BD_B_ids], [AB_B_ids, AB_A_ids])
        a_c = do_map(d, [CD_D_ids, CD_C_ids], [AC_C_ids, AC_A_ids])
        if a_b != a_c:
            return False

    return True


def do_map(x, *mappings):
    for m in mappings:
        try:
            idx = m[0].index(x)
            x = m[1][idx]
        except ValueError:
            return None
        except AttributeError:
            return None
        except IndexError:
            return None

    return x
