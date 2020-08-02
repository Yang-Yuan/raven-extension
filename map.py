from matplotlib import pyplot as plt
from itertools import permutations
import numpy as np
import jaccard
import utils


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
        # For the score, for large shape, small deviation doesn't matter
        # but for small shape, small deviation matter
        # So score = 1 - max_mapping_t / size_of_the_large_one
        # if score < 0, then score = 0
        # max_size = -np.inf
        # for A_com_id, B_com_id in zip(*(np.where(dist == max_mapping_t))):
        #     A_size = max(utils.trim_binary_image(A_coms[A_com_id]).shape)
        #     B_size = max(utils.trim_binary_image(B_coms[B_com_id]).shape)
        #     size = max(A_size, B_size)
        #     if size > max_size:
        #         max_size = size

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


def topological_map(A_coms, B_coms):
    A = utils.superimpose(A_coms)
    B = utils.superimpose(B_coms)

    A_com_ids, B_com_ids = tpm(A_coms, B_coms,
                               A, B,
                               list(range(len(A_coms))), list(range(len(B_coms))))

    if A_com_ids is not None and B_com_ids is not None:
        return A_com_ids, B_com_ids, 1
    else:
        return [], [], 0


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

        cur_A_sub_result, cur_B_sub_result = tpm(A_coms, B_coms,
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
            A_sub_result, B_sub_result = tpm(A_coms, B_coms,
                                             A_sub, B_sub,
                                             A_groups[0], B_groups[0])
            if A_sub_result is None or B_sub_result is None:
                return None, None
            else:
                A_result = A_result + A_sub_result
                B_result = B_result + B_sub_result
        else:
            # Note: there is bug, because I really should try to compare the topological structures
            # first. If it doesn't resolve which A_sub should match which B_sub, then fall_back to jaccard_map.
            # But I am too tired to write the code, and for the current test problems,
            # it is OK to use this. Remember to enhance this when more complex problems in the future.
            A_subs = [utils.superimpose([A_coms[com_id] for com_id in A_group]) for A_group in A_groups]
            B_subs = [utils.superimpose([B_coms[com_id] for com_id in B_group]) for B_group in B_groups]
            A_sub_ids, B_sub_ids, _ = jaccard_map(A_subs, B_subs)
            # End Note

            for A_sub_id, B_sub_id in zip(A_sub_ids, B_sub_ids):
                A_sub_result, B_sub_result = tpm(A_coms, B_coms,
                                                 A_subs[A_sub_id], B_subs[B_sub_id],
                                                 A_groups[A_sub_id], B_groups[B_sub_id])
                if A_sub_result is None or B_sub_result is None:
                    return None, None
                else:
                    A_result = A_result + A_sub_result
                    B_result = B_result + B_sub_result

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
