from matplotlib import pyplot as plt
from skimage.transform import rotate
from skimage.transform import rescale as rs
from skimage.transform import resize
from sys import modules
import numpy as np
import soft_jaccard
import jaccard
import asymmetric_jaccard
import map
import utils

THIS = modules[__name__]

unary_transformations = [
    {"name": "identity", "value": [{"name": None}], "type": "unary", "group": 0},
    {"name": "rot_45", "value": [{"name": "rot_binary", "args": {"angle": 45}}], "type": "unary", "groups": 0},
    {"name": "rot_135", "value": [{"name": "rot_binary", "args": {"angle": 135}}], "type": "unary", "groups": 0},
    {"name": "rot_225", "value": [{"name": "rot_binary", "args": {"angle": 225}}], "type": "unary", "groups": 0},
    {"name": "rot_315", "value": [{"name": "rot_binary", "args": {"angle": 315}}], "type": "unary", "groups": 0},
    {"name": "rot_90", "value": [{"name": "rot_binary", "args": {"angle": 90}}], "type": "unary", "group": 0},
    {"name": "rot_180", "value": [{"name": "rot_binary", "args": {"angle": 180}}], "type": "unary", "group": 0},
    {"name": "rot_270", "value": [{"name": "rot_binary", "args": {"angle": 270}}], "type": "unary", "group": 0},
    {"name": "mirror", "value": [{"name": "mirror_left_right"}], "type": "unary", "group": 0},
    {"name": "mirror_rot_90", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 90}}], "type": "unary", "group": 0},
    {"name": "mirror_rot_180", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 180}}], "type": "unary", "group": 0},
    {"name": "mirror_rot_270", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 270}}], "type": "unary", "group": 0},
    {"name": "rescale", "value": [{"name": "rescale", "args": {"x_factor": 1.3, "y_factor": 1.4}}], "type": "unary", "group": 0},
    # {"name": "upscale_to", "value": [{"name": "upscale_to"}], "type": "unary"},
    {"name": "add_diff", "value": [{"name": "add_diff"}], "type": "unary", "group": 1},
    {"name": "subtract_diff", "value": [{"name": "subtract_diff"}], "type": "unary", "group": 1},
    # {"name": "xor_diff", "value": [{"name": "xor_diff"}], "type": "unary", "group": 1},
    {"name": "duplicate", "value": [{"name": "duplicate"}], "type": "unary", "group": 2},
    {"name": "rearrange", "value": [{"name": "rearrange"}], "type": "unary", "group": 2},

    # {"name": "WWW", "value": [{"name": "WWW"}], "type": "unary", "group": 2},
    {"name": "duplicate_new", "value": [{"name": "duplicate_new"}], "type": "unary", "group": 2},
    {"name": "shape_texture_transfer", "value": [{"name": "shape_texture_transfer"}], "type": "unary", "group": 2},
    {"name": "shape_topo_mapping", "value": [{"name": "shape_topo_mapping"}], "type": "unary", "group": 2},
    {"name": "shape_loc_isomorphism", "value": [{"name": "shape_loc_isomorphism"}], "type": "unary", "group": 2},
    {"name": "shape_delta_loc_isomorphism", "value": [{"name": "shape_delta_loc_isomorphism"}], "type": "unary", "group": 2},
    {"name": "topo_delta_shape_isomorphism", "value": [{"name": "topo_delta_shape_isomorphism"}], "type": "unary", "group": 2}
]

binary_transformations = [
    {"name": "unite", "value": [{"name": "unite"}], "type": "binary", "align": "unite_align", "group": 3},
    {"name": "intersect", "value": [{"name": "intersect"}], "type": "binary", "group": 3},
    # {"name": "subtract", "value": [{"name": "subtract"}], "type": "binary"},
    # {"name": "backward_subtract", "value": [{"name": "backward_subtract"}], "type": "binary"},
    {"name": "xor", "value": [{"name": "xor"}], "type": "binary", "group": 3},
    {"name": "shadow_mask_unite", "value": [{"name": "shadow_mask_unite"}], "type": "binary", "group": 3},
    {"name": "inv_unite", "value": [{"name": "inv_unite"}], "type": "binary", "group": 3},
    {"name": "preserving_subtract_diff", "value": [{"name": "preserving_subtract_diff"}], "type": "binary", "group": 3}
]

all_trans = unary_transformations + binary_transformations


def get_tran(tran_name):
    for tran in all_trans:
        if tran_name == tran.get("name"):
            return tran


def duplicate(img, copies_to_img_x, copies_to_img_y):

    current = img.copy()
    current_to_img_x = 0
    current_to_img_y = 0
    for copy_to_img_x, copy_to_img_y in zip(copies_to_img_x, copies_to_img_y):
        copy_to_current_x = copy_to_img_x - current_to_img_x
        copy_to_current_y = copy_to_img_y - current_to_img_y
        copy_aligned, current_aligned, aligned_to_current_x, aligned_to_current_y = utils.align(
            img, current, copy_to_current_x, copy_to_current_y)
        current = np.logical_or(copy_aligned, current_aligned)
        current_to_img_x = aligned_to_current_x + current_to_img_x
        current_to_img_y = aligned_to_current_y + current_to_img_y

    return utils.trim_binary_image(current)


def evaluate_rearrange(u1, u2):

    u1_coms, u1_coms_x, u1_coms_y = utils.decompose(u1, 8)
    u2_coms, u2_coms_x, u2_coms_y = utils.decompose(u2, 8)

    if len(u1_coms) != len(u2_coms):
        return 0, [], [], [], []

    max_scores = []
    max_ids = []
    chosen = [False] * len(u2_coms)
    for u1_com in u1_coms:

        max_score = -1
        max_id = -1
        for id, u2_com in enumerate(u2_coms):
            if not chosen[id]:
                score, _, _ = jaccard.jaccard_coef(u1_com, u2_com)
                if score > max_score:
                    max_score = score
                    max_id = id

        max_scores.append(max_score)
        max_ids.append(max_id)
        chosen[max_id] = True

    min_max_score = min(max_scores)
    if min_max_score < 0.6:
        return 0, [], [], [], []
    else:
        u2_coms_x = [u2_coms_x[id] for id in max_ids]
        u2_coms_y = [u2_coms_y[id] for id in max_ids]
        return min_max_score, u1_coms_x, u1_coms_y, u2_coms_x, u2_coms_y


def rearrange(img, old_xs, old_ys, new_xs, new_ys):

    coms, coms_x, coms_y = utils.decompose(img, 8)

    if len(coms) != len(old_xs):
        return np.full_like(img, fill_value = False)

    new_coms = []
    for com, com_x, com_y in zip(coms, coms_x, coms_y):

        closet_dist = np.inf
        closest_ii = -1
        for ii, old_x, old_y in zip(range(len(old_xs)), old_xs, old_ys):
            dist = abs(com_x - old_x) + abs(com_y - old_y)
            if dist < closet_dist:
                closet_dist = dist
                closest_ii = ii

        x = new_xs[closest_ii]
        y = new_ys[closest_ii]
        bg = np.full((300, 300), fill_value = False)
        bg[y : y + com.shape[0], x : x + com.shape[1]] = com  # A bug here, but I am too tired to fix it today.
        new_coms.append(bg)

    current = np.full((300, 300), fill_value = False)
    for new_com in new_coms:
        current = np.logical_or(current, new_com)

    return utils.trim_binary_image(current)


def rescale(img, x_factor, y_factor):

    return utils.grey_to_binary(rs(np.logical_not(img), (y_factor, x_factor), order = 0), 0.7)

    # return utils.grey_to_binary(resize(np.logical_not(img), shape, order = 0), 0.7)


def upscale_to(img, ref):
    img_shape_y, img_shape_x = img.shape
    ref_shape_y, ref_shape_x = ref.shape
    max_y = max(img_shape_y, ref_shape_y)
    max_x = max(img_shape_x, ref_shape_x)
    return utils.grey_to_binary(resize(np.logical_not(img), (max_y, max_x), order = 0), 0.7)


def subtract_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref, coords = False):
    """

    :param img:
    :param diff_to_ref_x:
    :param diff_to_ref_y:
    :param diff:
    :param ref:
    :return:
    """

    if 0 == diff.sum():
        if coords:
            return img, 0, 0
        else:
            return img

    _, img_to_ref_x, img_to_ref_y = jaccard.jaccard_coef(img, ref)
    diff_to_img_x = diff_to_ref_x - img_to_ref_x
    diff_to_img_y = diff_to_ref_y - img_to_ref_y

    diff_aligned, img_aligned, aligned_to_img_x, aligned_to_img_y = utils.align(diff, img, diff_to_img_x, diff_to_img_y)

    result, result_to_aligned_x, result_to_aligned_y = utils.trim_binary_image(
        utils.erase_noise_point(np.logical_and(img_aligned, np.logical_not(diff_aligned)), 8), coord = True)

    if coords:
        diff_to_result_x = (diff_to_img_x - aligned_to_img_x) - result_to_aligned_x
        diff_to_result_y = (diff_to_img_y - aligned_to_img_y) - result_to_aligned_y
        return result, diff_to_result_x, diff_to_result_y
    else:
        return result


def add_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref):
    """

    :param img:
    :param diff_to_ref_x:
    :param diff_to_ref_y:
    :param diff:
    :param ref:
    :return:
    """

    if 0 == diff.sum():
        return img

    iimg = img.copy()
    iimg_to_img_x = 0
    iimg_to_img_y = 0
    while True:

        _, ref_to_iimg_x, ref_to_iimg_y = jaccard.jaccard_coef(ref, iimg)
        diff_to_iimg_x = diff_to_ref_x + ref_to_iimg_x
        diff_to_iimg_y = diff_to_ref_y + ref_to_iimg_y

        score, _ = asymmetric_jaccard.asymmetric_jaccard_coef_pos_fixed(
            diff, iimg, diff_to_iimg_x, diff_to_iimg_y)

        if score > 0.9:

            ref_aligned, iimg_aligned, aligned_to_iimg_x, aligned_to_iimg_y = utils.align(
                ref, iimg, ref_to_iimg_x, ref_to_iimg_y)

            iimg_aligned = utils.erase_noise_point(np.logical_and(iimg_aligned, np.logical_not(ref_aligned)), 4)

            iimg = iimg_aligned
            iimg_to_img_x = aligned_to_iimg_x + iimg_to_img_x
            iimg_to_img_y = aligned_to_iimg_y + iimg_to_img_y

            if iimg.sum() == 0:
                return img
        else:
            break

    diff_to_img_x = diff_to_iimg_x + iimg_to_img_x
    diff_to_img_y = diff_to_iimg_y + iimg_to_img_x

    diff_aligned, img_aligned, _, _ = utils.align(diff, img, diff_to_img_x, diff_to_img_y)

    return utils.trim_binary_image(np.logical_or(img_aligned, diff_aligned))


def xor_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref):
    """

    :param img:
    :param diff_to_ref_x:
    :param diff_to_ref_y:
    :param diff:
    :param ref:
    :return:
    """

    if 0 == diff.sum():
        return img

    _, img_to_ref_x, img_to_ref_y = jaccard.jaccard_coef(img, ref)
    diff_to_img_x = diff_to_ref_x - img_to_ref_x
    diff_to_img_y = diff_to_ref_y - img_to_ref_y

    diff_aligned, img_aligned, _, _ = utils.align(diff, img, diff_to_img_x, diff_to_img_y)
    return utils.erase_noise_point(utils.trim_binary_image(np.logical_xor(img_aligned, diff_aligned)), 4)


def rot_binary(img, angle):
    """

    :param img:
    :param angle:
    :return:
    """
    # pad false
    return rot(img, angle, cval = False)


def rot_raw(img, angle):
    """

    :param img:
    :param angle:
    :return:
    """
    # pad white
    return rot(img, angle, cval = 255)


def rot(img, angle, cval = 0):
    """

    :param img:
    :param angle:
    :param cval:
    :return:
    """
    return rotate(image = img,
                  angle = angle,
                  resize = True,
                  order = 0,
                  mode = "constant",
                  cval = cval,
                  clip = True,
                  preserve_range = True).astype(img.dtype)


def mirror_left_right(img):
    return mirror(img, "left_right")


def mirror_top_bottom(img):
    return mirror(img, "top_bottom")


def mirror(img, mode):
    if mode not in {"left_right", "top_bottom"}:
        raise ValueError("mode should either 'left_right' or 'top_bottom'.")

    if mode == "left_right":
        return img[:, :: -1]
    elif mode == "top_bottom":
        return img[:: -1, :]
    else:
        pass


def apply_unary_transformation(img, tran, show_me = False):
    unary_trans = tran.get("value")
    for tran in unary_trans:

        name = tran.get("name")

        if name is None:
            continue

        foo = getattr(THIS, name)

        args = tran.get("args")
        if args is None:
            img = foo(img)
        else:
            img = foo(img, **args)

    if show_me:
        plt.imshow(img)
        plt.show()

    return img


def apply_binary_transformation(imgA, imgB, tran,
                                imgA_to_imgB_x = None, imgA_to_imgB_y = None,
                                imgC = None):
    if imgA_to_imgB_x is None or imgA_to_imgB_y is None:
        align_foo_name = tran.get("align")
        if align_foo_name is  None:
            _, imgA_to_imgB_x, imgA_to_imgB_y = jaccard.jaccard_coef(imgA, imgB)
        else:
            align_foo = getattr(THIS, align_foo_name)
            imgA_to_imgB_x, imgA_to_imgB_y = align_foo(imgA, imgB, imgC)

    imgA_aligned, imgB_aligned, aligned_to_B_x, aligned_to_B_y = utils.align(imgA, imgB, imgA_to_imgB_x, imgA_to_imgB_y)

    img_aligned = None

    binary_trans = tran.get("value")

    for tran in binary_trans:

        name = tran.get("name")

        if name is None:
            continue
        foo = getattr(THIS, name)

        args = tran.get("args")
        if args is None:
            img_aligned = foo(imgA_aligned, imgB_aligned)
        else:
            img_aligned = foo(imgA_aligned, imgB_aligned, **args)

    return img_aligned, int(imgA_to_imgB_x), int(imgA_to_imgB_y), aligned_to_B_x, aligned_to_B_y


def unite(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return utils.erase_noise_point(np.logical_or(imgA, imgB), 4)


def intersect(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return utils.erase_noise_point(np.logical_and(imgA, imgB), 4)


def subtract(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return utils.erase_noise_point(np.logical_and(imgA, np.logical_not(np.logical_and(imgA, imgB))), 4)


def backward_subtract(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return utils.erase_noise_point(np.logical_and(imgB, np.logical_not(np.logical_and(imgB, imgA))), 4)


def xor(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return utils.erase_noise_point(np.logical_xor(imgA, imgB), 4)


def shadow_mask_unite(A, B):

    shadow_A = utils.fill_holes(A)
    shadow_B = utils.fill_holes(B)

    score, shadow_A_to_B_x, shadow_A_to_B_y = jaccard.jaccard_coef(A, B)

    shadow_A_aligned, shadow_B_aligned, _, _ = utils.align(shadow_A, shadow_B, shadow_A_to_B_x, shadow_A_to_B_y)
    mask = np.logical_and(shadow_A_aligned, shadow_B_aligned)

    A_aligned, B_aligned, _, _ = utils.align(A, B, shadow_A_to_B_x, shadow_A_to_B_y)
    union = np.logical_or(A_aligned, B_aligned)

    masked_union = np.logical_and(mask, union)

    return utils.trim_binary_image(masked_union)


def unite_align(A, B, C):
    _, A_to_C_x, A_to_C_y = jaccard.jaccard_coef(A, C)
    _, B_to_C_x, B_to_C_y = jaccard.jaccard_coef(B, C)
    A_to_B_x = A_to_C_x - B_to_C_x
    A_to_B_y = A_to_C_y - B_to_C_y
    return A_to_B_x, A_to_B_y


def inv_unite(A, B, C):
    _, B_to_A_x, B_to_A_y = jaccard.jaccard_coef(B, A)
    _, C_to_A_x, C_to_A_y = jaccard.jaccard_coef(C, A)
    B_to_C_x = B_to_A_x - C_to_A_x
    B_to_C_y = B_to_A_y - C_to_A_y
    B_aligned, C_aligned, aligned_to_C_x, aligned_to_C_y = utils.align(B, C, B_to_C_x, B_to_C_y)
    B_new = np.logical_and(B_aligned, np.logical_not(C_aligned))
    aligned_to_A_x = aligned_to_C_x + C_to_A_x
    aligned_to_A_y = aligned_to_C_y + C_to_A_y
    A_aligned, B_aligned, _, _ = utils.align(A, B_new, -aligned_to_A_x, -aligned_to_A_y)
    C_new = np.logical_and(A_aligned, np.logical_not(B_aligned))
    C_new = utils.trim_binary_image(utils.erase_noise_point(C_new, 8))

    return C_new


def preserving_subtract_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref, coords = False):
    return subtract_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref, coords)


def evaluate_shape_topo_mapping(u1, u2, u3):
    """
    1) horizontally, u1 and u2 have some common (or very similar)components such that
    an injective mapping exists by the correspondence of similar pairs.
    2) vertically, u1 and u3 share the same topological structure by considering
    two relations, "inside" and "outside".
    :param u1: an binary image
    :param u2: a binary iamge
    :param u3: a binary image
    :return: MAT score
    """

    u1_coms, _, _ = utils.decompose(u1, 8, trim = False)
    u2_coms, _, _ = utils.decompose(u2, 8, trim = False)
    u3_coms, _, _ = utils.decompose(u3, 8, trim = False)

    # old_jcm_u1_com_ids, old_jcm_u2_com_ids, old_jcm_score = map.jaccard_map(u1_coms, u2_coms)

    jcm_u1_com_ids, jcm_u2_com_ids, jcm_score = map.soft_jaccard_map(u1_coms, u2_coms)

    tpm_u1_com_ids, tpm_u3_com_ids, tpm_score = map.topological_map(u1_coms, u3_coms)

    mat_score = (jcm_score + tpm_score) / 2

    stub = utils.make_stub(u1_coms, u2_coms, u3_coms,
                           jcm_u1_com_ids, jcm_u2_com_ids, tpm_u1_com_ids, tpm_u3_com_ids)

    return mat_score, stub


def evaluate_shape_loc_isomorphism(u1, u2, u3):
    """
    1) horizontally, u1 and u2 have some components that are located at the same (or close)
    positions such that an injective mapping can be formed by location correspondence.
    2) horizontally, u1 and u2 have some components that have the same or similar shape such aht
    and injective mapping can be found by shape correspondence.
    3) vertically, u1 and u3 forms a placeholder mapping that requires only that u1 and u3 have the same number
    of components. The placeholder mapping will be initiated by one or more mappings that express the isomorphism
    between (u1, u2) and (u3, opt), where (u1, u2) represents the mappings btw components in u1 and u2 and (u3, opt)
    represents the mappings btw components in u3 and opt.
    :param u1: an binary image
    :param u2: a binary iamge
    :param u3: a binary image
    :return: MAT score
    """

    u1_coms, _, _ = utils.decompose(u1, 8, trim = False)
    u2_coms, _, _ = utils.decompose(u2, 8, trim = False)
    u3_coms, _, _ = utils.decompose(u3, 8, trim = False)

    lcm_u1_com_ids, lcm_u2_com_ids, lcm_score = map.location_map(u1_coms, u2_coms)
    # old_jcm_u1_com_ids, old_jcm_u2_com_ids, old_jcm_score = map.jaccard_map(u1_coms, u2_coms)
    jcm_u1_com_ids, jcm_u2_com_ids, jcm_score = map.soft_jaccard_map(u1_coms, u2_coms)
    if 1 != len(lcm_u1_com_ids) and \
            len(lcm_u1_com_ids) == len(jcm_u1_com_ids) and \
            len(lcm_u2_com_ids) == len(jcm_u2_com_ids) and \
            (np.unique(lcm_u1_com_ids) == np.unique(jcm_u1_com_ids)).all() and \
            (np.unique(lcm_u2_com_ids) == np.unique(jcm_u2_com_ids)).all():
        phm_u1_com_ids, phm_u3_com_ids, phm_score = map.placeholder_map(u1_coms, u3_coms)
    else:
        phm_u1_com_ids, phm_u3_com_ids, phm_score = (None, None, 0)

    mat_score = min((lcm_score + jcm_score) / 2, phm_score)

    stub = utils.make_stub(u1_coms, u2_coms, u3_coms,
                           lcm_u1_com_ids, lcm_u2_com_ids,
                           jcm_u1_com_ids, jcm_u2_com_ids,
                           phm_u1_com_ids, phm_u3_com_ids)

    return mat_score, stub


def evaluate_shape_delta_loc_isomorphism(u1, u2, u3):

    u1_coms, _, _ = utils.decompose(u1, 8, trim = False)
    u2_coms, _, _ = utils.decompose(u2, 8, trim = False)
    u3_coms, _, _ = utils.decompose(u3, 8, trim = False)

    # old_jcm_u1_com_ids, old_jcm_u2_com_ids, old_jcm_score = map.jaccard_map(u1_coms, u2_coms)
    jcm_u1_com_ids, jcm_u2_com_ids, jcm_score = map.soft_jaccard_map(u1_coms, u2_coms)

    if 1 == len(jcm_u1_com_ids):
        mat_score = 0
    else:
        mat_score = jcm_score

    stub = utils.make_stub(u1_coms, u2_coms, u3_coms,
                           jcm_u1_com_ids, jcm_u2_com_ids)

    return mat_score, stub


def evaluate_topo_delta_shape_isomorphism(u1, u2, u3):

    u1_coms, _, _ = utils.decompose(u1, 8, trim = False)
    u2_coms, _, _ = utils.decompose(u2, 8, trim = False)
    u3_coms, _, _ = utils.decompose(u3, 8, trim = False)

    tpm_u1_com_ids, tpm_u2_com_ids, tpm_score = map.topological_map(u1_coms, u2_coms)

    if 1 == len(tpm_u1_com_ids):
        mat_score = 0
    else:
        mat_score = tpm_score

    stub = utils.make_stub(u1_coms, u2_coms, u3_coms,
                           tpm_u1_com_ids, tpm_u2_com_ids)

    return mat_score, stub


# def evaluate_WWW(u1, u2, u3):
#
#     u1_coms, _, _ = utils.decompose(u1, 8, trim = False)
#     u2_coms, _, _ = utils.decompose(u2, 8, trim = False)
#     u3_coms, _, _ = utils.decompose(u3, 8, trim = False)
#
#     jcm_u1_u2_u1_com_ids, jcm_u1_u2_u2_com_ids, jcm_u1_u2_score = map.jaccard_map(u1_coms, u2_coms)
#     jcm_u1_u3_u1_com_ids, jcm_u1_u3_u3_com_ids, jcm_u1_u3_score = map.jaccard_map(u1_coms, u3_coms)
#
#     mat_score = (jcm_u1_u2_score + jcm_u1_u3_score) / 2
#
#     stub = utils.make_stub(u1_coms, u2_coms, u3_coms,
#                            jcm_u1_u2_u1_com_ids, jcm_u1_u2_u2_com_ids,
#                            jcm_u1_u3_u1_com_ids, jcm_u1_u3_u3_com_ids)
#
#     return mat_score, stub


def evaluate_duplicate(u1, u2):

    scores = []
    u1_to_u2_xs = []
    u1_to_u2_ys = []
    current = u2.copy()
    current_to_u2_x = 0
    current_to_u2_y = 0
    u1_tr = utils.trim_binary_image(u1)
    while current.sum():
        old_score_tmp, old_diff_tmp_to_u1_x, old_diff_tmp_to_u1_y, old_diff_tmp_to_current_x, old_diff_tmp_to_current_y, old_diff_tmp = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u1, current)

        score_tmp, diff_tmp_to_u1_x, diff_tmp_to_u1_y, diff_tmp_to_current_x, diff_tmp_to_current_y, diff_tmp = \
            soft_jaccard.soft_jaccard(u1, current, asymmetric = True)

        if score_tmp < 0.9:
            break

        scores.append(score_tmp)
        u1_to_current_x = (-diff_tmp_to_u1_x) - (-diff_tmp_to_current_x)
        u1_to_current_y = (-diff_tmp_to_u1_y) - (-diff_tmp_to_current_y)
        u1_to_u2_x = u1_to_current_x + current_to_u2_x
        u1_to_u2_y = u1_to_current_y + current_to_u2_y
        u1_to_u2_xs.append(u1_to_u2_x)
        u1_to_u2_ys.append(u1_to_u2_y)

        current = diff_tmp
        current_to_u2_x = diff_tmp_to_current_x + current_to_u2_x
        current_to_u2_y = diff_tmp_to_current_y + current_to_u2_y

    if 1 >= len(scores):
        mat_score = 0
        u1_to_u2_locs = []
    else:
        mat_score = np.mean(scores)
        u1_to_u2_xs = ((np.array(u1_to_u2_xs) - min(u1_to_u2_xs)) / u1_tr.shape[1]).tolist()
        u1_to_u2_ys = ((np.array(u1_to_u2_ys) - min(u1_to_u2_ys)) / u1_tr.shape[0]).tolist()
        u1_to_u2_locs = np.array(list(zip(u1_to_u2_xs, u1_to_u2_ys)))

    stub = utils.make_stub(u1_to_u2_locs)

    return mat_score, stub


def evaluate_shape_texture_transfer(u1, u2, u3):

    u1_filled = utils.fill_holes(u1)
    u2_filled = utils.fill_holes(u2)
    u3_filled = utils.fill_holes(u3)

    old_u1_u3_shape_index = jaccard.jaccard_coef(u1_filled, u3_filled)[0]

    u1_u3_shape_index = soft_jaccard.soft_jaccard(u1_filled, u3_filled)[0]

    u1_texture_index = np.logical_and(u1_filled, np.logical_not(u1)).sum() / u1_filled.sum()
    u2_texture_index = np.logical_and(u2_filled, np.logical_not(u2)).sum() / u2_filled.sum()
    u3_texture_index = np.logical_and(u3_filled, np.logical_not(u3)).sum() / u3_filled.sum()
    u1_u2_texture_index = u1_texture_index - u2_texture_index
    u1_u3_texture_index = u1_texture_index - u3_texture_index

    # _, u1_to_u3_x, u1_to_u3_y = jaccard.jaccard_coef(u1, u3)
    # u1_texture_index, u3_texture_index = utils.texture_index(u1, u3, u1_filled, u3_filled, u1_to_u3_x, u1_to_u3_y)
    # texture_score = 1 - abs(u1_texture_index - u3_texture_index)

    # mat_score = (1 - abs(u1_u3_texture_index) + abs(u1_u2_texture_index) + u1_u3_shape_index) / 3
    mat_score = 1 - abs(u1_u3_texture_index)

    stub = utils.make_stub(u2_texture_index, u1_u2_texture_index, u3_filled, u2_filled, u1_u3_shape_index)

    return mat_score, stub



