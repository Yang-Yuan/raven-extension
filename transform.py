from matplotlib import pyplot as plt
from skimage.transform import rotate
from skimage.transform import rescale as rs
from skimage.transform import resize
from sys import modules
import numpy as np
import jaccard
import utils

THIS = modules[__name__]

unary_transformations = [
    {"name": "identity", "value": [{"name": None}]},
    # {"name": "rot_90", "value": [{"name": "rot_binary", "args": {"angle": 90}}]},
    # {"name": "rot_180", "value": [{"name": "rot_binary", "args": {"angle": 180}}]},
    # {"name": "rot_270", "value": [{"name": "rot_binary", "args": {"angle": 270}}]},
    {"name": "mirror", "value": [{"name": "mirror_left_right"}]},
    # {"name": "mirror_rot_90", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 90}}]},
    # {"name": "mirror_rot_180", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 180}}]},
    # {"name": "mirror_rot_270", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 270}}]},
    {"name": "rescale", "value": [{"name": "rescale", "args": {"x_factor": 1.3, "x_factor": 1.4}}]},
    {"name": "add_diff", "value": [{"name": "add_diff"}]},
    {"name": "subtract_diff", "value": [{"name": "subtract_diff"}]}
]

binary_transformations = [
    {"name": "unite", "value": [{"name": "unite"}]},
    {"name": "intersect", "value": [{"name": "intersect"}]},
    # {"name": "subtract", "value": [{"name": "subtract"}]},
    # {"name": "backward_subtract", "value": [{"name": "backward_subtract"}]},
    # {"name": "xor", "value": [{"name": "xor"}]}
]

all_trans = unary_transformations + binary_transformations


def get_tran(tran_name):
    for tran in all_trans:
        if tran_name == tran.get("name"):
            return tran


def rescale(img, x_factor, y_factor):

    return utils.grey_to_binary(rs(np.logical_not(img), (y_factor, x_factor), order = 0), 0.7)

    # return utils.grey_to_binary(resize(np.logical_not(img), shape, order = 0), 0.7)


def subtract_diff(img, align_x, align_y, diff):
    """
    Subtract diff from img.
    The top-left corner of diff is at the (align_x, align_y)
    with the top-left corner of img as the origin
    :param img:
    :param align_x:
    :param align_y:
    :param diff:
    :return: trimmed result
    """
    diff_aligned, img_aligned = align(diff, img, align_x, align_y)
    return utils.trim(np.logical_and(img_aligned, np.logical_not(diff_aligned)))


def add_diff(img, align_x, align_y, diff):
    """
    add diff to img.
    The top-left corner of diff is at the (align_x, align_y)
    with the top-left corner of img as the origin
    :param img:
    :param align_x:
    :param align_y:
    :param diff:
    :return: trimmed result
    """
    diff_aligned, img_aligned = align(diff, img, align_x, align_y)
    return utils.trim(np.logical_or(img_aligned, diff_aligned))


# def add_diff(img, align_x, align_y, diff, diff_is_positive):
#     if diff_is_positive:
#         diff_aligned, img_aligned = align(diff, img, align_x, align_y)
#         result = np.logical_or(diff_aligned, img_aligned)
#     else:
#         _, align_x, align_y = jaccard.jaccard_coef(np.logical_not(diff), img)
#
#         diff_y, diff_x = diff.shape
#         img_y, img_x = img.shape
#
#         if align_y < 0:
#             diff_y_min = -align_y
#             img_y_min = 0
#         else:
#             diff_y_min = 0
#             img_y_min = align_y
#
#         if align_x < 0:
#             diff_x_min = -align_x
#             img_x_min = 0
#         else:
#             diff_x_min = 0
#             img_x_min = align_x
#
#         if align_y + diff_y > img_y:
#             diff_y_max = img_y - (align_y + diff_y)
#             img_y_max = img_y
#         else:
#             diff_y_max = diff_y
#             img_y_max = align_y + diff_y
#
#         if align_x + diff_x > img_x:
#             diff_x_max = img_x - (align_x + diff_x)
#             img_x_max = img_x
#         else:
#             diff_x_max = diff_x
#             img_x_max = align_x + diff_x
#
#         img_bounded = img[img_y_min: img_y_max, img_x_min: img_x_max]
#         diff_bounded = diff[diff_y_min: diff_y_max, diff_x_min: diff_x_max]
#
#         result = np.copy(img)
#         result[img_y_min: img_y_max, img_x_min: img_x_max] = np.logical_and(img_bounded, diff_bounded)
#
#     return result


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


# def unary_transform(img, show_me = False):
#     """
#
#     :param show_me:
#     :param img:
#     :return:
#     """
#     transformed_images = []
#     for angle in np.arange(0, 360, 90):
#         transformed_images.append(rot_binary(img, angle))
#
#     tmp = mirror_left_right(img)
#     for angle in np.arange(0, 360, 90):
#         transformed_images.append(rot_binary(tmp, angle))
#
#     if show_me:
#         fig, axs = plt.subplots(2, 4)
#         for img, ax in zip(transformed_images, axs.flatten()):
#             ax.imshow(img, cmap = "binary")
#         plt.show()
#
#     transformations = [
#         [{"name": None}],
#         [{"name": "rot_binary", "args": {"angle": 90}}],
#         [{"name": "rot_binary", "args": {"angle": 180}}],
#         [{"name": "rot_binary", "args": {"angle": 270}}],
#         [{"name": "mirror_left_right"}],
#         [{"name": "mirror_left_right"},
#          {"name": "rot_binary", "args": {"angle": 90}}],
#         [{"name": "mirror_left_right"},
#          {"name": "rot_binary", "args": {"angle": 180}}],
#         [{"name": "mirror_left_right"},
#          {"name": "rot_binary", "args": {"angle": 270}}],
#         [{"name": "extend"}],
#         [{"name": "backward_extend"}]
#     ]
#
#     return transformed_images, transformations


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
                                imgA_to_imgB_x = None, imgA_to_imgB_y = None):
    if imgA_to_imgB_x is None or imgA_to_imgB_y is None:
        _, imgA_to_imgB_x, imgA_to_imgB_y = jaccard.jaccard_coef(imgA, imgB)

    imgA_aligned, imgB_aligned = align(imgA, imgB, imgA_to_imgB_x, imgA_to_imgB_y)

    img = None

    binary_trans = tran.get("value")

    for tran in binary_trans:

        name = tran.get("name")

        if name is None:
            continue
        foo = getattr(THIS, name)

        args = tran.get("args")
        if args is None:
            img = foo(imgA_aligned, imgB_aligned)
        else:
            img = foo(imgA_aligned, imgB_aligned, **args)

    return img, int(imgA_to_imgB_x), int(imgA_to_imgB_y)


def unite(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return np.logical_or(imgA, imgB)


def intersect(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return np.logical_and(imgA, imgB)


def subtract(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return np.logical_and(imgA, np.logical_not(np.logical_and(imgA, imgB)))


def backward_subtract(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return np.logical_and(imgB, np.logical_not(np.logical_and(imgB, imgA)))


def xor(imgA, imgB):
    if imgA.shape != imgB.shape:
        raise Exception("Crap!")

    return np.logical_xor(imgA, imgB)


def align(imgA, imgB, x, y):
    """
    Align imgA to imgB.
    Consider the top-left corner of imgB as the origin
    the top-left corner of imgA should be as (x, y) using this origin.
    Output A_aligned and B_aligned trimmed to the smallest shape
    such that if you superimpose A_aligned on B_aligned
    no true pixels will fall out of the boundary.
    :param imgA:
    :param imgB:
    :param x:
    :param y:
    :return: A_aligned, B_aligned
    """
    A_shape_y, A_shape_x = imgA.shape

    B_expanded = np.pad(imgB, ((A_shape_y, A_shape_y), (A_shape_x, A_shape_x)), constant_values = False)
    A_expanded = np.full_like(B_expanded, False)
    A_expanded[y + A_shape_y: y + A_shape_y * 2, x + A_shape_x: x + A_shape_x * 2] = imgA
    AB_expanded = np.logical_or(A_expanded, B_expanded)

    y_AB, x_AB = np.where(AB_expanded)
    y_AB_min = y_AB.min()
    y_AB_max = y_AB.max() + 1
    x_AB_min = x_AB.min()
    x_AB_max = x_AB.max() + 1

    A_aligned = A_expanded[y_AB_min: y_AB_max, x_AB_min: x_AB_max]
    B_aligned = B_expanded[y_AB_min: y_AB_max, x_AB_min: x_AB_max]

    return A_aligned, B_aligned




# def binary_transform(imgA, imgB, show_me = False):
#     # TODO this alignment must be enhanced in the future.
#     _, align_x, align_y = jaccard.jaccard_coef(imgA, imgB)
#
#     A_aligned, B_aligned = align(imgA, imgB, align_x, align_y)
#
#     transformed_images = [unite(A_aligned, B_aligned),
#                           intersect(A_aligned, B_aligned),
#                           subtract(A_aligned, B_aligned),
#                           backward_subtract(A_aligned, B_aligned),
#                           xor(A_aligned, B_aligned)]
#
#     transformations = [
#         [{"name": "unite"}],
#         [{"name": "intersect"}],
#         [{"name": "subtract"}],
#         [{"name": "backward_subtract"}],
#         [{"name": "xor"}]]
#
#     if show_me:
#         fig, axs = plt.subplots(1, 5)
#         for img, ax in zip(transformed_images, axs):
#             ax.imshow(img, cmap = "binary")
#         plt.show()
#
#     return transformed_images, transformations, align_x, align_y


# def argmax_binary_sim(similarities, default = None):
#     tran_n = np.argmax(similarities)
#     switch = {
#         # code : [mirror_left_right, degree to rotate]
#         0: ["unite"],
#         1: ["intersect"],
#         2: ["subtract"],
#         3: [False, 270],
#         4: [True, 0],
#     }
#     return switch.get(tran_n, default)
