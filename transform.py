from matplotlib import pyplot as plt
from skimage.transform import rotate
from sys import modules
import numpy as np
import jaccard

THIS = modules[__name__]

unary_transformations = [
    [{"name": None}],
    [{"name": "rot_binary", "args": {"angle": 90}}],
    [{"name": "rot_binary", "args": {"angle": 180}}],
    [{"name": "rot_binary", "args": {"angle": 270}}],
    [{"name": "mirror_left_right"}],
    [{"name": "mirror_left_right"},
     {"name": "rot_binary", "args": {"angle": 90}}],
    [{"name": "mirror_left_right"},
     {"name": "rot_binary", "args": {"angle": 180}}],
    [{"name": "mirror_left_right"},
     {"name": "rot_binary", "args": {"angle": 270}}],
    # [{"name": "rescale", "args": {"scale": 0.25}}],
    [{"name": "rescale", "args": {"scale": 0.5}}],
    [{"name": "rescale", "args": {"scale": 2}}],
    # [{"name": "rescale", "args": {"scale": 4}}],
    [{"name": "add_diff"}]
]

binary_transformations = [
    [{"name": "unite"}],
    [{"name": "intersect"}],
    [{"name": "subtract"}],
    [{"name": "backward_subtract"}],
    [{"name": "xor"}]
]


def rescale(img, scale):
    """

    :param img:
    :param scale:
    :return:
    """
    y_shape, x_shape = img.shape
    if scale == 0.25:
        y_before = 0
        y_after = 0
        x_before = 0
        x_after = 0
        y_reminder = y_shape % 4
        x_reminder = x_shape % 4
        if 0 != y_reminder:
            y_pad = 4 - y_reminder
            y_before = int(y_pad / 2)
            y_after = y_pad - y_before

        if 0 != x_reminder:
            x_pad = 4 - x_reminder
            x_before = int(x_pad / 2)
            x_after = x_pad - x_before

        img_padded = np.pad(img, ((y_before, y_after), (x_before, x_after)), constant_values = False)
        y_padded_shape, x_padded_shape = img_padded.shape
        img_scaled = np.full((int(y_padded_shape / 4), int(x_padded_shape / 4)), False)
        for yy in range(img_scaled.shape[0]):
            for xx in range(img_scaled.shape[1]):
                yy_padded = 4 * yy
                xx_padded = 4 * xx
                img_scaled[yy, xx] = img_padded[yy_padded : yy_padded + 4, xx_padded : xx_padded + 4].sum() > 8

        return img_scaled

    elif 0.5 == scale:
        y_before = 0
        x_before = 0
        y_after = y_shape % 2
        x_after = x_shape % 2
        img_padded = np.pad(img, ((y_before, y_after), (x_before, x_after)), constant_values = False)
        y_padded_shape, x_padded_shape = img_padded.shape
        img_scaled = np.full((int(y_padded_shape / 2), int(x_padded_shape / 2)), False)
        for yy in range(img_scaled.shape[0]):
            for xx in range(img_scaled.shape[1]):
                yy_padded = 2 * yy
                xx_padded = 2 * xx
                img_scaled[yy, xx] = img_padded[yy_padded: yy_padded + 2, xx_padded: xx_padded + 2].sum() > 2

        return img_scaled

    elif 2 == scale:
        img_scaled = np.full((int(y_shape * 2), int(x_shape * 2)), False)
        for yy in range(y_shape):
            for xx in range(x_shape):
                yy_scaled = yy * 2
                xx_scaled = xx * 2
                img_scaled[yy_scaled : yy_scaled + 2, xx_scaled : xx_scaled + 2] = img[yy, xx]

        return img_scaled

    elif 4 == scale:
        img_scaled = np.full((int(y_shape * 4), int(x_shape * 4)), False)
        for yy in range(y_shape):
            for xx in range(x_shape):
                yy_scaled = yy * 4
                xx_scaled = xx * 4
                img_scaled[yy_scaled: yy_scaled + 4, xx_scaled: xx_scaled + 4] = img[yy, xx]

        return img_scaled

    else:
        raise Exception("Ryan")


def add_diff(img, align_x, align_y, diff, diff_is_positive):
    # if diff is negative, then align again.
    if not diff_is_positive:
        _, align_x, align_y = jaccard.jaccard_coef(np.logical_not(diff), img)

    diff_y, diff_x = diff.shape
    img_y, img_x = img.shape

    if align_y < 0:
        diff_y_min = -align_y
        img_y_min = 0
    else:
        diff_y_min = 0
        img_y_min = align_y

    if align_x < 0:
        diff_x_min = -align_x
        img_x_min = 0
    else:
        diff_x_min = 0
        img_x_min = align_x

    if align_y + diff_y > img_y:
        diff_y_max = img_y - (align_y + diff_y)
        img_y_max = img_y
    else:
        diff_y_max = diff_y
        img_y_max = align_y + diff_y

    if align_x + diff_x > img_x:
        diff_x_max = img_x - (align_x + diff_x)
        img_x_max = img_x
    else:
        diff_x_max = diff_x
        img_x_max = align_x + diff_x

    img_bounded = img[img_y_min: img_y_max, img_x_min: img_x_max]
    diff_bounded = diff[diff_y_min: diff_y_max, diff_x_min: diff_x_max]

    result = np.copy(img)
    if diff_is_positive:
        result[img_y_min: img_y_max, img_x_min: img_x_max] = np.logical_or(img_bounded, diff_bounded)
    else:
        result[img_y_min: img_y_max, img_x_min: img_x_max] = np.logical_and(img_bounded, diff_bounded)

    return result


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


def apply_unary_transformation(img, unary_trans, show_me = False):
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


def apply_binary_transformation(imgA, imgB, binary_trans,
                                align_x = None, align_y = None):
    if align_x is None or align_y is None:
        _, align_x, align_y = jaccard.jaccard_coef(imgA, imgB)

    imgA_aligned, imgB_aligned = align(imgA, imgB, align_x, align_y)

    img = None

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

    return img, align_x, align_y


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

    :param imgA:
    :param imgB:
    :param x: the output of  metrics.jaccard_coef(imgA, imgB)
    :param y: he output of  metrics.jaccard_coef(imgA, imgB)
    :return:
    """
    A_shape_y, A_shape_x = imgA.shape
    B_shape_y, B_shape_x = imgB.shape

    AB_expanded = np.pad(imgB,
                         ((A_shape_y, A_shape_y), (A_shape_x, A_shape_x)),
                         constant_values = False)
    AB_expanded[y + A_shape_y: y + 2 * A_shape_y, x + A_shape_x: x + 2 * A_shape_x] = imgA
    y_AB, x_AB = np.where(AB_expanded)
    y_AB_max = y_AB.max() + 1
    x_AB_min = x_AB.min()
    y_AB_min = y_AB.min()
    x_AB_max = x_AB.max() + 1

    A_expanded = np.full_like(AB_expanded, False)
    B_expanded = np.full_like(AB_expanded, False)
    A_expanded[y + A_shape_y: y + 2 * A_shape_y, x + A_shape_x: x + 2 * A_shape_x] = imgA
    B_expanded[A_shape_y: A_shape_y + B_shape_y, A_shape_x: A_shape_x + B_shape_x] = imgB

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
