from matplotlib import pyplot as plt
from skimage.transform import rotate
from sys import modules
import numpy as np
import metrics

THIS = modules[__name__]


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


def unary_transform(img, show_me = False):
    """

    :param show_me:
    :param img:
    :return:
    """
    transformed_images = []
    for angle in np.arange(0, 360, 90):
        transformed_images.append(rot_binary(img, angle))

    tmp = mirror_left_right(img)
    for angle in np.arange(0, 360, 90):
        transformed_images.append(rot_binary(tmp, angle))

    if show_me:
        fig, axs = plt.subplots(2, 4)
        for img, ax in zip(transformed_images, axs.flatten()):
            ax.imshow(img, cmap = "binary")
        plt.show()

    transformations = [
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
         {"name": "rot_binary", "args": {"angle": 270}}]
    ]

    return transformed_images, transformations


def apply_unary_transformation(img, unary_transformations, show_me = False):
    for tran in unary_transformations:

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


def apply_binary_transformation(imgA, imgB,
                                align_x, align_y,
                                binary_transformations):
    imgA_aligned, imgB_aligned = align(imgA, imgB, align_x, align_y)

    img = None

    for tran in binary_transformations:

        name = tran.get("name")

        if name is None:
            continue
        foo = getattr(THIS, name)

        args = tran.get("args")
        if args is None:
            img = foo(imgA_aligned, imgB_aligned)
        else:
            img = foo(imgA_aligned, imgB_aligned, **args)

    return img


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


def binary_transform(imgA, imgB, show_me = False):
    # TODO this alignment must be enhanced in the future.
    _, align_x, align_y = metrics.jaccard_coef(imgA, imgB)

    A_aligned, B_aligned = align(imgA, imgB, align_x, align_y)

    transformed_images = [unite(A_aligned, B_aligned),
                          intersect(A_aligned, B_aligned),
                          subtract(A_aligned, B_aligned),
                          backward_subtract(A_aligned, B_aligned),
                          xor(A_aligned, B_aligned)]

    transformations = [
        [{"name": "unite"}],
        [{"name": "intersect"}],
        [{"name": "subtract"}],
        [{"name": "backward_subtract"}],
        [{"name": "xor"}]]

    if show_me:
        fig, axs = plt.subplots(1, 5)
        for img, ax in zip(transformed_images, axs):
            ax.imshow(img, cmap = "binary")
        plt.show()

    return transformed_images, transformations, align_x, align_y


def argmax_binary_sim(similarities, default = None):
    tran_n = np.argmax(similarities)
    switch = {
        # code : [mirror_left_right, degree to rotate]
        0: ["unite"],
        1: ["intersect"],
        2: ["subtract"],
        3: [False, 270],
        4: [True, 0],
    }
    return switch.get(tran_n, default)
