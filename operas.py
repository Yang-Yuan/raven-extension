from matplotlib import pyplot as plt
from skimage.transform import rotate
import numpy as np


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


def octalize(img, show_me = False):
    """

    :param img:
    :return:
    """
    octals = []
    for angle in np.arange(0, 360, 90):
        octals.append(rot_binary(img, angle))

    tmp = mirror_left_right(img)
    for angle in np.arange(0, 360, 90):
        octals.append(rot_binary(tmp, angle))

    if show_me:
        fig, axs = plt.subplots(2, 4)
        for img, ax in zip(octals, axs.flatten()):
            ax.imshow(img, cmap = "binary")
        plt.show()

    return octals

def what_refection_rotation(int):
