from matplotlib import pyplot as plt
import numpy as np
from skimage import measure


def rgb_to_binary(img, bg_color, tolerance):
    """
    convert img to a binary image (0, 1)
    :param img: image to be converted
    :param bg_color: background color
    :param tolerance:  tolerance
    :return: binary image
    """

    rgb = img[:, :, : 3]

    return np.logical_not(np.average(abs(rgb - bg_color), axis = -1) < tolerance)


def grey_to_binary(img, threshold):
    """
    convert img to a binary image (0, 1)
    :param img: image to be converted
    :param threshold:
    :return: binary image
    """
    return img < threshold


def cut(img, rectangles, show_me = False):
    """
    cut our the rectangles from the img
    :param img: img
    :param rectangles: lists of [x1, x2, y1, y2]
    :return:
    """
    imgs = []
    for rect in rectangles:
        [x1, x2, y1, y2] = rect
        imgs.append(img[y1: y2, x1: x2])

    if show_me:
        fig, axs = plt.subplots(1, len(imgs))
        for img, ax in zip(imgs, axs.flatten()):
            ax.imshow(img, cmap = "binary")
        plt.show()

    return imgs


def pad(img, r, show_me):
    """

    :param img:
    :param r:
    :return:
    """

    y_shape, x_shape = img.shape[:2]
    y_pad = round(y_shape * r)
    x_pad = round(x_shape * r)

    padded = np.pad(img,
                    ((y_pad, y_pad), (x_pad, x_pad), (0, 0)),
                    mode = "constant",
                    constant_values = img.max())

    if show_me:
        plt.imshow(padded)
        plt.show()

    return padded


def extract_components(img, coords):
    """

    :param img:
    :param coords:  list of coordinates of objects in a single puzzle image
    :return:
    """
    return [img[y: y + delta_y, x: x + delta_x]
            for x, y, delta_x, delta_y in coords]


def trim_binary_image(img):
    if 2 != len(img.shape):
        raise Exception("Crap!")

    if (img == False).all():
        return img

    y, x = np.where(img)
    y_max = y.max() + 1
    x_min = x.min()
    x_max = x.max() + 1
    y_min = y.min()

    return img[y_min: y_max, x_min:x_max]


def erase_noise_point(img, noise_point_size):
    labels, label_num = measure.label(input = img, background = False, return_num = True, connectivity = 1)
    sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
    for size, label in zip(sizes, range(1, label_num + 1)):
        if size < noise_point_size:
            img[labels == label] = False
    return img



