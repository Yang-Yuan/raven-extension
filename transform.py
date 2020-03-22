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
    {"name": "identity", "value": [{"name": None}], "type": "unary"},
    {"name": "rot_90", "value": [{"name": "rot_binary", "args": {"angle": 90}}], "type": "unary"},
    {"name": "rot_180", "value": [{"name": "rot_binary", "args": {"angle": 180}}], "type": "unary"},
    {"name": "rot_270", "value": [{"name": "rot_binary", "args": {"angle": 270}}], "type": "unary"},
    {"name": "mirror", "value": [{"name": "mirror_left_right"}], "type": "unary"},
    {"name": "mirror_rot_90", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 90}}], "type": "unary"},
    {"name": "mirror_rot_180", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 180}}], "type": "unary"},
    {"name": "mirror_rot_270", "value": [{"name": "mirror_left_right"}, {"name": "rot_binary", "args": {"angle": 270}}], "type": "unary"},
    {"name": "rescale", "value": [{"name": "rescale", "args": {"x_factor": 1.3, "y_factor": 1.4}}], "type": "unary"},
    # {"name": "upscale_to", "value": [{"name": "upscale_to"}], "type": "unary"},
    {"name": "add_diff", "value": [{"name": "add_diff"}], "type": "unary"},
    {"name": "subtract_diff", "value": [{"name": "subtract_diff"}], "type": "unary"},
    {"name": "xor_diff", "value": [{"name": "xor_diff"}], "type": "unary"}
]

binary_transformations = [
    {"name": "unite", "value": [{"name": "unite"}], "type": "binary"},
    {"name": "intersect", "value": [{"name": "intersect"}], "type": "binary"},
    {"name": "subtract", "value": [{"name": "subtract"}], "type": "binary"},
    {"name": "backward_subtract", "value": [{"name": "backward_subtract"}], "type": "binary"},
    {"name": "xor", "value": [{"name": "xor"}], "type": "binary"}
]

all_trans = unary_transformations + binary_transformations


def get_tran(tran_name):
    for tran in all_trans:
        if tran_name == tran.get("name"):
            return tran


def rescale(img, x_factor, y_factor):

    return utils.grey_to_binary(rs(np.logical_not(img), (y_factor, x_factor), order = 0), 0.7)

    # return utils.grey_to_binary(resize(np.logical_not(img), shape, order = 0), 0.7)


def upscale_to(img, ref):
    img_shape_y, img_shape_x = img.shape
    ref_shape_y, ref_shape_x = ref.shape
    max_y = max(img_shape_y, ref_shape_y)
    max_x = max(img_shape_x, ref_shape_x)
    return utils.grey_to_binary(resize(np.logical_not(img), (max_y, max_x), order = 0), 0.7)


def subtract_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref):
    """

    :param img:
    :param diff_to_ref_x:
    :param diff_to_ref_y:
    :param diff:
    :param ref:
    :return:
    """

    _, img_to_ref_x, img_to_ref_y = jaccard.jaccard_coef(img, ref)
    diff_to_img_x = diff_to_ref_x - img_to_ref_x
    diff_to_img_y = diff_to_ref_y - img_to_ref_y

    diff_aligned, img_aligned, _, _ = utils.align(diff, img, diff_to_img_x, diff_to_img_y)
    return utils.erase_noise_point(utils.trim_binary_image(np.logical_and(img_aligned, np.logical_not(diff_aligned))), 4)


def add_diff(img, diff_to_ref_x, diff_to_ref_y, diff, ref):
    """

    :param img:
    :param diff_to_ref_x:
    :param diff_to_ref_y:
    :param diff:
    :param ref:
    :return:
    """

    _, img_to_ref_x, img_to_ref_y = jaccard.jaccard_coef(img, ref)
    diff_to_img_x = diff_to_ref_x - img_to_ref_x
    diff_to_img_y = diff_to_ref_y - img_to_ref_y

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
                                expect = None):
    if imgA_to_imgB_x is None or imgA_to_imgB_y is None:
        _, imgA_to_imgB_x, imgA_to_imgB_y = jaccard.jaccard_coef(imgA, imgB)

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




