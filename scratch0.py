
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import rescale
from RectSelector import RectSelector
import utils
import operas
import metrics

# preprocessing
raw_img = rescale(image = plt.imread("./problems/X1.gif"),
                  scale = (0.25, 0.25, 1))

raw_img = utils.pad(raw_img, r = 0.05, show_me = True)

bin_img = utils.rgb_to_binary(raw_img, [1, 1, 1], 0.2)

rect_selector = RectSelector(bin_img)

rectangles = rect_selector.rects

imgs = utils.cut(bin_img, rectangles, show_me = True)

top_left = imgs[0]
top_right = imgs[1]
bottom_left = imgs[2]
bottom_right = imgs[3]

octals_top_left = operas.octalize(top_left, show_me = True)

sim_top_left_right = []
for img, ii in zip(octals_top_left, np.arange(len(octals_top_left))):
    sim, _, _ = metrics.jaccard_coef_all_trans(img, top_right)
    sim_top_left_right.append(sim)

print(sim_top_left_right)

ops = operas.decode_reflection_rotation(np.argmax(sim_top_left_right))

guess = operas.apply_operations(bottom_left, ops)

options = imgs[4:]

option_sims = []
for img in options:
    sim, _, _ = metrics.jaccard_coef_all_trans(img, guess)
    option_sims.append(sim)

print(option_sims)