"""
hint 1: convert images to binary

hint 2:  similarity metric :  Jaccard coefficient

sim(A, B) = (# pixels in A intersect B) / (# pixels in A union B)

want: similarity under all possible relative translations

sim(A, B) -> similarity value, also (x, y) offset yielding maximum similarity

"""
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import rescale
from RectSelector import RectSelector
import utils
import operas
import metrics

raw_img = rescale(image = plt.imread("./problems/X1.gif"),
                  scale = (0.25, 0.25, 1))

raw_img = utils.pad(raw_img, r =0.05, show_me = True)

bin_img = utils.to_binary(raw_img, [1, 1, 1], 0.2)

rect_selector = RectSelector(bin_img)

rectangles = rect_selector.rects

imgs = utils.cut(bin_img, rectangles, show_me = True)

top_left = imgs[0]
top_right = imgs[1]
bottom_left = imgs[2]
bottom_right = imgs[3]

# options = imgs[4:]

octals_top_left = operas.octalize(top_left, show_me = True)

sim_top_left_right = []
for img in octals_top_left:
    sim_top_left_right.append(metrics.jaccard_coef_all_trans(img, top_right))

print(sim_top_left_right)
