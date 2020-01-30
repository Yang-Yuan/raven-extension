
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import rescale
from RectSelector import RectSelector
import utils
import transform
import jaccard

# preprocessing
raw_img = rescale(image = plt.imread("./problems/X1.gif"),
                  scale = (0.25, 0.25, 1))

raw_img = utils.pad(raw_img, r = 0.05, show_me = True)

bin_img = utils.rgb_to_binary(raw_img, [1, 1, 1], 0.2)

rect_selector = RectSelector(bin_img)

rectangles = rect_selector.rects

imgs = utils.cut(bin_img, rectangles, show_me = True)

u1 = imgs[0]
u2 = imgs[1]
u3 = imgs[2]
u4 = imgs[3]

u1_trans, u1_transformations = transform.unary_transform(u1, show_me = True)

sim_u1_u2 = []
for img, ii in zip(u1_trans, np.arange(len(u1_trans))):
    sim, _, _ = jaccard.jaccard_coef_naive(img, u2)
    sim_u1_u2.append(sim)

print("similarities between u1_trans and u2: " + str(sim_u1_u2))

unary_trans = u1_transformations[np.argmax(sim_u1_u2)]

print("apply transformation: " + str(unary_trans))

guess = transform.apply_unary_transformation(u3, unary_trans)

plt.imshow(guess)
plt.show()

options = imgs[4:]

option_sims = []
for img in options:
    sim, _, _ = jaccard.jaccard_coef_naive(img, guess)
    option_sims.append(sim)

print("similarties between options and guess: " + str(option_sims))
