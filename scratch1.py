from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imsave
import problem
import utils
from skimage.transform import resize
import map
import soft_jaccard


A = np.full((3, 3), False)
A[1, 0] = True
A[1, 2] = True
B = np.full((3, 3), False)
B[0, 1] = True
B[2, 1] = True

sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)

# A = np.full((3, 3), True)
# B = np.full((2, 2), True)
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)

# A = np.full((3, 3), True)
# B = np.full((3, 3), True)
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
#
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)

print("llohe")
# img = plt.imread("./problems/ace analogies - chopped up/m5/c.gif")
# img0 = img[:, :, 0]
# c0 = Image.fromarray(img0)
# c0.save("./problems/ace analogies - chopped up/m5/c0.gif", mode = "L")

# img = plt.imread("./problems/g1.png")
# img0 = rgb2gray(img)
# imsave("./problems/g1.png", img0)


# problems = problem.load_problems()
#
# prob = problems[55]

# a = prob.matrix[1, 1]
# b = prob.matrix[1, 2]
#
# c = utils.grey_to_binary(resize(np.logical_not(a), b.shape, order = 0), 0.7)
#
# d1 = utils.grey_to_binary(resize(np.logical_not(a), (b.shape[0], a.shape[1]), order = 0), 0.7)
# d2 = utils.grey_to_binary(resize(np.logical_not(a), (d1.shape[0], b.shape[1]), order = 0), 0.7)

# a11 = problems[10]
#
# u1 = a11.matrix[0, 0]
# u2 = a11.matrix[0, 1]
# u3 = a11.matrix[1, 0]
#
# x = np.logical_xor(u1, u2)
#
# u11 = np.logical_xor(u2, x)
#
# plt.imshow(u11)

# prob.matrix[2, 2] = prob.options[1]
#
# shapes = [[img.shape for img in row] for row in prob.matrix]
#
# sps_x = np.empty_like(prob.matrix)
# sps_y = np.empty_like(prob.matrix)
# for ii in range(prob.matrix.shape[0]):
#     for jj in range(prob.matrix.shape[1]):
#         sps_x[ii, jj] = prob.matrix[ii, jj].shape[1]
#         sps_y[ii, jj] = prob.matrix[ii, jj].shape[0]
#



