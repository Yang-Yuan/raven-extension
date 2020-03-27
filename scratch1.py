from matplotlib import pyplot as plt
import problem
import utils
import numpy as np
from skimage.transform import resize


problems = problem.load_problems()

prob = problems[55]

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



