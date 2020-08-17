# from matplotlib import pyplot as plt
# from PIL import Image
import numpy as np
# from skimage.color import rgb2gray
# from skimage.io import imsave
# import problem
# import utils
# from skimage.transform import resize
# import map
# from math import sin, cos, pi
# import jaccard
import soft_jaccard
# import norm

A = np.full((5, 5), False)
A[0, 0] = True
A[1:4, 1:4] = True
B = np.full((5, 5), False)
B[4, 4] = True
B[1:4, 1:4] = True

sj, diff_to_A_x, diff_to_A_y, diff_to_B_x, diff_to_B_y, diff = soft_jaccard.soft_jaccard(A, B, asymmetric = True)
sj0, diff_to_A_x0, diff_to_A_y0, diff_to_B_x0, diff_to_B_y0, diff0 = soft_jaccard.soft_jaccard(B, A, asymmetric = True)

print("aloha")

# norm.load_norm_to_p()

# r = 10
# A = np.full((21, 21), False)
# for gama in range(360):
#     y = 10 + r * sin(pi / 180 * gama)
#     x = 10 + r * cos(pi / 180 * gama)
#     if y - int(y) > 0.5:
#         y = int(y) + 1
#     else:
#         y = int(y)
#     if x - int(x) > 0.5:
#         x = int(x) + 1
#     else:
#         x = int(x)
#     A[y, x] = True
#
#
# r = 9
# B = np.full((21, 21), False)
# for gama in range(360):
#     y = 10 + r * sin(pi / 180 * gama)
#     x = 10 + r * cos(pi / 180 * gama)
#     if y - int(y) > 0.5:
#         y = int(y) + 1
#     else:
#         y = int(y)
#     if x - int(x) > 0.5:
#         x = int(x) + 1
#     else:
#         x = int(x)
#     B[y, x] = True
#
#
# r = 10
# C = np.full((21, 21), False)
# for gama in range(181):
#     y = 10 + r * sin(pi / 180 * gama)
#     x = 10 + r * cos(pi / 180 * gama)
#     if y - int(y) > 0.5:
#         y = int(y) + 1
#     else:
#         y = int(y)
#     if x - int(x) > 0.5:
#         x = int(x) + 1
#     else:
#         x = int(x)
#     C[y, x] = True
# C[10, :] = True
#
# j_AB, j_A_to_B_x, j_A_to_B_y = jaccard.jaccard_coef_naive(A, B)
# j_AC, j_A_to_C_x, j_A_to_C_y = jaccard.jaccard_coef_naive(A, C)
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# sj_AC, A_to_C_x, A_to_C_y = soft_jaccard.soft_jaccard(A, C)
# sj_CA, C_to_A_x, C_to_A_y = soft_jaccard.soft_jaccard(C, A)
#
# print("llohe")


# A = np.full((20, 20), False)
# A[0, :] = True
# A[:, 0] = True
# A[19, :] = True
# A[:, 19] = True
# B = np.full((20, 20), False)
# B[0, :] = True
# B[:, 19] = True
# for ii in range(20):
#     B[ii, ii] = True
# C = np.full((20, 20), False)
# C[0, 0 : 19] = True
# C[0 : 19, 0] = True
# C[18, 0 : 19] = True
# C[0 : 19, 18] = True
#
# j_AB = np.logical_and(A, B).sum() / np.logical_or(A, B).sum()
# j_AC = np.logical_and(A, C).sum() / np.logical_or(A, C).sum()
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# sj_AC, A_to_C_x, A_to_C_y = soft_jaccard.soft_jaccard(A, C)
# sj_CA, C_to_A_x, C_to_A_y = soft_jaccard.soft_jaccard(C, A)
#
# print("llohe")


# A = np.full((1, 1), False)
# A[0, 0] = True
# B = np.full((3, 3), False)
# B[1, 1] = True
# B[2, 2] = True
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("llohe")

# A = np.full((3, 3), False)
# A[0, 2] = True
# A[0, 0] = True
# A[1, 1] = True
# A[2, 2] = True
# B = np.full((3, 3), False)
# B[2, 0] = True
# B[0, 0] = True
# B[1, 1] = True
# B[2, 2] = True
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("aloha")

# A = np.full((3, 3), False)
# A[1, :] = True
# A[2, 1] = True
# B = np.full((3, 3), False)
# B[1, :] = True
# B[0, 1] = True
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("aloha")

# A = np.full((3, 3), False)
# A[1, 1] = True
# A[2, 0] = True
# A[2, 1] = True
# B = np.full((3, 3), False)
# B[0, 1] = True
# B[0, 2] = True
# B[1, 1] = True
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("aloha")

# A = np.full((3, 3), False)
# A[1, 0] = True
# A[1, 2] = True
# B = np.full((3, 3), False)
# B[0, 1] = True
# B[2, 1] = True
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("aloha")

# A = np.full((3, 3), True)
# B = np.full((2, 2), True)
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("aloha")

# A = np.full((3, 3), True)
# B = np.full((3, 3), True)
#
# sj_AB, A_to_B_x, A_to_B_y = soft_jaccard.soft_jaccard(A, B)
#
# sj_BA, B_to_A_x, B_to_A_y = soft_jaccard.soft_jaccard(B, A)
#
# print("aloha")

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



