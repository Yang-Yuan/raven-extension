from matplotlib import pyplot as plt
import problem
import utils
import numpy as np
from transform import rescale


problems = problem.load_problems()

prob = problems[39]
prob.matrix[2, 2] = prob.options[1]

shapes = [[img.shape for img in row] for row in prob.matrix]

sps_x = np.empty_like(prob.matrix)
sps_y = np.empty_like(prob.matrix)
for ii in range(prob.matrix.shape[0]):
    for jj in range(prob.matrix.shape[1]):
        sps_x[ii, jj] = prob.matrix[ii, jj].shape[1]
        sps_y[ii, jj] = prob.matrix[ii, jj].shape[0]




