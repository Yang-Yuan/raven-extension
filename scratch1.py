from matplotlib import pyplot as plt
import problem
import utils
import numpy as np
from transform import rescale


problems = problem.load_problems()

c2 = problems[25]

shapes = np.empty_like(c2.matrix, dtype = tuple)
for ii in range(3):
    for jj in range(3):
        shapes[ii, jj] = c2.matrix[ii, jj].shape

tmp = rescale(c2.matrix[1, 1], 1.3, 1.4)
plt.figure()
plt.imshow(tmp)
plt.figure()
plt.imshow(c2.matrix[1, 2])
plt.show()


print(shapes)

