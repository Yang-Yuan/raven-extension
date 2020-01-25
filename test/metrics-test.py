import metrics
import numpy as np
from matplotlib import pyplot as plt


A = np.full((5, 5), False)
A[: 3, : 3] = True
plt.imshow(A)

B = np.full((5, 5), False)
B[2 :, 2 :] = True
plt.imshow(B)

sim, x, y = metrics.jaccard_coef_naive(A, B)

A = np.full((5, 5), False)
A[0, 2] = True
A[1, 1 : 4] = True
plt.imshow(A)

B = np.full((5, 5), False)
B[3, 1] = True
B[4, :3] = True
plt.imshow(B)

sim, x, y = metrics.jaccard_coef_naive(A, B)

A = np.full((5, 5), False)
A[0, 2] = True
A[1, 1 : 4] = True
plt.imshow(A)

B = np.full((5, 5), False)
B[0, :3] = True
plt.imshow(B)

sim, x, y = metrics.jaccard_coef_naive(A, B)