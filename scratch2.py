import numpy as np

x = [(0,0), (1, 1), (2, 2)]

y = np.array([1, 0, 1])

for [ii, jj] in np.array(x)[np.where(y == 1)]:
    print(ii, jj)

x = {}
for a, b in x.items():
    print(a, b)
