from matplotlib import pyplot as plt
import problem
import utils
import numpy as np
from transform import rescale


problems = problem.load_problems()

d1 = problems[36]

mtrx = d1.matrix

k = 0
(mtrx[k, 0] == mtrx[k, 1]).all()
(mtrx[k, 0] == mtrx[k, 2]).all()
(mtrx[k, 1] == mtrx[k, 2]).all()
