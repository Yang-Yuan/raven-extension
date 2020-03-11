from matplotlib import pyplot as plt
import problem
import utils
import numpy as np
from transform import rescale


problems = problem.load_problems()

c7 = problems[30]

c7.plot_problem()
