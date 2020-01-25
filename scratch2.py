import json
import numpy as np
from matplotlib import pyplot as plt
from problems import load_problems
from analogy import get_analogies
import transform
import metrics

raven_folder = "./problems/SPMpadded"
raven_coordinates_file = "./problems/SPM coordinates.txt"

problems = load_problems(raven_folder,
                         raven_coordinates_file)

for problem in problems:

    print(problem.name)

    groupA = problem.matrix.flatten()
    groupB = problem.options
    for ii in range(len(groupA)):
        for jj in range(len(groupB)):

            print(ii, jj)

            sim_naive, x_naive, y_naive = metrics.jaccard_coef0(groupA[ii], groupB[jj])
            sim_fast, x_fast, y_fast = metrics.jaccard_coef2(groupA[ii], groupB[jj])

            if sim_naive != sim_fast:
                print(ii, jj, problem.name)
                raise Exception("Crap!")

            if x_naive != x_fast:
                print(ii, jj, problem.name)
                raise Exception("Crap!")

            if y_naive != y_fast:
                print(ii, jj, problem.name)
                raise Exception("Crap!")

print("Yeah!")


