import json
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from problems import load_problems
from analogy import get_analogies
import transform
import metrics
import time

start = time.time()

raven_folder = "./problems/SPMpadded"
raven_coordinates_file = "./problems/SPM coordinates.txt"
cache_folder = "./precomputed-similarities/jaccard"

problems = load_problems(raven_folder, raven_coordinates_file)

end = time.time()

print(end - start)
