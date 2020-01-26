import json
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from problems import load_problems
from analogy import get_analogies
import transform
import metrics

raven_folder = "./problems/SPMpadded"
raven_coordinates_file = "./problems/SPM coordinates.txt"
cache_folder = "./precomputed-similarities/jaccard"

problems = load_problems(raven_folder, raven_coordinates_file)

for problem in problems:

    print(problem.name)

    cache_file_name = join(cache_folder, problem.name + ".npz")

    try:
        cache_file = np.load(cache_file_name, allow_pickle = True)
    except FileNotFoundError:
        cache_file = None

    if cache_file is not None:
        similarities = cache_file["similarities"]
        cache_file.files.remove("similarities")

        images = []
        for img_name in cache_file.files:
            images.append(cache_file[img_name])
    else:
        similarities = np.full((5, 5), -1)
        images = []

    similarities = np.full((3, 3), None, dtype = object)
    similarities[0, 0] = np.arange(3)
    for ii in range(len(problem.options)):
        images.append(np.packbits(problem.options[ii], axis = -1))
    np.savez(cache_file_name, similarities = similarities, *images)



