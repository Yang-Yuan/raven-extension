from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import rescale
import numpy as np
import utils
from RavenProgressiveMatrix import RavenProgressiveMatrix


def load_problems(problem_folder, problem_coordinates_file, show_me = False):
    problem_paths = natsorted([join(problem_folder, f)
                               for f in listdir(problem_folder) if isfile(join(problem_folder, f))])

    raw_images = [plt.imread(path) for path in problem_paths]

    binary_images = [utils.grey_to_binary(img, 1, 0.2) for img in raw_images]

    with open(problem_coordinates_file) as f:
        problem_coordinates = []
        coordinates = []
        for line in f:
            rect_coords = [int(x) for x in line.split()]
            if 0 == len(rect_coords):
                problem_coordinates.append(coordinates)
                coordinates = []
            else:
                coordinates.append(rect_coords)

    problems = []
    for img, coords, problem_path in zip(binary_images, problem_coordinates, problem_paths):
        coms = utils.extract_components(img, coords)
        trimmed_coms = [utils.trim_binary_image(com) for com in coms]
        if 10 == len(coms):
            problems.append(RavenProgressiveMatrix(problem_path, coms[: 4], coms[4:]))
        elif 17 == len(coms):
            problems.append(RavenProgressiveMatrix(problem_path, coms[: 9], coms[9:]))
        else:
            raise Exception("Crap!")

    if show_me:
        for prob in problems:
            prob.plot_problem()

    return problems
