from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import rescale
import numpy as np
import utils
from RavenProgressiveMatrix import RavenProgressiveMatrix


def load_problems(problem_folder, problem_coordinates_file, show_me = False):

    problem_filenames = natsorted([f for f in listdir(problem_folder) if isfile(join(problem_folder, f))])

    problem_names = [f.split('.')[0] for f in problem_filenames]

    problem_paths = [join(problem_folder, f) for f in problem_filenames]

    raw_images = [plt.imread(path) for path in problem_paths]

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
    for img, coords, problem_name in zip(raw_images, problem_coordinates, problem_names):
        coms = utils.extract_components(img, coords)
        smaller_coms = [rescale(image = com, scale = (0.5, 0.5)) for com in coms]
        binary_smaller_coms = [utils.grey_to_binary(com, 1, 0.2) for com in smaller_coms]

        # shouldn't trim components. It's gonna mess up the alignment.
        # trimmed_coms = [utils.trim_binary_image(com) for com in coms]

        if 10 == len(coms):
            problems.append(RavenProgressiveMatrix(problem_name, binary_smaller_coms[: 4], binary_smaller_coms[4:]))
        elif 17 == len(coms):
            problems.append(RavenProgressiveMatrix(problem_name, binary_smaller_coms[: 9], binary_smaller_coms[9:]))
        else:
            raise Exception("Crap!")

    if show_me:
        for prob in problems:
            prob.plot_problem()

    return problems
