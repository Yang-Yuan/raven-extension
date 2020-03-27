from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import rescale
from skimage.filters import gaussian
import numpy as np
import utils
from RavenProgressiveMatrix import RavenProgressiveMatrix

raven_folder = "./problems/SPMpadded"
raven_coordinates_file = "./problems/SPM coordinates.txt"


def load_problems(problem_folder = None, problem_coordinates_file = None, show_me = False):
    global raven_folder
    global raven_coordinates_file

    if problem_folder is None:
        problem_folder = raven_folder

    if problem_coordinates_file is None:
        problem_coordinates_file = raven_coordinates_file

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

    with open("./problems/stuff.txt", "r") as f:
        answers = []
        for line in f:
            answers.append(int(line))

    problems = []
    for img, coords, problem_name, answer in zip(raw_images, problem_coordinates, problem_names, answers):
        coms = utils.extract_components(img, coords)

        smaller_coms = [rescale(image = com, scale = (0.5, 0.5)) for com in coms]

        binary_smaller_coms = [utils.grey_to_binary(com, 0.7) for com in smaller_coms]

        binary_smaller_coms = [utils.erase_noise_point(com, 4) for com in binary_smaller_coms]

        # let's try it first. Although somebody doesn't agree on this, but I think it won't hurt the alignment.
        # because even if the alignment is not what it is in the original images, it will still give the correct
        # answer.
        # and it will definitely accelerate the computation.
        binary_smaller_coms = [utils.trim_binary_image(com) for com in binary_smaller_coms]

        # some blatant magic code, because of the blatant hand drawing
        # if problem_name in ["c7"]:
        #     binary_smaller_coms = utils.resize_to_average_shape(binary_smaller_coms, ignore = [8])

        if 10 == len(coms):
            matrix = utils.create_object_matrix(binary_smaller_coms[: 4], (2, 2))
            options = binary_smaller_coms[4:]
            problems.append(RavenProgressiveMatrix(problem_name, matrix, options, answer))
        elif 17 == len(coms):
            matrix = utils.create_object_matrix(binary_smaller_coms[: 9], (3, 3))
            options = binary_smaller_coms[9:]
            problems.append(RavenProgressiveMatrix(problem_name, matrix, options, answer))
        else:
            raise Exception("Crap!")

    if show_me:
        for prob in problems:
            prob.plot_problem()

    return problems

