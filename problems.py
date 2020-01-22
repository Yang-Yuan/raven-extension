from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import rescale
import numpy as np
import utils
from RavenProgressiveMatrix import RavenProgressiveMatrix

problem_folder = ".\\problems\\SPMpadded"
problem_coordinates_file = ".\\problems\\SPM coordinates.txt"

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
for img, coords in zip(binary_images, problem_coordinates):
    coms = utils.extract_components(img, coords)
    if 10 == len(coms):
        problems.append(RavenProgressiveMatrix(coms[: 4], coms[4:]))
    elif 17 == len(coms):
        problems.append(RavenProgressiveMatrix(coms[: 9], coms[9:]))
    else:
        raise Exception("Crap!")

for prob in problems:
    if 4 == len(prob.matrix_components):
        fig, axs = plt.subplots(nrows = 3,
                                ncols = len(prob.option_components))
        for ax, com in zip(axs[: 2, : 2].flatten(order = 'C'), prob.matrix_components):
            ax.imshow(com, cmap = "binary")
        for ax in axs[: 2, 2:].flatten():
            ax.remove()
        for ax, com in zip(axs[2], prob.option_components):
            ax.imshow(com, cmap = "binary")
    elif 9 == len(prob.matrix_components):
        fig, axs = plt.subplots(nrows = 4,
                                ncols = len(prob.option_components))
        for ax, com in zip(axs[: 3, : 3].flatten(order = 'C'), prob.matrix_components):
            ax.imshow(com, cmap = "binary")
        for ax in axs[: 3, 3 :].flatten():
            ax.remove()
        for ax, com in zip(axs[3], prob.option_components):
            ax.imshow(com, cmap = "binary")
    else:
        raise Exception("Crap!")
    plt.show()




