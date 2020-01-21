from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
import utils

problem_folder = ".\\problems\\SPMpadded"
problem_coodrinates_file = ".\\problems\\SPD coordinates.txt"

problem_paths = [join(problem_folder, f)
                 for f in listdir(problem_folder) if isfile(join(problem_folder, f))]

raw_images = [plt.imread(path) for path in problem_paths]

for img in raw_images:
    plt.imshow(img)
    plt.show()



# binary_images = [utils.to_binary(img, [1, 1, 1], 0.1) for img in raw_images]
