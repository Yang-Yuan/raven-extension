import numpy as np
from matplotlib import pyplot as plt
import utils


class RavenProgressiveMatrix:

    def __init__(self, name, matrix, options, answer):

        self.name = name
        self.type = str(matrix.shape[0]) + "x" + str(matrix.shape[1])
        self.matrix = matrix
        self.matrix_ref = np.empty_like(self.matrix, dtype = np.object)
        if self.is_square_problem():
            for ii in range(self.matrix_ref.shape[0]):
                for jj in range(self.matrix_ref.shape[1]):
                    self.matrix_ref[ii, jj] = utils.fill_holes(self.matrix[ii, jj])
        else:
            self.matrix_ref = matrix
        self.options = options
        self.answer = answer

    def plot_problem(self):

        fig, axs = plt.subplots(nrows = self.matrix.shape[0] + 1,
                                ncols = max(self.matrix.shape[1], len(self.options)))
        fig.suptitle(self.name)

        for ii, axs_row in enumerate(axs):
            for jj, ax in enumerate(axs_row):
                if ii < self.matrix.shape[0]:
                    if jj < self.matrix.shape[1]:
                        ax.imshow(self.matrix[ii, jj], cmap = "binary")
                    else:
                        ax.remove()
                else:
                    if jj < len(self.options):
                        ax.imshow(self.options[jj], cmap = "binary")
                    else:
                        ax.remove()

        plt.show()

    # blatant magic code, because the defect of hand drawing
    def is_square_problem(self):

        square_problems = ["c12"]
        for prob_name in square_problems:
            if prob_name in self.name:
                return True

        return False

    def plot_solution(self):
        pass
