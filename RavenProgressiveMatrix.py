import numpy as np
from matplotlib import pyplot as plt


class RavenProgressiveMatrix:

    def __init__(self, name, matrix_components, option_components, answer):

        self.name = name

        if 4 == len(matrix_components):
            self.matrix_n = 2
        elif 9 == len(matrix_components):
            self.matrix_n = 3
        else:
            raise Exception("Crap!")

        self.matrix = np.empty((self.matrix_n, self.matrix_n), dtype = np.object)
        kk = 0
        for ii in np.arange(self.matrix_n):
            for jj in np.arange(self.matrix_n):
                self.matrix[ii, jj] = matrix_components[kk]
                kk = kk + 1

        self.options = option_components

        self.analogies = None
        self.data = None
        self.answer = answer

    def plot_problem(self):
        if 2 == self.matrix_n:
            fig, axs = plt.subplots(nrows = 3,
                                    ncols = len(self.options))
            fig.suptitle(self.name)
            for ax, com in zip(axs[: 2, : 2].flatten(order = 'C'), self.matrix.flatten(order = 'C')):
                ax.imshow(com, cmap = "binary")
            for ax in axs[: 2, 2:].flatten():
                ax.remove()
            for ax, com in zip(axs[2], self.options):
                ax.imshow(com, cmap = "binary")
        elif 3 == self.matrix_n:
            fig, axs = plt.subplots(nrows = 4,
                                    ncols = len(self.options))
            fig.suptitle(self.name)
            for ax, com in zip(axs[: 3, : 3].flatten(order = 'C'), self.matrix.flatten()):
                ax.imshow(com, cmap = "binary")
            for ax in axs[: 3, 3:].flatten():
                ax.remove()
            for ax, com in zip(axs[3], self.options):
                ax.imshow(com, cmap = "binary")
        else:
            raise Exception("Crap!")
        plt.show()

    def plot_solution(self):
        pass
