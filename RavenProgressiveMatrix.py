import numpy as np

class RavenProgressiveMatrix:

    def __init__(self, matrix_components, option_components):

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
