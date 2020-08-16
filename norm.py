from os.path import join
import numpy as np

norm_folder = "./precomputed-similarities/norm"
norm_name = "one_norm"

p = 3

A_x_max = 100
A_y_max = 100
B_x_max = 100
B_y_max = 100

norm_to_p = None

# infinite_norm_to_p = np.array([[[[max(abs(A_x - B_x), abs(A_y - B_y)) ** p # at least of infinite order to differentiate shapes
#                              for A_x in range(B_y_max)]
#                             for A_y in range(B_x_max)]
#                            for B_x in range(A_y_max)]
#                           for B_y in range(A_x_max)])


# two_norm_to_p = np.array([[[[((A_x - B_x) ** 2 + (A_y - B_y) ** 2) ** (p/2) # at least of order 4 to differentiate shapes
#                              for A_x in range(B_y_max)]
#                             for A_y in range(B_x_max)]
#                            for B_x in range(A_y_max)]
#                           for B_y in range(A_x_max)])


def load_norm_to_p():

    global norm_to_p

    norm_file_name = join(norm_folder, norm_name + "_to_p.npz")

    try:
        norm_file = np.load(norm_file_name, allow_pickle = True)
    except FileNotFoundError:
        norm_file = None

    if norm_file is not None:
        norm_to_p = norm_file["norm_to_p"]
    else:
        if "one_norm" == norm_name:
            norm_to_p = np.array(
                [[[[(abs(A_x - B_x) + abs(A_y - B_y)) ** p  # at least of order 3 to differentiate shapes
                    for A_x in range(B_y_max)]
                   for A_y in range(B_x_max)]
                  for B_x in range(A_y_max)]
                 for B_y in range(A_x_max)])
            np.savez(norm_file_name, norm_to_p = norm_to_p)


def get_distance_matrix(A_coords, B_coords):

    global norm_to_p

    A_x = np.full((len(A_coords), len(B_coords)), A_coords[:, [0]])
    A_y = np.full((len(A_coords), len(B_coords)), A_coords[:, [1]])
    B_x = np.full((len(A_coords), len(B_coords)), B_coords[:, 0])
    B_y = np.full((len(A_coords), len(B_coords)), B_coords[:, 1])

    return norm_to_p[A_x, A_y, B_x, B_y]
