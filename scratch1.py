import json
import numpy as np
from matplotlib import pyplot as plt
from problems import load_problems
from analogy import get_analogies
import transform
import metrics

raven_folder = "./problems/SPMpadded"
raven_coordinates_file = "./problems/SPM coordinates.txt"

problems = load_problems(raven_folder,
                         raven_coordinates_file)

for problem in problems:

    print(problem.name)

    metrics.load_jaccard_cache(problem.name)

    analogies = get_analogies(problem)

    unary_analogies = analogies.get("unary_analogies")
    unary_analogies_data = {}
    for unary_analog_name, unary_analog in unary_analogies.items():
        u1 = problem.matrix[unary_analog[0]]
        u2 = problem.matrix[unary_analog[1]]
        u3 = problem.matrix[unary_analog[2]]

        u1_trans, u1_transformations = transform.unary_transform(u1)
        sim_u1_trans_u2 = []
        for u1_t in u1_trans:
            sim, _, _ = metrics.jaccard_coef_naive(u1_t, u2)
            sim_u1_trans_u2.append(sim)

        unary_tran = u1_transformations[np.argmax(sim_u1_trans_u2)]
        u4_predicted = transform.apply_unary_transformation(u3, unary_tran)

        sim_u4_predicted_ops = []
        for op in problem.options:
            sim, _, _ = metrics.jaccard_coef_naive(op, u4_predicted)
            sim_u4_predicted_ops.append(sim)

        unary_analogies_data[unary_analog_name] = {
            "sim_u1_trans_u2": sim_u1_trans_u2,
            "unary_tran": unary_tran,
            "sim_u4_predicted_ops": sim_u4_predicted_ops,
            "argmax_sim_u4_predicted_ops": int(np.argmax(sim_u4_predicted_ops)) + 1
        }

    binary_analogies = analogies.get("binary_analogies")
    binary_analogies_data = {}
    for binary_analog_name, binary_analog in binary_analogies.items():
        b1 = problem.matrix[binary_analog[0]]
        b2 = problem.matrix[binary_analog[1]]
        b3 = problem.matrix[binary_analog[2]]
        b4 = problem.matrix[binary_analog[3]]
        b5 = problem.matrix[binary_analog[4]]

        b1_b2_trans, b1_b2_transformations, b1_b2_align_x, b1_b2_align_y = transform.binary_transform(b1, b2)
        sim_b1_b2_trans_b3 = []
        for b1_b2_t in b1_b2_trans:
            sim, _, _ = metrics.jaccard_coef_naive(b1_b2_t, b3)
            sim_b1_b2_trans_b3.append(sim)

        b1_b2_tran = b1_b2_transformations[np.argmax(sim_b1_b2_trans_b3)]
        b6_predicted = transform.apply_binary_transformation(b4, b5,
                                                             b1_b2_align_x, b1_b2_align_y,
                                                             b1_b2_tran)

        sim_b6_predicted_ops = []
        for op in problem.options:
            sim, _, _ = metrics.jaccard_coef_naive(op, b6_predicted)
            sim_b6_predicted_ops.append(sim)

        binary_analogies_data[binary_analog_name] = {
            "b1_b2_align_x" : int(b1_b2_align_x),
            "b1_b2_align_y" : int(b1_b2_align_y),
            "sim_b1_b2_trans_b3" : sim_b1_b2_trans_b3,
            "b1_b2_tran": b1_b2_tran,
            "sim_b6_predicted_ops": sim_b6_predicted_ops,
            "argmax_sim_b6_predicted_ops": int(np.argmax(sim_b6_predicted_ops)) + 1
        }

    problem_data = {
        "unary_analogies_data" : unary_analogies_data,
        "binary_analogies_data" : binary_analogies_data
    }

    with open("./data/" + problem.name + ".json", 'w+') as outfile:
        json.dump(problem_data, outfile)
        outfile.close()

    metrics.save_jaccard_cache(problem.name)
