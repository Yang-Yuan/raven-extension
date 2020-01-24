import numpy as np
from problems import load_problems
from analogy import get_analogies
import transform
import metrics

raven_folder = ".\\problems\\SPMpadded"
raven_coordinates_file = ".\\problems\\SPM coordinates.txt"

problems = load_problems(raven_folder,
                         raven_coordinates_file)

problem = problems[0]

analogies = get_analogies(problem)

unary_analogies = analogies.get("unary_analogies")
for unary_analog in unary_analogies:
    u1 = problem.matrix[unary_analog[0]]
    u2 = problem.matrix[unary_analog[1]]
    u3 = problem.matrix[unary_analog[2]]

    u1_trans, u1_transformations = transform.unary_transform(u1)
    sim_u1_trans_u2 = []
    for u1_t in u1_trans:
        sim, _, _ = metrics.jaccard_coef_shift_invariant(u1_t, u2)
        sim_u1_trans_u2.append(sim)

    unary_tran = u1_transformations[np.argmax(sim_u1_trans_u2)]
    u4_predicted = transform.apply_unary_transformation(u3, unary_tran)

    sim_u4_predicted_ops = []
    for op in problem.options:
        sim, _, _ = metrics.jaccard_coef_shift_invariant(op, u4_predicted)
        sim_u4_predicted_ops.append(sim)


binary_analogies = analogies.get("binary_analogies")
for binary_analog in binary_analogies:
    b1 = problem.matrix[binary_analog[0]]
    b2 = problem.matrix[binary_analog[1]]
    b3 = problem.matrix[binary_analog[2]]
    b4 = problem.matrix[binary_analog[3]]
    b5 = problem.matrix[binary_analog[4]]

    b1_b2_trans, b1_b2_transformations, b1_b2_align_x, b1_b2_align_y = transform.binary_transform(b1, b2, show_me = True)
    sim_b1_b2_trans_b3 = []
    for b1_b2_t in b1_b2_trans:
        sim, _, _ = metrics.jaccard_coef_shift_invariant(b1_b2_t, b3)
        sim_b1_b2_trans_b3.append(sim)

    b1_b2_tran = b1_b2_transformations[np.argmax(sim_b1_b2_trans_b3)]
    b6_predicted = transform.apply_binary_transformation(b4, b5,
                                                         b1_b2_align_x, b1_b2_align_y,
                                                         b1_b2_transformations)

    sim_b6_predicted_ops = []
    for op in problem.options:
        sim, _, _ = metrics.jaccard_coef_shift_invariant(op, b6_predicted)
        sim_b6_predicted_ops.append(sim)



