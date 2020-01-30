import json
import numpy as np
from matplotlib import pyplot as plt
import time
from problems import load_problems
from report import create_report
import transform
import metrics

problems = load_problems()


def run_raven(mode, analogies, transformations):
    """

    :param mode: {"explanatory", "greedy"}
    :param analogies:
    :param transformations:
    :return:
    """
    if "explanatory" == mode:
        run_raven_explanatory(analogies, transformations)
    elif "greedy" == mode:
        pass
    else:
        raise Exception("Ryan!")


def run_raven_explanatory(analogies_groups, transformation_groups):
    """

    :param analogies_groups: analogies grouped by the size of matrix and whether it is unary or binary
    :param transformations:
    :return:
    """

    global problems

    start_time = time.time()

    print("run raven in explanatory mode.")

    for problem in problems:

        print(problem.name)

        metrics.load_jaccard_cache(problem.name)

        if 2 == problem.matrix_n:
            unary_analogies = analogies_groups.get("2x2_unary_analogies")
            binary_analogies = {}
            unary_transformations = transformation_groups.get("2x2_unary_transformations")
            binary_transformations = {}

        elif 3 == problem.matrix_n:
            unary_analogies = analogies_groups.get("3x3_unary_analogies")
            binary_analogies = analogies_groups.get("3x3_binary_analogies")
            unary_transformations = transformation_groups.get("3x3_unary_transformations")
            binary_transformations = transformation_groups.get("3x3_binary_transformations")
        else:
            raise Exception("Ryan!")

        unary_analogies_data = {}
        for unary_analog_name, unary_analog in unary_analogies.items():
            u1 = problem.matrix[unary_analog[0]]
            u2 = problem.matrix[unary_analog[1]]
            u3 = problem.matrix[unary_analog[2]]

            sim_u1_trans_u2 = []
            u1_u2_align_x = []
            u1_u2_align_y = []
            u1_u2_diff = []
            for unary_tran in unary_transformations:
                if str(unary_tran) == "[{'name': 'extend'}]":
                    sim, align_x, align_y, diff = metrics.asymmetric_jaccard_coef_A_sub_B(u1, u2)
                    sim_u1_trans_u2.append(sim)
                    u1_u2_align_x.append(align_x)
                    u1_u2_align_y.append(align_y)
                    u1_u2_diff.append(diff)
                elif str(unary_tran) == "[{'name': 'backward_extend'}]":
                    sim, align_x, align_y, diff = metrics.asymmetric_jaccard_coef_B_sub_A(u1, u2)
                    sim_u1_trans_u2.append(sim)
                    u1_u2_align_x.append(align_x)
                    u1_u2_align_y.append(align_y)
                    u1_u2_diff.append(diff)
                else:
                    u1_t = transform.apply_unary_transformation(u1, unary_tran)
                    sim, _, _ = metrics.jaccard_coef(u1_t, u2)
                    sim_u1_trans_u2.append(sim)
                    u1_u2_align_x.append(None)  # only the weird extend transformation needs this
                    u1_u2_align_y.append(None)  # only the weird extend transformation needs this
                    u1_u2_diff.append(None)  # only the weird extend transformation needs this

            best_trans_id = int(np.argmax(sim_u1_trans_u2))
            unary_analogies_data[unary_analog_name] = {
                "best_unary_tran": unary_transformations[best_trans_id],
                "best_sim_u1_trans_u2": sim_u1_trans_u2[best_trans_id],
                "best_u1_u2_align_x": u1_u2_align_x[best_trans_id],
                "best_u1_u2_align_y": u1_u2_align_y[best_trans_id],
                "best_u1_u2_diff": u1_u2_diff[best_trans_id]
            }

        binary_analogies_data = {}
        for binary_analog_name, binary_analog in binary_analogies.items():
            b1 = problem.matrix[binary_analog[0]]
            b2 = problem.matrix[binary_analog[1]]
            b3 = problem.matrix[binary_analog[2]]
            b4 = problem.matrix[binary_analog[3]]
            b5 = problem.matrix[binary_analog[4]]

            sim_b1_b2_trans_b3 = []
            b1_b2_align_x = []
            b1_b2_align_y = []
            for binary_tran in binary_transformations:
                b1_b2_t, align_x, align_y = transform.apply_binary_transformation(b1, b2, binary_tran)
                sim, _, _ = metrics.jaccard_coef(b1_b2_t, b3)
                sim_b1_b2_trans_b3.append(sim)
                b1_b2_align_x.append(align_x)
                b1_b2_align_y.append(align_y)

            best_trans_id = int(np.argmax(sim_b1_b2_trans_b3))
            binary_analogies_data[binary_analog_name] = {
                "best_binary_tran": binary_transformations[best_trans_id],
                "best_sim_b1_b2_trans_b3": sim_b1_b2_trans_b3[best_trans_id],
                "best_b1_b2_align_x": b1_b2_align_x[best_trans_id],
                "best_b1_b2_align_y": b1_b2_align_y[best_trans_id]
            }

        best_unary_sim = -1
        best_unary_analog_name = None
        for unary_analog_name, unary_analog_data in unary_analogies_data.items():
            if best_unary_sim < unary_analog_data.get("best_sim_u1_trans_u2"):
                best_unary_analog_name = unary_analog_name

        best_binary_sim = -1
        best_binary_analog_name = None
        for binary_analog_name, binary_analog_data in binary_analogies_data.items():
            if best_binary_sim < binary_analog_data.get("best_sim_b1_b2_trans_b3"):
                best_binary_analog_name = binary_analog_name

        if best_unary_sim > best_binary_sim:
            unary_analog_data = unary_analogies_data.get(best_unary_analog_name)
            best_unary_tran = unary_analog_data.get("best_unary_tran")
            if str(unary_tran) == "[{'name': 'extend'}]":
                pass
            elif str(unary_tran) == "[{'name': 'backward_extend'}]":
                pass
            else:
                pass
        else:
            binary_analog_data = binary_analogies_data.get(best_binary_analog_name)
            best_binary_tran = binary_analog_data.get("best_binary_tran")
            best_b1_b2_align_x = binary_analog_data.get("best_b1_b2_align_x")
            best_b1_b2_align_y = binary_analog_data.get("best_b1_b2_align_y")
            pass


        problem_data = {
            "unary_analogies_data": unary_analogies_data,
            "binary_analogies_data": binary_analogies_data
        }

        problem.data = problem_data

        with open("./data/" + problem.name + ".json", 'w+') as outfile:
            json.dump(problem_data, outfile)
            outfile.close()

        metrics.save_jaccard_cache(problem.name)

    create_report(problems)

    end_time = time.time()

    print(end_time - start_time)


def run_raven_greedy(analogies, transformations):

    global problems

    start_time = time.time()

    print("run raven in greedy mode.")

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
                sim, _, _ = metrics.jaccard_coef(u1_t, u2)
                sim_u1_trans_u2.append(sim)

            unary_tran = u1_transformations[np.argmax(sim_u1_trans_u2)]
            u4_predicted = transform.apply_unary_transformation(u3, unary_tran)

            sim_u4_predicted_ops = []
            for op in problem.options:
                sim, _, _ = metrics.jaccard_coef(op, u4_predicted)
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
                sim, _, _ = metrics.jaccard_coef(b1_b2_t, b3)
                sim_b1_b2_trans_b3.append(sim)

            b1_b2_tran = b1_b2_transformations[np.argmax(sim_b1_b2_trans_b3)]
            b6_predicted = transform.apply_binary_transformation(b4, b5,
                                                                 b1_b2_align_x, b1_b2_align_y,
                                                                 b1_b2_tran)

            sim_b6_predicted_ops = []
            for op in problem.options:
                sim, _, _ = metrics.jaccard_coef(op, b6_predicted)
                sim_b6_predicted_ops.append(sim)

            binary_analogies_data[binary_analog_name] = {
                "b1_b2_align_x": int(b1_b2_align_x),
                "b1_b2_align_y": int(b1_b2_align_y),
                "sim_b1_b2_trans_b3": sim_b1_b2_trans_b3,
                "b1_b2_tran": b1_b2_tran,
                "sim_b6_predicted_ops": sim_b6_predicted_ops,
                "argmax_sim_b6_predicted_ops": int(np.argmax(sim_b6_predicted_ops)) + 1
            }

        problem_data = {
            "unary_analogies_data": unary_analogies_data,
            "binary_analogies_data": binary_analogies_data
        }

        problem.data = problem_data

        with open("./data/" + problem.name + ".json", 'w+') as outfile:
            json.dump(problem_data, outfile)
            outfile.close()

        metrics.save_jaccard_cache(problem.name)

    create_report(problems)

    end_time = time.time()

    print(end_time - start_time)
