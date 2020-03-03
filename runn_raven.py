import json
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import report_explanatory
import report_greedy
import report_brutal
import analogy
import transform
import jaccard
import asymmetric_jaccard
import prob_anlg_tran


def run_raven_explanatory(show_me = False, test_problems = None):
    """

    :param test_problems:
    :param show_me:
    :return:
    """

    start_time = time.time()

    print("run raven in explanatory mode.")

    probs = prob_anlg_tran.get_probs(test_problems)

    # for each problem
    for prob in probs:

        print(prob.name)

        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        anlgs = prob_anlg_tran.get_anlgs(prob)

        tran_data = []
        anlg_data = []
        for anlg in anlgs:
            # score all transformations given an analogy
            anlg_tran_data = run_prob_anlg(prob, anlg)
            tran_data.extend(anlg_tran_data)

            # optimize w.r.t. transformations for this analogy
            anlg_tran_d = find_best(anlg_tran_data, "pat_score")
            anlg_data.append(anlg_tran_d)

        # optimize w.r.t analogies for a problem
        anlg_d = find_best(anlg_data, "pat_score")

        # predict for the problem
        pred_data = predict(prob, anlg_d)

        # optimize w.r.t. options
        pred_d = find_best(pred_data, "pato_score")

        # imaging
        save_image(prob, pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], show_me)

        # data aggregation progression, TODO maybe save them as images
        for d in tran_data:
            del d["diff"]
        for d in anlg_data:
            del d["diff"]
        for d in pred_data:
            del d["diff"]
            del d["pred"]
        del pred_d["diff"]
        del pred_d["pred"]
        aggregation_progression = {
            "tran_data": tran_data,
            "anlg_data": anlg_data,
            "pred_data": pred_data,
            "pred_d": pred_d
        }
        with open("./data/explanatory_" + prob.name + ".json", 'w+') as outfile:
            json.dump(aggregation_progression, outfile)
            outfile.close()

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

        # output report
        # prob.data = aggregation_progression
        # report_explanatory.create_report_explanatory_mode(aggregation_progression)

    end_time = time.time()
    print(end_time - start_time)


def run_raven_greedy(show_me = False, test_problems = None):

    start_time = time.time()

    print("run raven in greedy mode.")

    probs = prob_anlg_tran.get_probs(test_problems)

    for prob in probs:

        print(prob.name)

        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        anlgs = prob_anlg_tran.get_anlgs(prob)

        tran_data = []
        anlg_data = []
        for anlg in anlgs:
            # score all transformations given an analogy
            anlg_tran_data = run_prob_anlg(prob, anlg)
            tran_data.extend(anlg_tran_data)

            # optimize w.r.t. transformations for this analogy
            anlg_tran_d = find_best(anlg_tran_data, "pat_score")
            anlg_data.append(anlg_tran_d)

        pred_data = []
        for anlg_d in anlg_data:
            # predict for an analogy, and score all options with the prediction
            anlg_pred_data = predict(prob, anlg_d)

            # optimize w.r.t. options for this analogy
            pred_d = find_best(anlg_pred_data, "pato_score")
            pred_data.append(pred_d)

        # optimize w.r.t. analogies
        pred_d = find_best(pred_data, "pato_score")

        # imaging
        save_image(prob, pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], show_me)

        # data aggregation progression, TODO maybe save them as images
        for d in tran_data:
            del d["diff"]
        for d in anlg_data:
            del d["diff"]
        for d in pred_data:
            del d["diff"]
            del d["pred"]
        del pred_d["diff"]
        del pred_d["pred"]
        aggregation_progression = {
            "tran_data": tran_data,
            "anlg_data": anlg_data,
            "pred_data": pred_data,
            "pred_d": pred_d
        }
        with open("./data/explanatory_" + prob.name + ".json", 'w+') as outfile:
            json.dump(aggregation_progression, outfile)
            outfile.close()

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

    end_time = time.time()
    print(end_time - start_time)



def run_rave_brutal(analogy_groups, transformation_groups, show_me = False, test_problems = None):
    global problems

    start_time = time.time()

    print("run raven in brutal mode.")

    for problem in problems:

        if test_problems is not None and problem.name not in test_problems:
            continue

        print(problem.name)

        jaccard.load_jaccard_cache(problem.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(problem.name)

        unary_analogies, binary_analogies, unary_transformations, binary_transformations = get_anlgs_trans(
            problem, analogy_groups, transformation_groups)

        unary_analogies_data = {}
        for unary_analog_name, unary_analog in unary_analogies.items():
            u1 = problem.matrix[unary_analog[0]]
            u2 = problem.matrix[unary_analog[1]]
            u3 = problem.matrix[unary_analog[2]]

            unary_analogies_data[unary_analog_name] = {}
            for unary_tran in unary_transformations:
                if str(unary_tran) == "[{'name': 'add_diff'}]":
                    _, align_x, align_y, diff, diff_is_positive = asymmetric_jaccard.asymmetric_jaccard_coef(u1, u2)
                    u4_predicted = transform.add_diff(u3, align_x, align_y, diff, diff_is_positive)
                    sim_u4_predicted_ops = []
                    for op in problem.options:
                        sim, _, _ = jaccard.jaccard_coef(op, u4_predicted)
                        sim_u4_predicted_ops.append(sim)
                else:
                    u4_predicted = transform.apply_unary_transformation(u3, unary_tran)
                    sim_u4_predicted_ops = []
                    for op in problem.options:
                        sim, _, _ = jaccard.jaccard_coef(op, u4_predicted)
                        sim_u4_predicted_ops.append(sim)
                unary_analogies_data[unary_analog_name][str(unary_tran)] = {
                    "sim_u4_predicted_ops": sim_u4_predicted_ops,
                    "max_sim_u4_predicted_ops": max(sim_u4_predicted_ops),
                    "argmax_sim_u4_predicted_ops": int(np.argmax(sim_u4_predicted_ops)) + 1,
                    "u4_predicted": u4_predicted
                }

        binary_analogies_data = {}
        for binary_analog_name, binary_analog in binary_analogies.items():
            b1 = problem.matrix[binary_analog[0]]
            b2 = problem.matrix[binary_analog[1]]
            b4 = problem.matrix[binary_analog[3]]
            b5 = problem.matrix[binary_analog[4]]

            binary_analogies_data[binary_analog_name] = {}
            for binary_tran in binary_transformations:
                _, align_x, align_y = transform.apply_binary_transformation(b1, b2, binary_tran)
                b6_predicted, _, _ = transform.apply_binary_transformation(b4, b5, binary_tran,
                                                                           align_x, align_y)
                sim_b6_predicted_ops = []
                for op in problem.options:
                    sim, _, _ = jaccard.jaccard_coef(op, b6_predicted)
                    sim_b6_predicted_ops.append(sim)
                binary_analogies_data[binary_analog_name][str(binary_tran)] = {
                    "sim_b6_predicted_ops": sim_b6_predicted_ops,
                    "max_sim_b6_predicted_ops": max(sim_b6_predicted_ops),
                    "argmax_sim_b6_predicted_ops": int(np.argmax(sim_b6_predicted_ops)) + 1,
                    "b6_predicted": b6_predicted
                }

        best_sim = None
        best_analog_name = None
        best_analog_type = None
        best_tran = None
        sim_predicted = -1
        argmax_sim_predicted_ops = None
        predicted = None
        for anlg_name, anlg in unary_analogies_data.items():
            for tran_name, tran in anlg.items():
                max_sim_u4_predicted_ops = tran.get("max_sim_u4_predicted_ops")
                if sim_predicted < max_sim_u4_predicted_ops:
                    best_analog_name = anlg_name
                    best_analog_type = "unary"
                    best_tran = tran_name
                    sim_predicted = max_sim_u4_predicted_ops
                    argmax_sim_predicted_ops = tran.get("argmax_sim_u4_predicted_ops")
                    predicted = tran.get("u4_predicted")
                del tran["u4_predicted"]

        for anlg_name, anlg in binary_analogies_data.items():
            for tran_name, tran in anlg.items():
                max_sim_b6_predicted_ops = tran.get("max_sim_b6_predicted_ops")
                if sim_predicted < max_sim_b6_predicted_ops:
                    best_analog_name = anlg_name
                    best_analog_type = "binary"
                    best_tran = tran_name
                    sim_predicted = max_sim_b6_predicted_ops
                    argmax_sim_predicted_ops = tran.get("argmax_sim_b6_predicted_ops")
                    predicted = tran.get("b6_predicted")
                del tran["b6_predicted"]

        if show_me:
            plt.figure()
            plt.imshow(predicted)
            plt.figure()
            plt.imshow(problem.options[argmax_sim_predicted_ops - 1])
            plt.show()

        problem_data = {
            "unary_analogies_data": unary_analogies_data,
            "binary_analogies_data": binary_analogies_data,
            "best_analog_name": best_analog_name,
            "best_analog_type": best_analog_type,
            "best_tran": best_tran,
            "best_sim": best_sim,
            "sim_predicted": sim_predicted,
            "argmax_sim_predicted_ops": argmax_sim_predicted_ops
        }

        problem.data = problem_data

        with open("./data/brutal_" + problem.name + ".json", 'w+') as outfile:
            json.dump(problem_data, outfile)
            outfile.close()

        jaccard.save_jaccard_cache(problem.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(problem.name)

    report_brutal.create_report_brutal_mode(problems, test_problems)

    end_time = time.time()

    print(end_time - start_time)


def run_prob_anlg(prob, anlg):
    """
    compute the result given a combination of a problem and an analogy
    :param prob:
    :param anlg:
    :return:
    """

    tran_data = []
    trans = prob_anlg_tran.get_trans(prob, anlg)
    for tran in trans:
        tran_d = run_prob_anlg_tran(prob, anlg, tran)
        tran_data.append(tran_d)

    return tran_data


def run_prob_anlg_tran(prob, anlg, tran):
    """
    compute the result for a combination of a problem, an analogy and a transformation.
    :param prob:
    :param anlg:
    :param tran:
    :return:
    """

    diff_to_u1_x = None
    diff_to_u1_y = None
    diff = None
    b1_to_b2_x = None
    b1_to_b2_y = None

    if 3 == len(anlg.get("value")):  # unary anlg and tran

        u1 = prob.matrix[anlg.get("value")[0]]
        u2 = prob.matrix[anlg.get("value")[1]]

        if "add_diff" == tran.get("name"):
            score, diff_to_u1_x, diff_to_u1_y, _, _, diff = asymmetric_jaccard.asymmetric_jaccard_coef(u1, u2)
        elif "subtract_diff" == tran.get("name"):
            score, _, _, diff_to_u1_x, diff_to_u1_y, diff = asymmetric_jaccard.asymmetric_jaccard_coef(u2, u1)
        else:
            u1_t = transform.apply_unary_transformation(u1, tran)
            score, _, _ = jaccard.jaccard_coef(u1_t, u2)

    elif 5 == len(anlg.get("value")):  # binary anlg and tran

        b1 = prob.matrix[anlg.get("value")[0]]
        b2 = prob.matrix[anlg.get("value")[1]]
        b3 = prob.matrix[anlg.get("value")[2]]

        b1_b2_t, b1_to_b2_x, b1_to_b2_y = transform.apply_binary_transformation(b1, b2, tran)
        score, _, _ = jaccard.jaccard_coef(b1_b2_t, b3)

    else:
        raise Exception("Ryan!")

    return {
        "prob_name": prob.name,
        "anlg_name": anlg.get("name"),
        "tran_name": tran.get("name"),
        "pat_score": score,  # pat = prob + anlg + tran
        "diff_to_u1_x": diff_to_u1_x,
        "diff_to_u1_y": diff_to_u1_y,
        "diff": diff,
        "b1_to_b2_x": b1_to_b2_x,
        "b1_to_b2_y": b1_to_b2_y
    }


def save_image(prob, prediction, selection, show_me = False):
    if show_me:
        plt.figure()
        plt.imshow(prediction)
        plt.figure()
        plt.imshow(selection)
        plt.show()
    else:
        plt.figure()
        plt.imshow(prediction)
        plt.savefig("./data/explanatory_" + prob.name + "_prediction.png")
        plt.close()
        plt.figure()
        plt.imshow(selection)
        plt.savefig("./data/explanatory_" + prob.name + "_selection.png")
        plt.close()


def find_best(data, by_which):
    best_score = -1
    best_ii = None
    for ii, d in enumerate(data):
        if best_score < d.get(by_which):
            best_ii = ii
            best_score = d.get(by_which)

    return copy.copy(data[best_ii])


def predict(prob, d):

    anlg = analogy.get_anlg(d.get("anlg_name"))
    tran = transform.get_tran(d.get("tran_name"))

    if 3 == len(anlg.get("value")):
        best_u1_u2_align_x = d.get("diff_to_u1_x")
        best_u1_u2_align_y = d.get("diff_to_u1_y")
        best_u1_u2_diff = d.get("diff")
        u3 = prob.matrix[anlg.get("value")[2]]

        if tran.get("name") == "add_diff":
            prediction = transform.add_diff(u3, best_u1_u2_align_x, best_u1_u2_align_y, best_u1_u2_diff)
        elif tran.get("name") == "subtract_diff":
            prediction = transform.subtract_diff(u3, best_u1_u2_align_x, best_u1_u2_align_y, best_u1_u2_diff)
        else:
            prediction = transform.apply_unary_transformation(u3, tran)

    elif 5 == len(anlg.get("value")):
        best_b1_to_b2_x = d.get("b1_to_b2_x")
        best_b1_to_b2_y = d.get("b1_to_b2_y")
        b4 = prob.matrix[anlg.get("value")[3]]
        b5 = prob.matrix[anlg.get("value")[4]]
        prediction, _, _ = transform.apply_binary_transformation(b4, b5, tran, best_b1_to_b2_x, best_b1_to_b2_y)

        return prediction
    else:
        raise Exception("Ryan!")

    pred_data = []
    for ii, opt in enumerate(prob.options):
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data

