import json
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import report
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

        anlg_tran_data = []
        anlg_data = []
        for anlg in anlgs:
            # score all transformations given an analogy
            tran_data = run_prob_anlg(prob, anlg)
            anlg_tran_data.extend(tran_data)

            # optimize w.r.t. transformations (pat_score) for this analogy
            anlg_tran_d = find_best(tran_data, "pat_score")
            anlg_data.append(anlg_tran_d)

        # optimize w.r.t analogies (pat_score)
        anlg_d = find_best(anlg_data, "pat_score")

        # predict with the best analogy and score all options with the prediction
        pred_data = predict(prob, anlg_d)

        # optimize w.r.t.options (pato_score)
        pred_d = find_best(pred_data, "pat_score", "pato_score")

        # imaging
        save_image(prob, pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], show_me)

        # data aggregation progression, TODO maybe save them as images
        for d in anlg_tran_data:
            del d["diff"]
        for d in anlg_data:
            del d["diff"]
        for d in pred_data:
            del d["diff"]
            del d["pred"]
        del pred_d["diff"]
        del pred_d["pred"]
        aggregation_progression = {
            "anlg_tran_data": anlg_tran_data,
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

        prob.data = aggregation_progression

    # output report
    report.create_report(probs, "explanatory_")

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

        anlg_tran_data = []
        anlg_data = []
        for anlg in anlgs:
            # score all transformations given an analogy
            tran_data = run_prob_anlg(prob, anlg)
            anlg_tran_data.extend(tran_data)

            # optimize w.r.t. transformations for this analogy
            anlg_tran_d = find_best(tran_data, "pat_score")
            anlg_data.append(anlg_tran_d)

        pred_data = []
        for anlg_d in anlg_data:
            # predict with an analogy, and score all options with the prediction
            anlg_pred_data = predict(prob, anlg_d)
            pred_data.extend(anlg_pred_data)

        # optimize w.r.t. options
        pred_d = find_best(pred_data, "pat_score", "pato_score")

        # imaging
        save_image(prob, pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], show_me)

        # data aggregation progression, TODO maybe save them as images
        for d in anlg_tran_data:
            del d["diff"]
        for d in anlg_data:
            del d["diff"]
        for d in pred_data:
            del d["diff"]
            del d["pred"]
        del pred_d["diff"]
        del pred_d["pred"]
        aggregation_progression = {
            "anlg_tran_data": anlg_tran_data,
            "anlg_data": anlg_data,
            "pred_data": pred_data,
            "pred_d": pred_d
        }
        with open("./data/greedy_" + prob.name + ".json", 'w+') as outfile:
            json.dump(aggregation_progression, outfile)
            outfile.close()

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

        prob.data = aggregation_progression

    # output report
    report.create_report(probs, "greedy_")

    end_time = time.time()
    print(end_time - start_time)


def run_rave_brutal(show_me = False, test_problems = None):

    start_time = time.time()

    print("run raven in greedy mode.")

    probs = prob_anlg_tran.get_probs(test_problems)

    for prob in probs:

        print(prob.name)

        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        anlgs = prob_anlg_tran.get_anlgs(prob)

        anlg_tran_data = []
        for anlg in anlgs:
            # score all transformations given an analogy
            tran_data = run_prob_anlg(prob, anlg)
            anlg_tran_data.extend(tran_data)

        pred_data = []
        for anlg_tran_d in anlg_tran_data:
            # predict with an analogy and a transformation, and score all options with the prediction
            anlg_tran_pred_data = predict(prob, anlg_tran_d)
            pred_data.extend(anlg_tran_pred_data)

        # optimize w.r.t. options
        pred_d = find_best(pred_data, "pat_score", "pato_score")

        # imaging
        save_image(prob, pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], show_me)

        # data aggregation progression, TODO maybe save them as images
        for d in anlg_tran_data:
            del d["diff"]
        for d in pred_data:
            del d["diff"]
            del d["pred"]
        del pred_d["diff"]
        del pred_d["pred"]
        aggregation_progression = {
            "anlg_tran_data": anlg_tran_data,
            "pred_data": pred_data,
            "pred_d": pred_d
        }
        with open("./data/brutal_" + prob.name + ".json", 'w+') as outfile:
            json.dump(aggregation_progression, outfile)
            outfile.close()

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

        prob.data = aggregation_progression

    # output report
    report.create_report(probs, "brutal_")

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

        print(prob.name, anlg.get("name"), tran.get("name"))

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
        "prob_ansr": prob.answer,
        "prob_type": prob.type,
        "anlg_type": anlg.get("type"),
        "tran_type": tran.get("type"),
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


def find_best(data, *score_names):
    best_score = -1
    best_ii = None
    for ii, d in enumerate(data):
        score = 0
        for score_name in score_names:
            score += d.get(score_name)
        if best_score < score:
            best_ii = ii
            best_score = score

    return copy.copy(data[best_ii])


def predict(prob, d):

    anlg = analogy.get_anlg(d.get("anlg_name"))
    tran = transform.get_tran(d.get("tran_name"))

    if 3 == len(anlg.get("value")):
        best_u1_u2_align_x = d.get("diff_to_u1_x")
        best_u1_u2_align_y = d.get("diff_to_u1_y")
        best_u1_u2_diff = d.get("diff")
        u3 = prob.matrix[anlg.get("value")[2]]
        u1 = prob.matrix[anlg.get("value")[0]]

        if tran.get("name") == "add_diff":
            prediction = transform.add_diff(u3, best_u1_u2_align_x, best_u1_u2_align_y, best_u1_u2_diff, u1)
        elif tran.get("name") == "subtract_diff":
            prediction = transform.subtract_diff(u3, best_u1_u2_align_x, best_u1_u2_align_y, best_u1_u2_diff, u1)
        else:
            prediction = transform.apply_unary_transformation(u3, tran)

    elif 5 == len(anlg.get("value")):
        best_b1_to_b2_x = d.get("b1_to_b2_x")
        best_b1_to_b2_y = d.get("b1_to_b2_y")
        b4 = prob.matrix[anlg.get("value")[3]]
        b5 = prob.matrix[anlg.get("value")[4]]
        prediction, _, _ = transform.apply_binary_transformation(b4, b5, tran, best_b1_to_b2_x, best_b1_to_b2_y)

    else:
        raise Exception("Ryan!")

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data

