import json
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import report
import analogy_new
import transform
import jaccard
import asymmetric_jaccard
import prob_anlg_tran_new
from RavenProgressiveMatrix import RavenProgressiveMatrix as RPM
import utils


def run_rave_brutal(show_me = False, test_problems = None):

    start_time = time.time()

    print("run raven in greedy mode.")

    probs = prob_anlg_tran_new.get_probs(test_problems)

    for prob in probs:

        print(prob.name)

        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        anlg_tran_pairs = prob_anlg_tran_new.get_anlg_tran_pairs(prob)

        # given a pair of anlg and tran, compute the predictions
        anlg_tran_data, pred_data = run_prob(prob, anlg_tran_pairs)

        # optimize w.r.t. options
        pred_d = utils.find_best(pred_data, "pat_score", "pato_score")

        # imaging
        save_image(prob, pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], "brutal", show_me)

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
    if test_problems is None:
        report.create_report(probs, "brutal_new_")

    end_time = time.time()
    print(end_time - start_time)


def run_prob(prob, anlg_tran_pairs):
    anlg_tran_data = []
    pred_data = []
    for p in anlg_tran_pairs:
        at_data, at_pred_data = run_prob_anlg_tran(prob, p.get("anlg"), p.get("tran"))
        anlg_tran_data.extend(at_data)
        pred_data.extend(at_pred_data)

    return anlg_tran_data, pred_data


def run_prob_anlg_tran(prob, anlg, tran):
    """
    compute the result for a combination of a problem, an analogy and a transformation.
    :param prob:
    :param anlg:
    :param tran:
    :return: anlg_tran_data, pred_data
    """

    print(prob.name, anlg.get("name"), tran.get("name"))

    if "unary_2x2" == anlg.get("type"):
        # if "B:A::C:?" == anlg.get("name") and "add_diff" == tran.get("name"):
        #     print("asdfasdf")
        return run_prob_anlg_tran_2x2(prob, anlg, tran)
    elif "binary_3x2" == anlg.get("type"):
        return run_prob_anlg_tran_3x2_and_3x2(prob, anlg, tran)
    elif "binary_2x3" == anlg.get("type"):
        # if "D:E:F::G:H:?" == anlg.get("name") and "subtract" == tran.get("name"):
        #     print("asdfasdf")
        return run_prob_anlg_tran_3x2_and_3x2(prob, anlg, tran)
    elif "unary_3x3" == anlg.get("type"):
        # if "B:A::H:G:::A:C::G:?" == anlg.get("name") and "mirror" == tran.get("name"):
        #     print("asdfasdf")
        # if "B:A::H:G:::A:C::G:?" == anlg.get("name") and "mirror_rot_180" == tran.get("name"):
        #     print("asdfasdf")
        return run_prob_anlg_tran_3x3(prob, anlg, tran)
    elif "binary_3x3" == anlg.get("type"):
        # if "A:B:C::D:E:F:::D:E:F::G:H:?" == anlg.get("name") and "inv_unite" == tran.get("name"):
        #     print("asdfasdf")
        return run_prob_anlg_tran_3x3(prob, anlg, tran)
    else:
        raise Exception("Ryan!")


def save_image(prob, prediction, selection, prefix, show_me = False):
    if show_me:
        plt.figure()
        plt.imshow(prediction)
        plt.figure()
        plt.imshow(selection)
        plt.show()
    else:
        plt.figure()
        plt.imshow(prediction)
        plt.savefig("./data/" + prefix + "_" + prob.name + "_prediction.png")
        plt.close()
        plt.figure()
        plt.imshow(selection)
        plt.savefig("./data/" + prefix + "_" + prob.name + "_selection.png")
        plt.close()


def predict(prob, d):
    anlg = analogy_new.get_anlg(d.get("anlg_name"))
    tran = transform.get_tran(d.get("tran_name"))

    if "unary_2x2" == anlg.get("type"):
        return predict_unary(prob, anlg, tran, d)
    elif "binary_3x2" == anlg.get("type") or "binary_2x3" == anlg.get("type"):
        return predict_binary(prob, anlg, tran, d)
    else:
        raise Exception("Ryan!")


def predict_unary(prob, anlg, tran, d):

    if tran.get("name") == "add_diff":
        return predict_add_diff(prob, anlg, tran, d)
    elif tran.get("name") == "subtract_diff":
        return predict_subtract_diff(prob, anlg, tran, d)
    elif tran.get("name") == "xor_diff":
        return predict_xor_diff(prob, anlg, tran, d)
    elif tran.get("name") == "upscale_to":
        return predict_upscale_to(prob, anlg, tran, d)
    elif tran.get("name") == "duplicate":
        return predict_duplicate(prob, anlg, tran, d)
    elif tran.get("name") == "rearrange":
        return predict_rearrange(prob, anlg, tran, d)
    else:
        return predict_unary_default(prob, anlg, tran, d)


def predict_unary_default(prob, anlg, tran, d):
    u3 = prob.matrix[anlg.get("value")[2]]
    prediction = transform.apply_unary_transformation(u3, tran)
    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_rearrange(prob, anlg, tran, d):
    u1_coms_x = d.get("u1_coms_x")
    u1_coms_y = d.get("u1_coms_y")
    u2_coms_x = d.get("u2_coms_x")
    u2_coms_y = d.get("u2_coms_y")

    u3 = prob.matrix[anlg.get("value")[2]]
    prediction = transform.rearrange(u3, u1_coms_x, u1_coms_y, u2_coms_x, u2_coms_y)

    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_duplicate(prob, anlg, tran, d):

    best_copies_to_u1_x = d.get("copies_to_u1_x")
    best_copies_to_u1_y = d.get("copies_to_u1_y")
    u3 = prob.matrix[anlg.get("value")[2]]
    prediction = transform.duplicate(u3, best_copies_to_u1_x, best_copies_to_u1_y)

    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_upscale_to(prob, anlg, tran, d):

    u3 = prob.matrix[anlg.get("value")[2]]

    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        prediction = transform.upscale_to(u3, opt)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_xor_diff(prob, anlg, tran, d):
    best_diff_to_u1_x = d.get("diff_to_u1_x")
    best_diff_to_u1_y = d.get("diff_to_u1_y")
    best_diff = d.get("diff")
    u3 = prob.matrix[anlg.get("value")[2]]
    u1_ref = prob.matrix_ref[anlg.get("value")[0]]

    prediction = transform.xor_diff(u3, best_diff_to_u1_x, best_diff_to_u1_y, best_diff, u1_ref)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        pred_score, _, _ = jaccard.jaccard_coef(prediction, opt)
        u3_score, u3_to_opt_x, u3_to_opt_y = jaccard.jaccard_coef(u3, opt)
        u3_score = 1 - u3_score
        u1_aligned, u2_aligned, aligned_to_u2_x, aligned_to_u2_y = utils.align(u3, opt, u3_to_opt_x, u3_to_opt_y)
        diff = utils.erase_noise_point(np.logical_xor(u1_aligned, u2_aligned), 4)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        score = (diff_score + u3_score + pred_score) / 3
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_subtract_diff(prob, anlg, tran, d):
    best_diff_to_u1_x = d.get("diff_to_u1_x")
    best_diff_to_u1_y = d.get("diff_to_u1_y")
    best_diff_to_u2_x = d.get("diff_to_u2_x")
    best_diff_to_u2_y = d.get("diff_to_u2_y")
    best_diff = d.get("diff")
    u3 = prob.matrix[anlg.get("value")[2]]
    u1_ref = prob.matrix_ref[anlg.get("value")[0]]

    prediction = transform.subtract_diff(u3, best_diff_to_u1_x, best_diff_to_u1_y, best_diff, u1_ref)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        u3_score, diff_to_opt_x, diff_to_opt_y, diff_to_u3_x, diff_to_u3_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(opt, u3)
        opt_to_u3_x = (-diff_to_opt_x) - (-diff_to_u3_x)
        opt_to_u3_y = (-diff_to_opt_y) - (-diff_to_u3_y)
        u2_to_u1_x = (-best_diff_to_u2_x) - (-best_diff_to_u1_x)
        u2_to_u1_y = (-best_diff_to_u2_y) - (-best_diff_to_u1_y)
        if abs(opt_to_u3_x - u2_to_u1_x) > 2 or abs(opt_to_u3_y - u2_to_u1_y) > 2:
            u3_score, diff = asymmetric_jaccard.asymmetric_jaccard_coef_pos_fixed(opt, u3, u2_to_u1_x, u2_to_u1_y)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        opt_score, _, _ = jaccard.jaccard_coef(opt, prediction)
        score = (diff_score + opt_score + u3_score) / 3
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_add_diff(prob, anlg, tran, d):

    best_diff_to_u1_x = d.get("diff_to_u1_x")
    best_diff_to_u1_y = d.get("diff_to_u1_y")
    best_diff_to_u2_x = d.get("diff_to_u2_x")
    best_diff_to_u2_y = d.get("diff_to_u2_y")
    best_diff = d.get("diff")

    u3 = prob.matrix[anlg.get("value")[2]]
    u1_ref = prob.matrix_ref[anlg.get("value")[0]]
    prediction = transform.add_diff(u3, best_diff_to_u1_x, best_diff_to_u1_y, best_diff, u1_ref)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        u3_score, diff_to_u3_x, diff_to_u3_y, diff_to_opt_x, diff_to_opt_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u3, opt)
        u3_to_opt_x = (-diff_to_u3_x) - (-diff_to_opt_x)
        u3_to_opt_y = (-diff_to_u3_y) - (-diff_to_opt_y)
        u1_to_u2_x = (-best_diff_to_u1_x) - (-best_diff_to_u2_x)
        u1_to_u2_y = (-best_diff_to_u1_y) - (-best_diff_to_u2_y)
        if abs(u3_to_opt_x - u1_to_u2_x) > 2 or abs(u3_to_opt_y - u1_to_u2_y) > 2:
            u3_score, diff = asymmetric_jaccard.asymmetric_jaccard_coef_pos_fixed(u3, opt, u1_to_u2_x, u1_to_u2_y)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        opt_score, _, _ = jaccard.jaccard_coef(opt, prediction)
        score = (diff_score + opt_score + u3_score) / 3

        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_binary(prob, anlg, tran, d):

    if "unite" == tran.get("name"):
        return predict_unite(prob, anlg, tran, d)
    elif "inv_unite" == tran.get("name"):
        return predict_inv_unite(prob, anlg, tran, d)
    else:
        return predict_binary_default(prob, anlg, tran, d)


def predict_inv_unite(prob, anlg, tran, d):
    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        # prediction = transform.inv_unite(b4, b5, opt)
        # score, _, _ = jaccard.jaccard_coef(opt, prediction)
        b4_new, _, _, _, _ = transform.apply_binary_transformation(b5, opt, transform.get_tran("unite"), imgC = b4)
        score, _, _ = jaccard.jaccard_coef(b4_new, b4)
        b5_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b5, opt)
        opt_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(opt, b5)
        if max(b5_score, opt_score) > 0.85:
            score = 0
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": opt})

    return pred_data


def predict_unite(prob, anlg, tran, d):
    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        prediction, _, _, _, _ = transform.apply_binary_transformation(b4, b5, tran, imgC = opt)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def predict_binary_default(prob, anlg, tran, d):

    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]

    prediction, _, _, _, _ = transform.apply_binary_transformation(b4, b5, tran)

    pred_data = []
    for ii, opt in enumerate(prob.options):
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "pato_score": score, "pred": prediction})

    return pred_data


def run_prob_anlg_tran_2x2(prob, anlg, tran):
    u1 = prob.matrix[anlg.get("value")[0]]
    u2 = prob.matrix[anlg.get("value")[1]]

    diff_to_u1_x = None
    diff_to_u1_y = None
    diff_to_u2_x = None
    diff_to_u2_y = None
    diff = None
    copies_to_u1_x = None
    copies_to_u1_y = None
    u1_coms_x = None
    u1_coms_y = None
    u2_coms_x = None
    u2_coms_y = None

    if "add_diff" == tran.get("name"):
        score, diff_to_u1_x, diff_to_u1_y, diff_to_u2_x, diff_to_u2_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u1, u2)
    elif "subtract_diff" == tran.get("name"):
        score, diff_to_u2_x, diff_to_u2_y, diff_to_u1_x, diff_to_u1_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u2, u1)
    elif "xor_diff" == tran.get("name"):
        score, u1_to_u2_x, u1_to_u2_y = jaccard.jaccard_coef(u1, u2)
        score = 1 - score
        u1_aligned, u2_aligned, aligned_to_u2_x, aligned_to_u2_y = utils.align(u1, u2, u1_to_u2_x, u1_to_u2_y)
        diff = utils.erase_noise_point(np.logical_xor(u1_aligned, u2_aligned), 4)
        diff_to_u2_x = int(aligned_to_u2_x)
        diff_to_u2_y = int(aligned_to_u2_y)
        diff_to_u1_x = int(diff_to_u2_x - u1_to_u2_x)
        diff_to_u1_y = int(diff_to_u2_y - u1_to_u2_y)
    elif "upscale_to" == tran.get("name"):
        u1_upscaled = transform.upscale_to(u1, u2)
        score, _, _ = jaccard.jaccard_coef(u2, u1_upscaled)
    elif "duplicate" == tran.get("name"):
        scores = []
        u1_to_u2_xs = []
        u1_to_u2_ys = []
        current = u2.copy()
        current_to_u2_x = 0
        current_to_u2_y = 0
        while current.sum():
            score_tmp, diff_tmp_to_u1_x, diff_tmp_to_u1_y, diff_tmp_to_current_x, diff_tmp_to_current_y, _ = \
                asymmetric_jaccard.asymmetric_jaccard_coef(u1, current)

            if score_tmp < 0.6:
                break

            scores.append(score_tmp)
            u1_to_current_x = (-diff_tmp_to_u1_x) - (-diff_tmp_to_current_x)
            u1_to_current_y = (-diff_tmp_to_u1_y) - (-diff_tmp_to_current_y)
            u1_to_u2_x = u1_to_current_x + current_to_u2_x
            u1_to_u2_y = u1_to_current_y + current_to_u2_y
            u1_to_u2_xs.append(u1_to_u2_x)
            u1_to_u2_ys.append(u1_to_u2_y)
            u1_aligned, current_aligned, aligned_to_current_x, aligned_to_current_y = utils.align(
                u1, current, u1_to_current_x, u1_to_current_y)
            current = utils.erase_noise_point(np.logical_and(current_aligned, np.logical_not(u1_aligned)), 8)
            current_to_u2_x = aligned_to_current_x + current_to_u2_x
            current_to_u2_y = aligned_to_current_y + current_to_u2_y

        if 1 >= len(scores):
            score = 0
            copies_to_u1_x = [0]
            copies_to_u1_y = [0]
        else:
            score = min(scores)
            copies_to_u1_x = (np.array(u1_to_u2_xs[1 :]) - u1_to_u2_xs[0]).tolist()
            copies_to_u1_y = (np.array(u1_to_u2_ys[1 :]) - u1_to_u2_ys[0]).tolist()
    elif "rearrange" == tran.get("name"):
        score, u1_coms_x, u1_coms_y, u2_coms_x, u2_coms_y = transform.evaluate_rearrange(u1, u2)
    else:
        u1_t = transform.apply_unary_transformation(u1, tran)
        score, _, _ = jaccard.jaccard_coef(u1_t, u2)

    if "mirror" == tran.get("name") or "mirror_rot_180" == tran.get("name"):
        u1_mirror_score, _, _ = jaccard.jaccard_coef(u1_t, u1)
        u2_mirror = transform.apply_unary_transformation(u2, tran)
        u2_mirror_score, _, _ = jaccard.jaccard_coef(u2_mirror, u2)
        if max(u1_mirror_score, u2_mirror_score) > 0.9:
            score = 0

    prob_anlg_tran_d = assemble_prob_anlg_tran_d(prob, anlg, tran, score,
                                                 diff_to_u1_x = diff_to_u1_x, diff_to_u1_y = diff_to_u1_y,
                                                 diff_to_u2_x = diff_to_u2_x, diff_to_u2_y = diff_to_u2_y,
                                                 diff = diff,
                                                 copies_to_u1_x = copies_to_u1_x,
                                                 copies_to_u1_y = copies_to_u1_y,
                                                 u1_coms_x = u1_coms_x,
                                                 u1_coms_y = u1_coms_y,
                                                 u2_coms_x = u2_coms_x,
                                                 u2_coms_y = u2_coms_y)

    pred_data = predict(prob, prob_anlg_tran_d)

    return [prob_anlg_tran_d], pred_data


def run_prob_anlg_tran_3x2_and_3x2(prob, anlg, tran):
    b1 = prob.matrix[anlg.get("value")[0]]
    b2 = prob.matrix[anlg.get("value")[1]]
    b3 = prob.matrix[anlg.get("value")[2]]

    if "inv_unite" == tran.get("name"):
        b1_new, _, _, _, _ = transform.apply_binary_transformation(b2, b3, transform.get_tran("unite"), imgC = b1)
        score, _, _ = jaccard.jaccard_coef(b1_new, b1)
        b1_to_b2_x = None
        b1_to_b2_y = None
        # b1_b2_t = transform.inv_unite(b1, b2, b3)
        # score, _, _ = jaccard.jaccard_coef(b1_b2_t, b3)
    else:
        b1_b2_t, b1_to_b2_x, b1_to_b2_y, _, _ = transform.apply_binary_transformation(b1, b2, tran, imgC = b3)
        score, _, _ = jaccard.jaccard_coef(b1_b2_t, b3)

    if "inv_unite" == tran.get("name"):
        b2_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b2, b3)
        b3_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b3, b2)
        if max(b2_score, b3_score) > 0.9:
            score = 0

    if "unite" == tran.get("name") or "shadow_mask_unite" == tran.get("name"):
        b1_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b1, b2)
        b2_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b2, b1)
        if max(b1_score, b2_score) > 0.9:  # if b1 is almost a subset of b2 or vice versa
            score = 0

    prob_anlg_tran_d = assemble_prob_anlg_tran_d(prob, anlg, tran, score,
                                                 b1_to_b2_x = b1_to_b2_x, b1_to_b2_y = b1_to_b2_y)
    pred_data = predict(prob, prob_anlg_tran_d)

    return [prob_anlg_tran_d], pred_data


def run_prob_anlg_tran_3x3(prob, anlg, tran):

    sub_probs = get_sub_probs(prob, anlg)
    sub_prob_n = len(sub_probs)

    chld_anlg = analogy_new.get_anlg(anlg.get("chld_name"))

    anlg_tran_data = []
    apriori_pred_data = []
    aposteriori_pred_data = None
    for ii, p in enumerate(sub_probs):
        sub_prob_anlg_tran_data, sub_pred_data = run_prob_anlg_tran(p, chld_anlg, tran)
        anlg_tran_data.extend(sub_prob_anlg_tran_data)

        if ii < len(sub_probs) - 1:
            apriori_pred_data.extend(sub_pred_data)
        else:
            aposteriori_pred_data = sub_pred_data

    pat_score_sum, pato_score_sum = utils.sum_score(apriori_pred_data, "pat_score", "pato_score")

    pred_data = []
    for d in aposteriori_pred_data:
        pred_data.append({
            "prob_name": prob.name,
            "anlg_name": anlg.get("name"),
            "tran_name": tran.get("name"),
            "prob_type": prob.type,
            "anlg_type": anlg.get("type"),
            "tran_type": tran.get("type"),
            "pat_score": (d["pat_score"] + pat_score_sum) / sub_prob_n,
            "pato_score": (d["pato_score"] + pato_score_sum) / sub_prob_n,
            "prob_ansr": prob.answer,
            "diff_to_u1_x": d.get("diff_to_u1_x"),
            "diff_to_u1_y": d.get("diff_to_u1_y"),
            "diff_to_u2_x": d.get("diff_to_u2_x"),
            "diff_to_u2_y": d.get("diff_to_u2_y"),
            "diff": d.get("diff"),
            "b1_to_b2_x": d.get("b1_to_b2_x"),
            "b1_to_b2_y": d.get("b1_to_b2_y"),
            "optn": d.get("optn"),
            "pred": d.get("pred")
        })

    return anlg_tran_data, pred_data


def get_sub_probs(prob, anlg):

    value = anlg.get("value")
    child_name = anlg.get("chld_name")
    child_n = anlg.get("chld_n")

    child_anlg = analogy_new.get_anlg(child_name)
    shape = child_anlg.get("shape")

    sub_probs = []
    for ii in range(child_n):
        if "binary" in anlg.get("type"):
            coords = value[ii * 6 : (ii + 1) * 6]
        else:
            coords = value[ii * 4 : (ii + 1) * 4]

        prob_name = prob.name + "_sub_" + anlg.get("name") + "_" + str(ii)

        coms = []
        for coord in coords:
            coms.append(prob.matrix[coord])
        matrix = utils.create_object_matrix(coms, shape)

        if ii == child_n - 1:
            options = prob.options
            answer = prob.answer
        else:
            matrix[-1, -1] = np.full_like(coms[-1], fill_value = False)
            options = [coms[-1]]
            answer = 1

        sub_probs.append(RPM(prob_name, matrix, options, answer))

    return sub_probs


def assemble_prob_anlg_tran_d(prob, anlg, tran, pat_score,
                              diff_to_u1_x = None,
                              diff_to_u1_y = None,
                              diff_to_u2_x = None,
                              diff_to_u2_y = None,
                              diff = None,
                              b1_to_b2_x = None,
                              b1_to_b2_y = None,
                              copies_to_u1_x = None,
                              copies_to_u1_y = None,
                              u1_coms_x = None,
                              u1_coms_y = None,
                              u2_coms_x = None,
                              u2_coms_y = None):

    return {
        "prob_name": prob.name,
        "anlg_name": anlg.get("name"),
        "tran_name": tran.get("name"),
        "pat_score": pat_score,  # pat = prob + anlg + tran
        "prob_ansr": prob.answer,
        "prob_type": prob.type,
        "anlg_type": anlg.get("type"),
        "tran_type": tran.get("type"),
        "diff_to_u1_x": diff_to_u1_x,
        "diff_to_u1_y": diff_to_u1_y,
        "diff_to_u2_x": diff_to_u2_x,
        "diff_to_u2_y": diff_to_u2_y,
        "diff": diff,
        "b1_to_b2_x": b1_to_b2_x,
        "b1_to_b2_y": b1_to_b2_y,
        "copies_to_u1_x": copies_to_u1_x,
        "copies_to_u1_y": copies_to_u1_y,
        "u1_coms_x": u1_coms_x,
        "u1_coms_y": u1_coms_y,
        "u2_coms_x": u2_coms_x,
        "u2_coms_y": u2_coms_y
    }
