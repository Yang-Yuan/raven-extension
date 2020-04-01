import json
from matplotlib import pyplot as plt
import time
import report
import jaccard
import asymmetric_jaccard
import prob_anlg_tran_new
from digest import digest
from predict import predict
import utils


def run_rave_brutal(show_me = False, test_problems = None):

    start_time = time.time()

    print("run raven in greedy mode.")

    probs = prob_anlg_tran_new.get_probs(test_problems)

    for prob in probs:

        print(prob.name)

        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        # get all pairs of anlg and trans for this prob
        anlg_tran_pairs = prob_anlg_tran_new.get_anlg_tran_pairs(prob)

        # explain the problem with each pair of anlg and tran
        anlg_tran_data = []
        for anlg, tran in anlg_tran_pairs:
            anlg_tran_data.append(digest(prob, anlg, tran))

        # predict by the explanation of each pair of anlg ang tran
        pred_data = []
        for anlg_tran_d in anlg_tran_data:
            pred_data.extend(predict(prob, anlg_tran_d))

        # optimize w.r.t. options
        pred_d = utils.find_best(pred_data, "pat_score", "pato_score")

        # save image
        save_image(pred_d.get("pred"), prob.options[pred_d.get("optn") - 1],
                   "./data/brutal_" + prob.name,
                   show_me)

        # save data
        prob.data = save_data(anlg_tran_data, pred_data, pred_d,
                              "./data/brutal_" + prob.name)

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

    # generate report
    if test_problems is None:
        report.create_report(probs, "brutal_new_")

    end_time = time.time()
    print(end_time - start_time)


def save_image(prediction, selection, prefix, show_me = False):
    if show_me:
        plt.figure()
        plt.imshow(prediction)
        plt.figure()
        plt.imshow(selection)
        plt.show()
    else:
        plt.figure()
        plt.imshow(prediction)
        plt.savefig(prefix + "_prediction.png")
        plt.close()
        plt.figure()
        plt.imshow(selection)
        plt.savefig(prefix + "_selection.png")
        plt.close()


def save_data(anlg_tran_data, pred_data, pred_d, prefix):
    for d in anlg_tran_data:
        d.pop("last_sub_prob", None)
        d.pop("last_sub_prob_anlg_tran_d", None)
        d.pop("diff", None)
        d.pop("diff_to_u1_x", None)
        d.pop("diff_to_u1_y", None)
        d.pop("diff_to_u2_x", None)
        d.pop("diff_to_u2_y", None)
    for d in pred_data:
        d.pop("diff", None)
        d.pop("pred", None)
        d.pop("diff_to_u1_x", None)
        d.pop("diff_to_u1_y", None)
        d.pop("diff_to_u2_x", None)
        d.pop("diff_to_u2_y", None)
    pred_d.pop("diff", None)
    pred_d.pop("pred", None)

    aggregation_progression = {
        "anlg_tran_data": anlg_tran_data,
        "pred_data": pred_data,
        "pred_d": pred_d
    }

    with open(prefix + ".json", 'w+') as outfile:
        json.dump(aggregation_progression, outfile)
        outfile.close()

    return aggregation_progression
