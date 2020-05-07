from matplotlib import pyplot as plt
import time
import report
import jaccard
import asymmetric_jaccard
import prob_anlg_tran_new
import utils


def run_raven(strategy, explanation_score_name = "pat_score", prediction_score_name = "pato_score",
              show_me = False, test_problems = None):

    start_time = time.time()

    print("run raven in greedy mode.")

    probs = prob_anlg_tran_new.get_probs(test_problems)

    for prob in probs:

        print(prob.name)

        # initialize cache
        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        # run strategy
        anlg_tran_data, pred_data, pred_d = strategy(prob, explanation_score_name = explanation_score_name,
                                                     prediction_score_name = prediction_score_name)

        # save data
        prob.data = utils.save_data(prob, anlg_tran_data, pred_data, pred_d, "./data/" + strategy.__name__ + "_" + prob.name, show_me)

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

    # generate report
    if test_problems is None:
        report.create_report(probs, strategy.__name__ + "_")

    end_time = time.time()
    print(end_time - start_time)


