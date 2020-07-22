from matplotlib import pyplot as plt
import time
import report
import jaccard
import asymmetric_jaccard
import prob_anlg_tran_new
import utils


def run(strategy, explanation_score_name = "mat_score", prediction_score_name = "mato_score",
        show_me = False, test_problems = None, test_name = "spm", test_anlgs = None, test_trans = None):
    start_time = time.time()

    probs = prob_anlg_tran_new.get_probs(test_problems, test_name)

    for prob in probs:
        print(prob.name)

        # initialize cache
        jaccard.load_jaccard_cache(prob.name)
        asymmetric_jaccard.load_asymmetric_jaccard_cache(prob.name)

        # run strategy
        anlg_tran_data, pred_data, pred_d = strategy(prob,
                                                     explanation_score_name = explanation_score_name,
                                                     prediction_score_name = prediction_score_name,
                                                     test_anlgs = test_anlgs,
                                                     test_trans = test_trans)

        # save data
        prob.data = utils.save_data(prob, anlg_tran_data, pred_data, pred_d,
                                    "./data/" + test_name + "_" + strategy.__name__ + "_" + prediction_score_name + "_" + prob.name,
                                    show_me)

        # update cache
        jaccard.save_jaccard_cache(prob.name)
        asymmetric_jaccard.save_asymmetric_jaccard_cache(prob.name)

    # generate report
    report.create_report(probs, test_name + "_" + strategy.__name__ + "_" + prediction_score_name + "_")

    end_time = time.time()
    print(end_time - start_time)
