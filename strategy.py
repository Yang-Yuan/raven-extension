import prob_anlg_tran_new
from digest import digest
from predict import predict
import utils


def strategy_1(prob):

    anlg_tran_data = []
    pred_data = []
    pred_d = None

    # get all pairs of anlg and trans for this prob
    anlg_tran_pairs = prob_anlg_tran_new.get_anlg_tran_pairs(prob)

    # explain the problem with each pair of anlg and tran
    for anlg, tran in anlg_tran_pairs:
        anlg_tran_data.append(digest(prob, anlg, tran))

    # get the best explanation over all pairs of anlg and tran
    anlg_tran_d = utils.find_best(anlg_tran_data, "pat_score")
    anlg_tran_data = [anlg_tran_d]

    # get the prediction of the best explanation
    pred_data = predict(prob, anlg_tran_d)

    # find the best prediction
    pred_d = utils.find_best(pred_data, "pato_score")

    return anlg_tran_data, pred_data, pred_d


def strategy_2(prob):

    anlg_tran_data = []
    pred_data = []
    pred_d = None

    # get all the anlgs
    anlgs = prob_anlg_tran_new.get_anlgs(prob)

    # explain the prob with each analogy
    for anlg in anlgs:
        tran_data = []

        # get all the trans for this anlg
        trans = prob_anlg_tran_new.get_trans(prob, anlg)

        # explan the prob with the anlg and the tran
        for tran in trans:
            tran_data.append(digest(prob, anlg, tran))

        # find the best explanation of this anlg over all trans
        anlg_tran_data.append(utils.find_best(tran_data, "pat_score"))

    # predict with the best explanation of each anlg
    for anlg_tran_d in anlg_tran_data:
        pred_data.extend(predict(prob, anlg_tran_d))

    # get the best prediction
    pred_d = utils.find_best(pred_data, "pato_score")

    return anlg_tran_data, pred_data, pred_d


def strategy_3(prob):

    anlg_tran_data = []
    pred_data = []
    pred_d = None

    # get all pairs of anlg and trans for this prob
    anlg_tran_pairs = prob_anlg_tran_new.get_anlg_tran_pairs(prob)

    # explain the problem with each pair of anlg and tran
    for anlg, tran in anlg_tran_pairs:
        anlg_tran_data.append(digest(prob, anlg, tran))

    # predict by the explanation of each pair of anlg ang tran
    for anlg_tran_d in anlg_tran_data:
        pred_data.extend(predict(prob, anlg_tran_d))

    # get the best prediction
    pred_d = utils.find_best(pred_data, "pato_score")

    return anlg_tran_data, pred_data, pred_d
