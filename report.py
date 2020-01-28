import xlsxwriter
from datetime import datetime
from os.path import join

report_folder = "./reports"


# TODO improve this part with pandas

def create_report(problems):
    file_name = "raven_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".xlsx"
    workbook = xlsxwriter.Workbook(join(report_folder,file_name))

    problem_data_frame = get_problem_data_frame(problems)

    create_problem_worksheet(workbook, problem_data_frame)

    # anlg_worksheet = workbook.add_worksheet("Analogies")
    #
    # tran_worksheet = workbook.add_worksheet("Transformations")

    workbook.close()


def create_problem_worksheet(workbook, problem_data_frame):
    prob_worksheet = workbook.add_worksheet("Problems")

    col_names = ["Problem", "Prediction", "Truth", "Winning Analogy Type",
                 "Winning Analogy", "Win-With Similarity (Analogy)",
                 "Winning Transformation", "Win-With Similarity (Transformation)"]
    col_widths = [15, 15, 15, 20, 15, 25, 21, 31]

    bold = workbook.add_format({'bold': True})

    for jj, col_name, width in zip(range(len(col_names)), col_names, col_widths):
        prob_worksheet.set_column(jj, jj, width)
        prob_worksheet.write(0, jj, col_name, bold)

    for ii, prob_d in zip(range(len(problem_data_frame)), problem_data_frame):
        prob_worksheet.write(ii + 1, 0, prob_d.get("problem_name"))
        prob_worksheet.write(ii + 1, 1, prob_d.get("prediction"))
        prob_worksheet.write(ii + 1, 2, prob_d.get("truth"))
        prob_worksheet.write(ii + 1, 3, prob_d.get("win_unary_or_binary"))
        prob_worksheet.write(ii + 1, 4, prob_d.get("winning_anlg"))
        prob_worksheet.write(ii + 1, 5, prob_d.get("win_with_sim_anlg"))
        prob_worksheet.write(ii + 1, 6, prob_d.get("winning_tran"))
        prob_worksheet.write(ii + 1, 7, prob_d.get("win_with_sim_tran"))


def get_problem_data_frame(problems):
    data_frame = []

    for prob in problems:
        prob_data = prob.data

        d_result = {"problem_name": prob.name}

        prediction = None
        winning_anlg = None
        win_with_sim_anlg = None
        winning_tran = None
        win_with_sim_tran = None
        win_unary_or_binary = None

        unary_analogies = prob_data.get("unary_analogies_data")
        for anlg_name, anlg in unary_analogies.items():

            sim_u4_predicted_ops = anlg.get("sim_u4_predicted_ops")
            argmax_sim_u4_predicted_ops = anlg.get("argmax_sim_u4_predicted_ops") - 1
            if win_with_sim_anlg is None or win_with_sim_anlg < sim_u4_predicted_ops[argmax_sim_u4_predicted_ops]:
                prediction = argmax_sim_u4_predicted_ops + 1
                winning_anlg = anlg_name
                win_with_sim_anlg = sim_u4_predicted_ops[argmax_sim_u4_predicted_ops]
                winning_tran = str(anlg.get("unary_tran"))
                win_with_sim_tran = max(anlg.get("sim_u1_trans_u2"))
                win_unary_or_binary = "unary"

        binary_analogies = prob_data.get("binary_analogies_data")
        for anlg_name, anlg in binary_analogies.items():

            sim_b6_predicted_ops = anlg.get("sim_b6_predicted_ops")
            argmax_sim_b6_predicted_ops = anlg.get("argmax_sim_b6_predicted_ops") - 1
            if win_with_sim_anlg is None or win_with_sim_anlg < sim_b6_predicted_ops[argmax_sim_b6_predicted_ops]:
                prediction = argmax_sim_b6_predicted_ops + 1
                winning_anlg = anlg_name
                win_with_sim_anlg = sim_b6_predicted_ops[argmax_sim_b6_predicted_ops]
                winning_tran = str(anlg.get("b1_b2_tran"))
                win_with_sim_tran = max(anlg.get("sim_b1_b2_trans_b3"))
                win_unary_or_binary = "binary"

        d_result["prediction"] = prediction
        d_result["win_unary_or_binary"] = win_unary_or_binary
        d_result["winning_anlg"] = winning_anlg
        d_result["win_with_sim_anlg"] = win_with_sim_anlg
        d_result["winning_tran"] = winning_tran
        d_result["win_with_sim_tran"] = win_with_sim_tran

        data_frame.append(d_result)

    return data_frame
