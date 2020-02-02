import xlsxwriter
from datetime import datetime
from os.path import join
from analogy import unary_analogies_2by2, unary_analogies_3by3, binary_analogies_3by3
from transform import unary_transformations, binary_transformations

report_folder = "./reports/explanatory"


# TODO improve this part with pandas or other framework


def create_report_explanatory_mode(problems):
    file_name = "raven_explanatory_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".xlsx"
    workbook = xlsxwriter.Workbook(join(report_folder, file_name))

    problem_data_frame, analogy_data_frame, transformation_data_frame = get_data_frame(problems)

    create_problem_worksheet(workbook, problem_data_frame)

    create_analogy_worksheet(workbook, analogy_data_frame)

    create_transformation_worksheet(workbook, transformation_data_frame)

    workbook.close()


def create_problem_worksheet(workbook, problem_data_frame):
    prob_worksheet = workbook.add_worksheet("Problems")

    col_names = ["Problem", "Truth", "Prediction", "Prediction Similarity",
                 "Matrix Type", "Winning Analogy Type",
                 "Winning Analogy", "Winning Transformation",
                 "Win-With Similarity (Analogy & Transformation)"]
    col_widths = [15, 15, 15, 15, 15, 15, 15, 32, 15]

    bold = workbook.add_format({'bold': True, 'text_wrap': True})
    red_text = workbook.add_format({"color": "#FF0000"})

    for jj, col_name, width in zip(range(len(col_names)), col_names, col_widths):
        prob_worksheet.set_column(jj, jj, width)
        prob_worksheet.write(0, jj, col_name, bold)

    accuracy = 0
    for ii, prob_d in zip(range(len(problem_data_frame)), problem_data_frame):
        prob_worksheet.write(ii + 1, 0, prob_d.get("problem_name"))
        truth = prob_d.get("truth")
        prediction = prob_d.get("prediction")
        if truth == prediction:
            prob_worksheet.write(ii + 1, 1, truth)
            prob_worksheet.write(ii + 1, 2, prediction)
            accuracy += 1
        else:
            prob_worksheet.write(ii + 1, 1, truth, red_text)
            prob_worksheet.write(ii + 1, 2, prediction, red_text)
        prob_worksheet.write(ii + 1, 3, prob_d.get("prediction_sim"))
        prob_worksheet.write(ii + 1, 4, prob_d.get("matrix_type"))
        prob_worksheet.write(ii + 1, 5, prob_d.get("win_unary_or_binary"))
        prob_worksheet.write(ii + 1, 6, prob_d.get("winning_anlg"))
        prob_worksheet.write(ii + 1, 7, prob_d.get("winning_tran"))
        prob_worksheet.write(ii + 1, 8, prob_d.get("win_with_sim"))

    prob_worksheet.write(62, 0, "Accuracy:", bold)
    prob_worksheet.write(62, 1, str(accuracy) + "/60", bold)

def create_analogy_worksheet(workbook, analogy_data_frame):
    anlg_worksheet = workbook.add_worksheet("Analogies")

    col_names = ["Analogy", "Analogy Type", "Matrix Type", "Num of Optimal Choices",
                 "Num of Optimal Choices for 2x2 Prob", "Num of Optimal Choices for 3x3 Prob",
                 "Problems"]
    col_widths = [15, 15, 15, 15, 15, 15, 50]

    bold = workbook.add_format({'bold': True, 'text_wrap': True})

    for jj, col_name, width in zip(range(len(col_names)), col_names, col_widths):
        anlg_worksheet.set_column(jj, jj, width)
        anlg_worksheet.write(0, jj, col_name, bold)

    for ii, anlg_d in zip(range(len(analogy_data_frame)), analogy_data_frame):
        anlg_worksheet.write(ii + 1, 0, anlg_d.get("analogy_name"))
        anlg_worksheet.write(ii + 1, 1, anlg_d.get("analogy_type"))
        anlg_worksheet.write(ii + 1, 2, anlg_d.get("matrix_type"))
        anlg_worksheet.write(ii + 1, 3, anlg_d.get("n_optimal"))
        anlg_worksheet.write(ii + 1, 4, anlg_d.get("n_optimal_2x2"))
        anlg_worksheet.write(ii + 1, 5, anlg_d.get("n_optimal_3x3"))
        anlg_worksheet.write(ii + 1, 6, anlg_d.get("problems"))


def create_transformation_worksheet(workbook, transformation_data_frame):
    tran_worksheet = workbook.add_worksheet("Transformations")

    col_names = ["Transformation", "Transformation Type", "Num of Optimal Choices",
                 "Num of Optimal Choices for 2x2 Prob", "Num of Optimal Choices for 3x3 Prob",
                 "Problems"]
    col_widths = [15, 15, 15, 15, 15, 50]

    bold = workbook.add_format({'bold': True, 'text_wrap': True})

    for jj, col_name, width in zip(range(len(col_names)), col_names, col_widths):
        tran_worksheet.set_column(jj, jj, width)
        tran_worksheet.write(0, jj, col_name, bold)

    for ii, tran_d in zip(range(len(transformation_data_frame)), transformation_data_frame):
        tran_worksheet.write(ii + 1, 0, tran_d.get("tran_name"))
        tran_worksheet.write(ii + 1, 1, tran_d.get("tran_type"))
        tran_worksheet.write(ii + 1, 2, tran_d.get("n_optimal"))
        tran_worksheet.write(ii + 1, 3, tran_d.get("n_optimal_2x2"))
        tran_worksheet.write(ii + 1, 4, tran_d.get("n_optimal_3x3"))
        tran_worksheet.write(ii + 1, 5, tran_d.get("problems"))


def get_data_frame(problems):
    problem_data_frame = []
    analogy_data_frame = []
    transformation_data_frame = []

    for prob in problems:

        prob_data = prob.data

        d_result = {"problem_name": prob.name, "truth": prob.answer}

        if (3, 3) == prob.matrix.shape:
            d_result["matrix_type"] = "3x3"
        elif (2, 2) == prob.matrix.shape:
            d_result["matrix_type"] = "2x2"
        else:
            raise Exception("Crap!")

        prediction = prob_data.get("argmax_sim_predicted_ops")
        prediction_sim = prob_data.get("sim_predicted")
        winning_anlg = prob_data.get("best_analog_name")
        winning_tran = str(prob_data.get("best_tran"))
        win_with_sim = prob_data.get("best_sim")
        win_unary_or_binary = prob_data.get("best_analog_type")

        d_result["prediction"] = prediction
        d_result["prediction_sim"] = prediction_sim
        d_result["win_unary_or_binary"] = win_unary_or_binary
        d_result["winning_anlg"] = winning_anlg
        d_result["winning_tran"] = winning_tran
        d_result["win_with_sim"] = win_with_sim

        problem_data_frame.append(d_result)

    for anlg_name in unary_analogies_2by2.keys():
        d = {"analogy_name": anlg_name,
             "analogy_type": "unary",
             "matrix_type": "2x2",
             "n_optimal": 0,
             "n_optimal_2x2": 0,
             "n_optimal_3x3": 0,
             "problems": ""}
        for prob_d in problem_data_frame:
            if prob_d.get("winning_anlg") == anlg_name:
                d["problems"] = d["problems"] + " " + prob_d.get("problem_name")
                d["n_optimal"] = d["n_optimal"] + 1
                if "2x2" == prob_d.get("matrix_type"):
                    d["n_optimal_2x2"] = d["n_optimal_2x2"] + 1
                elif "3x3" == prob_d.get("matrix_type"):
                    d["n_optimal_3x3"] = d["n_optimal_3x3"] + 1
                else:
                    raise Exception("Crap!")
        analogy_data_frame.append(d)

    for anlg_name in unary_analogies_3by3:
        d = {"analogy_name": anlg_name,
             "analogy_type": "unary",
             "matrix_type": "3x3",
             "n_optimal": 0,
             "n_optimal_2x2": 0,
             "n_optimal_3x3": 0,
             "problems": ""}
        for prob_d in problem_data_frame:
            if prob_d.get("winning_anlg") == anlg_name:
                d["problems"] = d["problems"] + " " + prob_d.get("problem_name")
                d["n_optimal"] = d["n_optimal"] + 1
                if "2x2" == prob_d.get("matrix_type"):
                    d["n_optimal_2x2"] = d["n_optimal_2x2"] + 1
                elif "3x3" == prob_d.get("matrix_type"):
                    d["n_optimal_3x3"] = d["n_optimal_3x3"] + 1
                else:
                    raise Exception("Crap!")
        analogy_data_frame.append(d)

    for anlg_name in binary_analogies_3by3:
        d = {"analogy_name": anlg_name,
             "analogy_type": "binary",
             "matrix_type": "3x3",
             "n_optimal": 0,
             "n_optimal_2x2": 0,
             "n_optimal_3x3": 0,
             "problems": ""}
        for prob_d in problem_data_frame:
            if prob_d.get("winning_anlg") == anlg_name:
                d["problems"] = d["problems"] + " " + prob_d.get("problem_name")
                d["n_optimal"] = d["n_optimal"] + 1
                if "2x2" == prob_d.get("matrix_type"):
                    d["n_optimal_2x2"] = d["n_optimal_2x2"] + 1
                elif "3x3" == prob_d.get("matrix_type"):
                    d["n_optimal_3x3"] = d["n_optimal_3x3"] + 1
                else:
                    raise Exception("Crap!")
        analogy_data_frame.append(d)

    for tran in unary_transformations:
        d = {"tran_name": str(tran),
             "tran_type": "unary",
             "n_optimal": 0,
             "n_optimal_2x2": 0,
             "n_optimal_3x3": 0,
             "problems": ""}
        for prob_d in problem_data_frame:
            if prob_d.get("winning_tran") == str(tran):
                d["problems"] = d["problems"] + " " + prob_d.get("problem_name")
                d["n_optimal"] = d["n_optimal"] + 1
                if "2x2" == prob_d.get("matrix_type"):
                    d["n_optimal_2x2"] = d["n_optimal_2x2"] + 1
                elif "3x3" == prob_d.get("matrix_type"):
                    d["n_optimal_3x3"] = d["n_optimal_3x3"] + 1
                else:
                    raise Exception("Crap!")
        transformation_data_frame.append(d)

    for tran in binary_transformations:
        d = {"tran_name": str(tran),
             "tran_type": "binary",
             "n_optimal": 0,
             "n_optimal_2x2": 0,
             "n_optimal_3x3": 0,
             "problems": ""}
        for prob_d in problem_data_frame:
            if prob_d.get("winning_tran") == str(tran):
                d["problems"] = d["problems"] + " " + prob_d.get("problem_name")
                d["n_optimal"] = d["n_optimal"] + 1
                if "2x2" == prob_d.get("matrix_type"):
                    d["n_optimal_2x2"] = d["n_optimal_2x2"] + 1
                elif "3x3" == prob_d.get("matrix_type"):
                    d["n_optimal_3x3"] = d["n_optimal_3x3"] + 1
                else:
                    raise Exception("Crap!")
        transformation_data_frame.append(d)

    return problem_data_frame, analogy_data_frame, transformation_data_frame
