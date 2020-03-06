import xlsxwriter
from datetime import datetime
from os.path import join
from analogy import unary_analogies_2by2, unary_analogies_3by3, binary_analogies_3by3
from transform import unary_transformations, binary_transformations
import pandas as pd
import numpy as np

report_folder = "reports/"

translations = {
    "prob_name": "Problem",
    "prob_ansr": "Truth",
    "optn": "Prediction",
    "pato_score": "Prediction Score",
    "prob_type": "Problem Type",
    "anlg_name": "Analogy",
    "anlg_type": "Analogy Type",
    "tran_name": "Transformation",
    "tran_type": "Transformation Type",
    "pat_score": "Explanation Score",
    "anlg_n": "# of Hits",
    "tran_n": "# of Hits",
    "crct_probs": "Correctly Answered",
    "incr_probs": "Incorrectly Answered",
    "crct/incr": "Correct v.s. Incorrect"
}

sltn_cols = ["prob_name", "prob_type", "prob_ansr", "optn", "pato_score",
             "anlg_name", "anlg_type", "tran_name", "tran_type", "pat_score"]
sltn_hdrs = [translations.get(col) for col in sltn_cols]
sltn_col_widths = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

sltn_anlg_cols = ["anlg_name", "anlg_type", "anlg_n", "crct_probs", "incr_probs", "crct/incr"]
sltn_anlg_hdrs = [translations.get(col) for col in sltn_anlg_cols]
sltn_anlg_col_widths = [15, 15, 15, 30, 30, 15]

sltn_tran_cols = ["tran_name", "tran_type", "tran_n", "crct_probs", "incr_probs", "crct/incr"]
sltn_tran_hdrs = [translations.get(col) for col in sltn_tran_cols]
sltn_tran_col_widths = [15, 15, 15, 30, 30, 15]


def create_report(probs, prefix):

    file_name = prefix + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".xlsx"

    _, _, _, sltn_df, sltn_anlg_df, sltn_tran_df = extract_data(probs)

    with pd.ExcelWriter(file_name) as writer:
        sltn_df.to_excel(writer, sheet_name = 'sltn_sheet', header = sltn_hdrs, index = False)
        sltn_sheet = writer.sheets["sltn_sheet"]
        for jj, w in enumerate(sltn_col_widths):
            sltn_sheet.set_column(jj, jj, w)

        sltn_anlg_df.to_excel(writer, sheet_name = 'sltn_anlg_sheet', index_label = sltn_anlg_hdrs[0],
                              header = sltn_anlg_hdrs[1:])
        sltn_anlg_sheet = writer.sheets["sltn_anlg_sheet"]
        for jj, w in enumerate(sltn_col_widths):
            sltn_anlg_sheet.set_column(jj, jj, w)

        sltn_tran_df.to_excel(writer, sheet_name = "sltn_tran_sheet", index_label = sltn_tran_hdrs[0],
                              header = sltn_tran_hdrs[1:])
        sltn_tran_sheet = writer.sheets["sltn_tran_sheet"]
        for jj, w in enumerate(sltn_col_widths):
            sltn_tran_sheet.set_column(jj, jj, w)


def extract_data(probs):

    global sltn_cols

    anlg_tran_data = []
    anlg_data = []
    pred_data = []
    sltn_data = []
    for prob in probs:
        aggregation_progression = prob.data
        anlg_tran_data.extend(aggregation_progression.get("anlg_tran_data"))
        anlg_data.extend(aggregation_progression.get("anlg_data"))
        pred_data.extend(aggregation_progression.get("pred_data"))
        sltn_data.append(aggregation_progression.get("pred_d"))

    anlg_tran_df = pd.DataFrame(data = anlg_tran_data)
    anlg_df = pd.DataFrame(data = anlg_data)
    pred_df = pd.DataFrame(data = pred_data)
    sltn_df = pd.DataFrame(data = sltn_data, columns = sltn_cols)

    sltn_anlg_df = sltn_df.groupby("anlg_name").apply(sltn2anlg)
    sltn_tran_df = sltn_df.groupby("tran_name").apply(sltn2tran)

    return anlg_tran_df, anlg_df, pred_df, sltn_df, sltn_anlg_df, sltn_tran_df


def create_worksheet(workbook, data):
    prob_worksheet = workbook.add_worksheet("Problems")


def sltn2anlg(anlg_group):
    correct_ones = anlg_group.loc[anlg_group["prob_ansr"] == anlg_group["optn"]]
    incorrect_ones = anlg_group.loc[anlg_group["prob_ansr"] != anlg_group["optn"]]
    d = {
        "anlg_type": anlg_group["anlg_type"].iloc[0],
        "anlg_n": len(anlg_group),  # row number
        "crct_probs": ','.join(correct_ones["prob_name"].to_list()),
        "incr_probs": ','.join(incorrect_ones["prob_name"].to_list()),
        "crct/incr": str(len(correct_ones)) + '/' + str(len(incorrect_ones))
    }
    return pd.Series(d)


def sltn2tran(tran_group):
    correct_ones = tran_group.loc[tran_group["prob_ansr"] == tran_group["optn"]]
    incorrect_ones = tran_group.loc[tran_group["prob_ansr"] != tran_group["optn"]]
    d = {
        "tran_type": tran_group["tran_type"].iloc[0],
        "tran_n": len(tran_group),
        "crct_probs": ','.join(correct_ones["prob_name"].to_list()),
        "incr_probs": ','.join(incorrect_ones["prob_name"].to_list()),
        "crct/incr": str(len(correct_ones)) + '/' + str(len(incorrect_ones))
    }
    return pd.Series(d)