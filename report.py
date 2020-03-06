import xlsxwriter
from datetime import datetime
from os.path import join
from analogy import unary_analogies_2by2, unary_analogies_3by3, binary_analogies_3by3
from transform import unary_transformations, binary_transformations

report_folder = "./reports/"


def create_report(probs, prefix):

    file_name = prefix + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".xlsx"
    workbook = xlsxwriter.Workbook(join(report_folder, file_name))

    anlg_tran_data, anlg_data, pred_data, sltn_data = extract_data(probs)



def extract_data(probs):

    anlg_tran_data = []
    anlg_data = []
    pred_data = []
    sltn_data = []
    for prob in probs:
        aggregation_progression = prob.data
        anlg_tran_data.extend(aggregation_progression.get())
        anlg_data.extend(aggregation_progression.get())
        pred_data.extend(aggregation_progression.get())
        sltn_data.append(aggregation_progression.get())

    return anlg_tran_data, anlg_data, pred_data, sltn_data


def create_worksheet(workbook, data):
    prob_worksheet = workbook.add_worksheet("Problems")