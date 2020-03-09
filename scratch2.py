import pandas as pd
import numpy as np
import report
import copy


pd.set_option('display.max_columns', 500)

test_report = np.load("test_report.npz", allow_pickle = True)
probs = test_report["probs"].tolist()
probs.append(copy.deepcopy(probs[0]))
probs.append(copy.deepcopy(probs[1]))
probs[2].data.get("pred_d")["prob_name"] = "d3"
probs[3].data.get("pred_d")["prob_name"] = "e3"

report.create_report(probs, "mode_")

x = [{"a": "a1", "b": 1, "c": 1},
{"a": "a2", "b": 1, "c": 1},
{"a": "a3", "b": 1, "c": 1},
{"a": "a4", "b": 1, "c": 1},
{"a": "a5", "b": 1, "c": 1},
{"a": "a11", "b": 1, "c": 1},
]

