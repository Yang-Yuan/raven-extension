import run
import strategy

# run.run(strategy.prudent,
#         prediction_score_name = "mato_score",
#         test_name = "ace")

# M-prudent strategy
run.run(strategy.prudent,
        prediction_score_name = "mato_score",
        test_name = "ace",
        test_trans = ["YYY"],
        test_anlgs = ["A:B::C:?"],
        test_problems = ["m1"])

