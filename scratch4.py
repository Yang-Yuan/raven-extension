import run
import strategy

# M-prudent strategy
run.run(strategy.prudent,
        test_problems = ["m5"],
        prediction_score_name = "mato_score",
        test_name = "ace",
        test_anlgs = ["A:B::C:?"],
        test_trans = ["XXX"])
