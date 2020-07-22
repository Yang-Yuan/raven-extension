import run
import strategy

# M-confident strategy
run.run(strategy.confident, prediction_score_name = "mato_score")

# M-neutral strategy
run.run(strategy.neutral, prediction_score_name = "mato_score")

# M-prudent strategy
run.run(strategy.prudent, prediction_score_name = "mato_score")

# O-confident strategy
run.run(strategy.confident, prediction_score_name = "optn_score")

# O-neutral strategy
run.run(strategy.neutral, prediction_score_name = "optn_score")

# O-prudent strategy
run.run(strategy.prudent, prediction_score_name = "optn_score")


