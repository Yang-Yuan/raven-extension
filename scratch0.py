import run_raven_new
import strategy

# run_raven_new.run_raven(strategy.confident)
#
# run_raven_new.run_raven(strategy.neutral)
#
# run_raven_new.run_raven(strategy.prudent)

run_raven_new.run_raven(strategy.confident, prediction_score_name = "optn_score")

run_raven_new.run_raven(strategy.neutral,  prediction_score_name = "optn_score")

run_raven_new.run_raven(strategy.prudent,  prediction_score_name = "optn_score")




