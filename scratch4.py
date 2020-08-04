import run
import strategy

# run.run(strategy.prudent,
#         prediction_score_name = "mato_score",
#         test_name = "ace")

# M-prudent strategy
run.run(strategy.prudent,
        prediction_score_name = "mato_score",
        test_name = "ace",
        test_trans = ["WWW", "XXX", "YYY", "ZZZ", "ZZ", "duplicate_new", "shape_texture_transfer",
                      "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
                      "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],
        test_anlgs = ["A:B::C:?"])

# test_trans = ["WWW", "XXX", "YYY", "ZZZ", "ZZ", "duplicate_new", "shape_texture_transfer",
#               "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
#               "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],
