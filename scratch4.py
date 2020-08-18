import run
import strategy

# run.run(strategy.prudent,
#         prediction_score_name = "mato_score",
#         test_name = "ace")

# M-prudent strategy
# run.run(strategy.prudent,
#         prediction_score_name = "mato_score",
#         test_name = "ace",
#         test_trans = ["WWW", "XXX", "YYY", "ZZZ", "ZZ", "duplicate_new", "shape_texture_transfer",
#                       "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
#                       "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],
#         test_anlgs = ["A:B::C:?"])

# test_trans = ["shape_topo_mapping", "shape_loc_isomorphism", "shape_delta_loc_isomorphism", "topo_delta_shape_isomorphism",
#               "duplicate_new", "shape_texture_transfer",
#               "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
#               "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],

run.run(strategy.prudent,
        prediction_score_name = "mato_score",
        test_name = "ace",
        test_trans = ["shape_topo_mapping", "shape_loc_isomorphism", "shape_delta_loc_isomorphism",
                      "topo_delta_shape_isomorphism", "identity_shape_loc_isomorphism"
                      "duplicate_new", "shape_texture_transfer",
                      "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
                      "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],
        test_anlgs = ["A:B::C:?"])

# test_trans = ["shape_topo_mapping", "shape_loc_isomorphism", "shape_delta_loc_isomorphism",
#               "topo_delta_shape_isomorphism", "identity_shape_loc_isomorphism"
#               "duplicate_new", "shape_texture_transfer",
#               "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
#               "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],

# shape_delta_shape
