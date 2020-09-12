import run
import strategy

run.run(strategy.prudent,
        prediction_score_name = "mato_score",
        test_name = "ace",
        test_trans = ["identity_shape_loc_isomorphism"],
        # test_trans = ["shape_topo_mapping",
        #               "shape_delta_loc_isomorphism",
        #               "topo_delta_shape_isomorphism",
        #               "shape_loc_isomorphism",
        #               "identity_shape_loc_isomorphism",
        #               "shape_texture_transfer",
        #               "duplicate_new",
        #               "rot_45", "rot_90", "rot_135", "rot_180", "rot_225", "rot_270", "rot_315",
        #               "mirror", "mirror_rot_90", "mirror_rot_180", "mirror_rot_270"],
        test_anlgs = ["A:B::C:?"],
        test_problems = ["m15"])
