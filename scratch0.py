import runn_raven
import analogy
import transform

# (1) using all analogies and all transformations
analogy_groups = {
    "2x2_unary_analogies": analogy.unary_analogies_2by2,
    "2x2_binary_analogies": {},
    "3x3_unary_analogies": analogy.unary_analogies_3by3,
    "3x3_binary_analogies": analogy.binary_analogies_3by3
}

transformation_groups = {
    "2x2_unary_transformations": transform.unary_transformations,
    "2x2_binary_transformations": [],
    "3x3_unary_transformations": transform.unary_transformations,
    "3x3_binary_transformations": transform.binary_transformations
}

runn_raven.run_raven(mode = "explanatory",
                     analogy_groups = analogy_groups,
                     transformation_groups = transformation_groups,
                     test_problems = ["c4"])

# runn_raven.run_raven(mode = "greedy",
#                      analogy_groups = analogy_groups,
#                      transformation_groups = transformation_groups)

# runn_raven.run_raven(mode = "brutal",
#                      analogy_groups = analogy_groups,
#                      transformation_groups = transformation_groups)
