import problem
import analogy_new
import transform

problems = problem.load_problems()

analogy_groups = {
    "2x2_unary_analogies": analogy_new.unary_analogies_2by2,
    "2x2_binary_analogies": {},
    "3x3_unary_analogies": analogy_new.unary_analogies_3by3,
    "3x3_binary_analogies": analogy_new.binary_analogies_3by3
}

transformation_groups = {
    "2x2_unary_transformations": transform.unary_transformations,
    "2x2_binary_transformations": [],
    "3x3_unary_transformations": transform.unary_transformations,
    "3x3_binary_transformations": transform.binary_transformations
}


def get_probs(test_problems):
    if test_problems is None:
        return problems
    else:
        return [prob for prob in problems if prob.name in test_problems]


def get_anlgs(prob):
    if 2 == prob.matrix_n:
        return analogy_groups.get("2x2_unary_analogies")
    elif 3 == prob.matrix_n:
        return analogy_groups.get("3x3_unary_analogies") + analogy_groups.get("3x3_binary_analogies")
    else:
        raise Exception("Ryan!")


def get_trans(prob, anlg):
    if 2 == prob.matrix_n:
        return transformation_groups.get("2x2_unary_transformations")
    elif 3 == prob.matrix_n:
        if 3 == len(anlg.get("value")):
            return transformation_groups.get("3x3_unary_transformations")
        elif 5 == len(anlg.get("value")):
            return transformation_groups.get("3x3_binary_transformations")
    else:
        raise Exception("Ryan don't forget anlg")


def get_anlg_tran_pairs(prob):
    pairs = []
    anlgs = get_anlgs(prob)
    for anlg in anlgs:
        trans = get_trans(prob, anlg)
        for tran in trans:
            pairs.append({"anlg": anlg, "tran": tran})

    return pairs
