import problem
import analogy_new
import transform

problems = problem.load_problems()


def get_probs(test_problems):
    if test_problems is None:
        return problems
    else:
        return [prob for prob in problems if prob.name in test_problems]


def get_anlgs(prob):
    if "2x2" in prob.type:
        return analogy_new.unary_2x2
    elif "2x3" in prob.type:
        return analogy_new.binary_2x3
    elif "3x3" in prob.type:
        return analogy_new.unary_3x3 + analogy_new.binary_3x3
    else:
        raise Exception("Ryan!")


def get_trans(prob, anlg):
    if "unary" in anlg.get("type"):
        return transform.unary_transformations
    elif "binary" in anlg.get("type"):
        return transform.binary_transformations
    else:
        raise Exception("Ryan don't forget anlg")


def get_anlg_tran_pairs(prob):
    pairs = []
    anlgs = get_anlgs(prob)
    for anlg in anlgs:
        trans = get_trans(prob, anlg)
        for tran in trans:
            if is_valid(anlg, tran):
                pairs.append({"anlg": anlg, "tran": tran})

    return pairs


def is_valid(anlg, tran):

    valid = True

    if "mirror" == tran.get("name") or "mirror_rot_180" == tran.get("name"):

        value = anlg.get("value")
        for ii, u in enumerate(value):
            for jj, v in enumerate(value):
                if u == v and (ii % 2) != (jj % 2):
                    valid = False

    return valid

