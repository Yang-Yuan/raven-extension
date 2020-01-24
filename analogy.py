# TODO deliberately exclude the diagonal analogies

unary_analogies_2by2 = {
    "A : B  ::  C : ?": [(0, 0), (0, 1), (1, 0)],
    "A : C  ::  B : ?": [(0, 0), (1, 0), (0, 1)]
}

unary_analogies_3by3 = {
    "A : B  ::  H : ?": [(0, 0), (0, 1), (2, 1)],
    "B : C  ::  H : ?": [(0, 1), (0, 2), (2, 1)],
    "D : E  ::  H : ?": [(1, 0), (1, 1), (2, 1)],
    "E : F  ::  H : ?": [(1, 1), (1, 2), (2, 1)],
    "G : H  ::  H : ?": [(2, 0), (2, 1), (2, 1)],
    "A : C  ::  G : ?": [(0, 0), (0, 2), (2, 0)],
    "D : F  ::  G : ?": [(1, 0), (1, 2), (2, 0)],
    "A : D  ::  F : ?": [(0, 0), (1, 0), (1, 2)],
    "D : G  ::  F : ?": [(1, 0), (2, 0), (1, 2)],
    "B : E  ::  F : ?": [(0, 1), (1, 1), (1, 2)],
    "E : H  ::  F : ?": [(1, 1), (2, 1), (1, 2)],
    "C : F  ::  F : ?": [(0, 2), (1, 2), (1, 2)],
    "A : G  ::  C : ?": [(0, 0), (2, 0), (0, 2)],
    "B : H  ::  C : ?": [(0, 1), (2, 1), (0, 2)]
}

binary_analogies_3by3 = {
    "A : B : C  ::  G : H : ?": [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1)],
    "D : E : F  ::  G : H : ?": [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
    "A : D : G  ::  C : F : ?": [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2)],
    "B : E : H  ::  C : F : ?": [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2)]
}


def get_analogies(problem):
    if 2 == problem.matrix_n:
        return {"unary_analogies": unary_analogies_2by2,
                "binary_analogies": {}}
    elif 3 == problem.matrix_n:
        return {"unary_analogies": unary_analogies_3by3,
                "binary_analogies": binary_analogies_3by3}
    else:
        raise Exception("Crap!")
