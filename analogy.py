# TODO deliberately exclude the diagonal analogies

unary_analogies_2by2 = [
    {"name": "A : B  ::  C : ?", "value": [(0, 0), (0, 1), (1, 0)]},
    {"name": "A : C  ::  B : ?", "value": [(0, 0), (1, 0), (0, 1)]}
]

unary_analogies_3by3 = [
    {"name": "A : B  ::  H : ?", "value": [(0, 0), (0, 1), (2, 1)]},  # row and column analogies
    {"name": "B : C  ::  H : ?", "value": [(0, 1), (0, 2), (2, 1)]},
    # {"name": "D : E  ::  H : ?", "value": [(1, 0), (1, 1), (2, 1)]},
    # {"name": "E : F  ::  H : ?", "value": [(1, 1), (1, 2), (2, 1)]},
    # {"name": "G : H  ::  H : ?", "value": [(2, 0), (2, 1), (2, 1)]},
    # {"name": "A : C  ::  G : ?", "value": [(0, 0), (0, 2), (2, 0)]},
    # {"name": "D : F  ::  G : ?", "value": [(1, 0), (1, 2), (2, 0)]},
    # {"name": "A : D  ::  F : ?", "value": [(0, 0), (1, 0), (1, 2)]},
    # {"name": "D : G  ::  F : ?", "value": [(1, 0), (2, 0), (1, 2)]},
    # {"name": "B : E  ::  F : ?", "value": [(0, 1), (1, 1), (1, 2)]},
    # {"name": "E : H  ::  F : ?", "value": [(1, 1), (2, 1), (1, 2)]},
    # {"name": "C : F  ::  F : ?", "value": [(0, 2), (1, 2), (1, 2)]},
    # {"name": "A : G  ::  C : ?", "value": [(0, 0), (2, 0), (0, 2)]},
    # {"name": "B : H  ::  C : ?", "value": [(0, 1), (2, 1), (0, 2)]},
    # {"name": "F : G  ::  E : ?", "value": [(1, 2), (2, 0), (1, 1)]},  # diagonal analogies
    # {"name": "G : B  ::  E : ?", "value": [(2, 0), (0, 1), (1, 1)]},
    # {"name": "H : C  ::  E : ?", "value": [(2, 1), (0, 2), (1, 1)]},
    # {"name": "C : D  ::  E : ?", "value": [(0, 2), (1, 0), (1, 1)]},
    # {"name": "A : E  ::  E : ?", "value": [(0, 0), (1, 1), (1, 1)]},
    # {"name": "F : B  ::  A : ?", "value": [(1, 2), (0, 1), (0, 0)]},
    # {"name": "H : D  ::  A : ?", "value": [(2, 1), (1, 0), (0, 0)]},
    # {"name": "F : H  ::  D : ?", "value": [(1, 2), (2, 1), (1, 0)]},
    # {"name": "H : A  ::  D : ?", "value": [(2, 1), (0, 0), (1, 0)]},
    # {"name": "G : C  ::  D : ?", "value": [(2, 0), (0, 2), (1, 0)]},
    # {"name": "C : E  ::  D : ?", "value": [(0, 2), (1, 1), (1, 0)]},
    # {"name": "B : D  ::  D : ?", "value": [(0, 1), (1, 0), (1, 0)]},
    # {"name": "F : A  ::  B : ?", "value": [(1, 2), (0, 0), (0, 1)]},
    # {"name": "G : E  ::  B : ?", "value": [(2, 0), (1, 1), (0, 1)]}
]

binary_analogies_3by3 = [
    {"name": "A : B : C  ::  G : H : ?", "value": [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1)]},  # row and column analogies
    {"name": "D : E : F  ::  G : H : ?", "value": [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]},
    # {"name": "A : D : G  ::  C : F : ?", "value": [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2)]},
    # {"name": "B : E : H  ::  C : F : ?", "value": [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2)]},
    # {"name": "F : G : B  ::  A : E : ?", "value": [(1, 2), (2, 0), (0, 1), (0, 0), (1, 1)]},  # diagonal analogies
    # {"name": "H : C : D  ::  A : E : ?", "value": [(2, 1), (0, 2), (1, 0), (0, 0), (1, 1)]},
    # {"name": "F : H : A  ::  B : D : ?", "value": [(1, 2), (2, 1), (0, 0), (0, 1), (1, 0)]},
    # {"name": "G : C : E  ::  B : D : ?", "value": [(2, 0), (0, 2), (1, 1), (0, 1), (1, 0)]},
]

all_anlgs = unary_analogies_2by2 + unary_analogies_3by3 + binary_analogies_3by3


def get_anlg(anlg_name):
    for anlg in all_anlgs:
        if anlg_name == anlg.get("name"):
            return anlg


