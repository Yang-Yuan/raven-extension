
unary_analogies_2by2 = [
    {"name": "A : B  ::  C : ?", "value": [(0, 0), (0, 1), (1, 0)], "type": "unary_2x2"},
    {"name": "A : C  ::  B : ?", "value": [(0, 0), (1, 0), (0, 1)], "type": "unary_2x2"}
]

unary_analogies_3by3 = [
    {"name": "A : B  ::  H : ?", "value": [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1), (1, 2), (2, 1)], "type": "unary_3x3"},  # row and column analogies
    {"name": "B : C  ::  H : ?", "value": [(0, 1), (0, 2), (1, 1), (1, 2), (1, 1), (1, 2), (2, 1)], "type": "unary_3x3"},
    {"name": "D : E  ::  H : ?", "value": [(1, 0), (1, 1), (2, 0), (2, 1), (1, 1), (1, 2), (2, 1)], "type": "unary_3x3"},
    {"name": "A : C  ::  G : ?", "value": [(0, 0), (0, 2), (1, 0), (1, 2), (1, 0), (1, 2), (2, 0)], "type": "unary_3x3"},
    {"name": "A : D  ::  F : ?", "value": [(0, 0), (1, 0), (0, 1), (1, 2), (1, 1), (2, 1), (1, 2)], "type": "unary_3x3"},
    {"name": "D : G  ::  F : ?", "value": [(1, 0), (2, 0), (1, 1), (2, 1), (1, 1), (2, 1), (1, 2)], "type": "unary_3x3"},
    {"name": "B : E  ::  F : ?", "value": [(0, 1), (1, 1), (0, 2), (1, 2), (1, 1), (2, 1), (1, 2)], "type": "unary_3x3"},
    {"name": "A : G  ::  C : ?", "value": [(0, 0), (2, 0), (0, 1), (2, 1), (0, 1), (2, 1), (0, 2)], "type": "unary_3x3"}
]

binary_analogies_3by3 = [
    {"name": "A : B : C  ::  G : H : ?", "value": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)], "type": "binary_3x3"},  # row and column analogies
    {"name": "A : D : G  ::  C : F : ?", "value": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2)], "type": "binary_3x3"},
]


all_anlgs = unary_analogies_2by2 + unary_analogies_3by3 + binary_analogies_3by3


def get_anlg(anlg_name):
    for anlg in all_anlgs:
        if anlg_name == anlg.get("name"):
            return anlg


